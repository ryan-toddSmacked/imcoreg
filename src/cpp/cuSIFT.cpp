
#include "cuSIFT.hpp"

#include <chrono>

#include "cudaImage.h"
#include "cudaSift.h"
#include "Logger.hpp"
#include "cudautils.h"
#include "geomFuncs_blas.hpp"

inline void pHomo(float *H)
{
    float val;
    Logger::cerr("Homography:\n");
    for (int i = 0; i < 3; i++)
    {
        Logger::cerr("\t|");
        for (int j = 0; j < 3; j++)
        {
            val = H[i * 3 + j];
            if (val == 0.0f) val = 0.0f;
            if (val < 0.0f) fprintf(stderr, " %.9e", val);
            else fprintf(stderr, "  %.9e", val);    
        }
        fprintf(stderr, " |\n");
    }
}

int cudaSift(
    const float *h_img1, int w1, int h1,
    const float *h_img2, int w2, int h2,
    std::vector<float> &homography,
    float *inlier_ratio,
    float **d_img1, float **d_img2,
    int num_features, float initBlur, float thresh, float lowestScale)
{
    std::chrono::high_resolution_clock::time_point tstart, tend;

    tstart = std::chrono::high_resolution_clock::now();

    Logger::cerr("========================================\n");
    homography.resize(9, 0.0f);

    Logger::cerr("Initializing cudaSift\n");
    InitCuda();

    std::chrono::high_resolution_clock::time_point t1, t2;
    std::chrono::duration<double> time_span;
    double elapsed_time;

    const unsigned int max_width = (w1 > w2) ? w1 : w2;
    const unsigned int max_height = (h1 > h2) ? h1 : h2;

    int find_homography_num_loops;
    int num_matches;
    int num_inliers;

    SiftData siftData1, siftData2;  // Sift data for the two images, these do not have destructor, so they need to be freed manually.
    bool free_sift_data1 = false, free_sift_data2 = false;  // Flags to check if the sift data needs to be freed, before returning from the function.
    CudaImage img1, img2;
    SiftTempMem tempMem;


    // Allocate memory for the images on the GPU
    Logger::cerr("Allocating memory for CUDA images\n");
    try
    {
        t1 = std::chrono::high_resolution_clock::now();
        img1.Allocate(w1, h1, iAlignUp(w1, 128), false, NULL, h_img1);
        img2.Allocate(w2, h2, iAlignUp(w2, 128), false, NULL, h_img2);
        t2 = std::chrono::high_resolution_clock::now();
    }
    catch(const std::exception& e)
    {
        Logger::error("Could not allocate memory for CUDA images: %s\n", e.what());
        if (free_sift_data1) FreeSiftData(siftData1);   // I think the compiler will optimize this out...
        if (free_sift_data2) FreeSiftData(siftData2);
        return CUSIFT_ERROR_ALLOCATING_DEVICE_MEMORY;
    }
    time_span = (t2 - t1);
    Logger::cerr("Memory allocated for CUDA images (%.6lf s)\n", time_span.count());


    // Download the images to the device
    Logger::cerr("Downloading CUDA images to device\n");
    try
    {
        elapsed_time = img1.Download();
        elapsed_time += img2.Download();
    }
    catch(const std::exception& e)
    {
        Logger::error("Could not download CUDA images to device: %s\n", e.what());
        if (free_sift_data1) FreeSiftData(siftData1);   // I think the compiler will optimize this out...
        if (free_sift_data2) FreeSiftData(siftData2);
        return CUSIFT_ERROR_COPYING_TO_DEVICE;
    }
    Logger::cerr("CUDA images downloaded to device (%.6lf s)\n", elapsed_time / 1000.0);


    // Initialize SiftData for the two images
    Logger::cerr("Initializing SiftData for the two images\n");
    try
    {
        t1 = std::chrono::high_resolution_clock::now();
        InitSiftData(siftData1, num_features, true, true);
        free_sift_data1 = true; // Set the flag to true, so that the data is freed before returning from the function.
        InitSiftData(siftData2, num_features, true, true);
        free_sift_data2 = true; // Set the flag to true, so that the data is freed before returning from the function.
        t2 = std::chrono::high_resolution_clock::now();
    }
    catch(const std::exception& e)
    {
        Logger::error("Could not initialize SiftData: %s\n", e.what());
        if (free_sift_data1) FreeSiftData(siftData1);   // I think the compiler will optimize this out...
        if (free_sift_data2) FreeSiftData(siftData2);
        return CUSIFT_ERROR_ALLOCATING_DEVICE_MEMORY;
    }
    time_span = (t2 - t1);
    Logger::cerr("SiftData initialized (%.6lf s)\n", time_span.count());


    // Allocate temporary memory for Sift extraction
    Logger::cerr("Allocating temporary memory for Sift extraction\n");
    try
    {
        t1 = std::chrono::high_resolution_clock::now();
        tempMem.Allocate(max_width, max_height, 6);
        t2 = std::chrono::high_resolution_clock::now();
    }
    catch(const std::exception& e)
    {
        Logger::error("Could not allocate temporary memory for sift extraction: %s\n", e.what());
        if (free_sift_data1) FreeSiftData(siftData1);
        if (free_sift_data2) FreeSiftData(siftData2);
        return CUSIFT_ERROR_ALLOCATING_DEVICE_MEMORY;
    }
    time_span = (t2 - t1);
    Logger::cerr("Temporary memory allocated for Sift extraction (%.6lf s)\n", time_span.count());


    // Extract Sift features from the two images
    Logger::cerr("Extracting Sift features from the two images\n");
    try
    {
        elapsed_time = ExtractSift(siftData1, img1, 6, initBlur, thresh, lowestScale, tempMem.get_device_pointer());
    }
    catch(const std::exception& e)
    {
        Logger::error("Could not extract SiftData for image 1: %s\n", e.what());
        if (free_sift_data1) FreeSiftData(siftData1);
        if (free_sift_data2) FreeSiftData(siftData2);
        return CUSIFT_ERROR_EXTRACTING_FEATURES;
    }
    Logger::cerr("Sift features extracted from image 1 (%.6lf s)\n", elapsed_time / 1000.0);


    // Extract Sift features from the second image
    try
    {
        elapsed_time = ExtractSift(siftData2, img2, 6, initBlur, thresh, lowestScale, tempMem.get_device_pointer());
    }
    catch(const std::exception& e)
    {
        Logger::error("Could not extract SiftData for image 2: %s\n", e.what());
        if (free_sift_data1) FreeSiftData(siftData1);
        if (free_sift_data2) FreeSiftData(siftData2);
        return CUSIFT_ERROR_EXTRACTING_FEATURES;
    }
    Logger::cerr("Sift features extracted from image 2 (%.6lf s)\n", elapsed_time / 1000.0);
    Logger::cerr("Image1 SiftFeatures: %d\n", siftData1.numPts);
    Logger::cerr("Image2 SiftFeatures: %d\n", siftData2.numPts);


    // Match the Sift features from the two images
    Logger::cerr("Matching Sift features from the two images\n");
    try
    {
        elapsed_time = MatchSiftData(siftData1, siftData2);
    }
    catch(const std::exception& e)
    {
        Logger::error("Could not match SiftData beteen images: %s\n", e.what());
        if (free_sift_data1) FreeSiftData(siftData1);
        if (free_sift_data2) FreeSiftData(siftData2);
        return CUSIFT_ERROR_MATCHING_FEATURES;
    }
    Logger::cerr("Sift features matched (%.6lf s)\n", elapsed_time / 1000.0);


    // Set the number of loops for finding the homography, 10 * the maximum number of points.
    find_homography_num_loops = 10 * ((siftData1.numPts > siftData2.numPts) ? siftData1.numPts : siftData2.numPts);

    // Find the homography between the two images
    Logger::cerr("Finding the homography between the two images\n");
    try
    {
        elapsed_time = FindHomography(siftData1, homography.data(), &num_matches, find_homography_num_loops, 0.85f, 0.95f, 5.0f);
    }
    catch(const std::exception& e)
    {
        Logger::error("could not find homography: %s\n", e.what());
        if (free_sift_data1) FreeSiftData(siftData1);
        if (free_sift_data2) FreeSiftData(siftData2);
        return CUSIFT_ERROR_COMPUTING_HOMOGRAPHY;
    }
    Logger::cerr("Homography found (%.6lf s)\n", elapsed_time / 1000.0);
    Logger::cerr("Number of matches: %d\n", num_matches);

    // Improve the homography
    Logger::cerr("Improving the homography\n");
    try
    {

        t1 = std::chrono::high_resolution_clock::now();
        //num_inliers = ImproveHomography(siftData1, homography.data(), 5, 0.85f, 0.95f, 3.5f);
        num_inliers = ImproveHomography_Mat(siftData1, homography.data(), 5, 0.85f, 0.95f, 3.5f);
        t2 = std::chrono::high_resolution_clock::now();

        num_inliers = ImproveHomography_Mat(siftData1, homography.data(), 5, 0.85f, 0.95f, 3.5f);
        pHomo(homography.data());

    }
    catch(const std::exception& e)
    {
        Logger::error("Could not improve homography: %s\n", e.what());
        if (free_sift_data1) FreeSiftData(siftData1);
        if (free_sift_data2) FreeSiftData(siftData2);
        return CUSIFT_ERROR_IMPROVIDING_HOMOGRAPHY;
    }
    time_span = (t2 - t1);
    Logger::cerr("Homography improved (%.6lf s)\n", time_span.count());
    Logger::cerr("Number of inliers: %d\n", num_inliers);
    Logger::cerr("Inlier ratio: %.4lf%%\n", 100.0 * (num_inliers / (double)siftData1.numPts));
    
    if (inlier_ratio != nullptr)
    {
        *inlier_ratio = (float)num_inliers / std::min((float)siftData1.numPts, (float)siftData2.numPts);
    }

    if (d_img1 != nullptr)
    {
        float *d_img1_ptr = nullptr;
        try
        {
            safeCall(cudaMalloc(&d_img1_ptr, w1 * h1 * sizeof(float)));
        }
        catch(const std::exception& e)
        {
            Logger::error("Could not allocate memory for on device for return image1: %s\n", e.what());
            if (free_sift_data1) FreeSiftData(siftData1);
            if (free_sift_data2) FreeSiftData(siftData2);
            return CUSIFT_ERROR_ALLOCATING_DEVICE_MEMORY;
        }

        try
        {
            safeCall(cudaMemcpy2D(d_img1_ptr, w1 * sizeof(float), img1.d_data, img1.pitch * sizeof(float), w1 * sizeof(float), h1, cudaMemcpyDeviceToDevice));
        }
        catch(const std::exception& e)
        {
            Logger::error("Could not copy image1 to device: %s\n", e.what());
            cudaFree(d_img1_ptr);
            if (free_sift_data1) FreeSiftData(siftData1);
            if (free_sift_data2) FreeSiftData(siftData2);
            return CUSIFT_ERROR_COPYING_TO_DEVICE;
        }

        *d_img1 = d_img1_ptr;
    }

    if (d_img2 != nullptr)
    {
        float *d_img2_ptr = nullptr;
        try
        {
            safeCall(cudaMalloc(&d_img2_ptr, w2 * h2 * sizeof(float)));
        }
        catch(const std::exception& e)
        {
            Logger::error("Could not allocate memory for on device for return image2: %s\n", e.what());
            if (free_sift_data1) FreeSiftData(siftData1);
            if (free_sift_data2) FreeSiftData(siftData2);

            if (d_img1 != nullptr)
            {
                cudaFree(*d_img1);
                *d_img1 = nullptr;
            }
            return CUSIFT_ERROR_ALLOCATING_DEVICE_MEMORY;
        }

        try
        {
            safeCall(cudaMemcpy2D(d_img2_ptr, w2 * sizeof(float), img2.d_data, img2.pitch * sizeof(float), w2 * sizeof(float), h2, cudaMemcpyDeviceToDevice));
        }
        catch(const std::exception& e)
        {
            Logger::error("Could not copy image2 to device: %s\n", e.what());
            cudaFree(d_img2_ptr);
            if (free_sift_data1) FreeSiftData(siftData1);
            if (free_sift_data2) FreeSiftData(siftData2);

            if (d_img1 != nullptr)
            {
                cudaFree(*d_img1);
                *d_img1 = nullptr;
            }
            if (d_img2 != nullptr)
            {
                cudaFree(*d_img2);
                *d_img2 = nullptr;
            }
            return CUSIFT_ERROR_COPYING_TO_DEVICE;
        }

        *d_img2 = d_img2_ptr;
    }

    // Free the SiftData
    if (free_sift_data1) FreeSiftData(siftData1);
    if (free_sift_data2) FreeSiftData(siftData2);

    tend = std::chrono::high_resolution_clock::now();
    time_span = (tend - tstart);

    Logger::cerr("CudaSift Completed (%.6lfs)\n", time_span.count());
    Logger::cerr("========================================\n\n");

    return CUSIFT_ERROR_NONE;
}
