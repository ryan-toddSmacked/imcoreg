#include "cuSIFT.hpp"
#include "imwback.h"


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>
#include <iostream>
#include <cstdarg>
#include <cuda_runtime.h>

void mexErrMsgIdAndTxt(const char *id, const char *msg)
{
    std::cerr << id << ": " << msg << std::endl;
    exit(1);
}

void mexPrintf(const char *msg, ...)
{
    va_list args;
    va_start(args, msg);
    vprintf(msg, args);
    va_end(args);
}

int main()
{
    size_t freeMem, totalMem;
    size_t freeMem2, totalMem2;

    cudaMemGetInfo(&freeMem, &totalMem);
    mexPrintf("Free memory: %lu\n", freeMem);

    const char *img1_path = "../test/im1.png";
    const char *img2_path = "../test/im2.png";

    cv::Mat h_img1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
    cv::Mat h_img2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);

    h_img1.convertTo(h_img1, CV_32FC1);
    h_img2.convertTo(h_img2, CV_32FC1);

    float *img1, *img2;
    int img1_rows, img1_cols, img2_rows, img2_cols;

    img1 = h_img1.ptr<float>();
    img1_cols = h_img1.cols;
    img1_rows = h_img1.rows;

    img2 = h_img2.ptr<float>();
    img2_cols = h_img2.cols;
    img2_rows = h_img2.rows;

    // These are device pointers, and will be allocated in the GPU
    float *d_img1, *d_img2;
    std::vector<float> homography;
    float inlier_ratio;

    // Variable arguments
    int numFeatures = 4000;
    float initBlur = 1.0f;
    float thresh = 3.0f;
    float lowestScale = 0.0f;

    // Call the CUDA SIFT function
    // Swap cols and rows, because the image is in column-major order
    int result = cudaSift(
        img1, img1_cols, img1_rows,
        img2, img2_cols, img2_rows,
        homography, &inlier_ratio,
        &d_img1, &d_img2,
        numFeatures, initBlur, thresh, lowestScale);

    if (result != CUSIFT_ERROR_NONE)
    {
        mexErrMsgIdAndTxt("imcoregmx:cudaSiftError", "Error in cudaSift function");
    }

    // Get the meshgrid
    float *d_X, *d_Y;
    int rows, cols;
    int result2 = imwback_get_meshgrid(&d_X, &d_Y, &rows, &cols, homography.data(), img1_rows, img1_cols, img2_rows, img2_cols);
    if (result2 != 0)
    {
        cudaFree((void*)d_img1);
        cudaFree((void*)d_img2);
        cudaDeviceReset();
        mexErrMsgIdAndTxt("imcoregmx:imwbackError", "Error in imwback_get_meshgrid function");
    }

    float *img1p, *img2p;
    int result3 = imwback_warpImage(d_img1, img1_cols, img1_rows, &img1p, cols, rows, d_X, d_Y);
    if (result3 != 0)
    {
        cudaFree((void*)d_img1);
        cudaFree((void*)d_img2);
        cudaFree((void*)d_X);
        cudaFree((void*)d_Y);
        cudaDeviceReset();
        mexErrMsgIdAndTxt("imcoregmx:imwbackError", "Error in imwback_warpImage function");
    }

    int result4 = imwback_inv_meshgrid(d_X, d_Y, cols, rows, homography.data());
    if (result4 != 0)
    {
        cudaFree((void*)d_img1);
        cudaFree((void*)d_img2);
        cudaFree((void*)d_X);
        cudaFree((void*)d_Y);
        cudaFree((void*)img1p);
        cudaDeviceReset();
        mexErrMsgIdAndTxt("imcoregmx:imwbackError", "Error in imwback_inv_meshgrid function");
    }

    int result5 = imwback_warpImage(d_img2, img2_cols, img2_rows, &img2p, cols, rows, d_X, d_Y);
    if (result5 != 0)
    {
        cudaFree((void*)d_img1);
        cudaFree((void*)d_img2);
        cudaFree((void*)d_X);
        cudaFree((void*)d_Y);
        cudaFree((void*)img1p);
        cudaDeviceReset();
        mexErrMsgIdAndTxt("imcoregmx:imwbackError", "Error in imwback_warpImage function");
    }

    cudaFree((void*)d_img1);
    cudaFree((void*)d_img2);
    cudaFree((void*)d_X);
    cudaFree((void*)d_Y);

    cv::Mat h_img1p(rows, cols, CV_32FC1);
    cv::Mat h_img2p(rows, cols, CV_32FC1);

    cudaMemcpy(h_img1p.ptr<float>(), img1p, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_img2p.ptr<float>(), img2p, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    cv::imwrite("img1p.png", h_img1p);
    cv::imwrite("img2p.png", h_img2p);

    // Allocate new gpu memory for the output matrices, transposed
    float *img1p_T, *img2p_T;
    int result6 = imwback_transpose(&img1p_T, &img2p_T, img1p, img2p, cols, rows);
    if (result6 != 0)
    {
        cudaFree((void*)img1p);
        cudaFree((void*)img2p);
        cudaDeviceReset();
        mexErrMsgIdAndTxt("imcoregmx:imwbackError", "Error in imwback_transpose function");
    }

    cudaFree((void*)img1p);
    cudaFree((void*)img2p);


    cv::Mat h_img1p_T(rows, cols, CV_32FC1);
    cv::Mat h_img2p_T(rows, cols, CV_32FC1);

    cudaMemcpy(h_img1p_T.ptr<float>(), img1p_T, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_img2p_T.ptr<float>(), img2p_T, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    cv::imwrite("img1p_T.png", h_img1p_T);
    cv::imwrite("img2p_T.png", h_img2p_T);

    
    cudaFree((void*)img1p_T);
    cudaFree((void*)img2p_T);
    cudaDeviceReset();
    cudaMemGetInfo(&freeMem2, &totalMem2);
    mexPrintf("Free memory: %lu\n", freeMem2);

    mexPrintf("Free memory difference: %lf\n", (double)freeMem - (double)freeMem2);

    return 0;
}



