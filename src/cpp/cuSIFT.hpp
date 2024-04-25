#pragma once
#ifndef __CUSIFT_HPP__
#define __CUSIFT_HPP__

#include <vector>

#define CUSIFT_ERROR_NONE 0
#define CUSIFT_ERROR_ALLOCATING_DEVICE_MEMORY -1
#define CUSIFT_ERROR_COPYING_TO_DEVICE -2
#define CUSIFT_ERROR_COPYING_FROM_DEVICE -3
#define CUSIFT_ERROR_EXTRACTING_FEATURES -4
#define CUSIFT_ERROR_MATCHING_FEATURES -5
#define CUSIFT_ERROR_COMPUTING_HOMOGRAPHY -6
#define CUSIFT_ERROR_IMPROVIDING_HOMOGRAPHY -7

/**
 * @brief Use CUDA to compute SIFT features for two images, and return the homography matrix.
 * The homography matrix is a 3x3 matrix that maps points in the second image to points in the first image.
 * 
 * @param h_img1 Host pointer to the first image.
 * @param w1 Width of the first image. Fastest dimension.
 * @param h1 Height of the first image. Slowest dimension.
 * @param h_img2 Host pointer to the second image.
 * @param w2 Width of the second image. Fastest dimension.
 * @param h2 Height of the second image. Slowest dimension.
 * @param d_img1 We will allocate a device image for the first image and return a pointer to it here, if not NULL.
 * @param d_img2 We will allocate a device image for the second image and return a pointer to it here, if not NULL.
 * @param numFeatures Number of features to extract in each image.
 * @param initBlur Initial blur to apply to the images.
 * @param thresh Feature threshold, larger values removes weaker features.
 * @param lowestScale Lowest scale to extract features at.
 */
int cudaSift(
    const float *h_img1, int w1, int h1,
    const float *h_img2, int w2, int h2,
    std::vector<float> &homography,
    float *inlier_ratio=nullptr,
    float **d_img1=nullptr, float **d_img2=nullptr,
    int numFeatures=4000, float initBlur=1.0f, float thresh=3.0f, float lowestScale=0.0f);

#endif // __CUSIFT_HPP__
