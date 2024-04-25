
#include "mex.h"
#include "matrix.h"

#include <vector>
#include <cstdbool>
#include <cmath>

#include <cuda_runtime.h>
#include "cuSIFT.hpp"
#include "imwback.h"

const char *usage()
{
    return "Usage: [img1p, img2p, varargout] = imcoregmx(img1, img2, varargin)\n"
           "varargin = {numFeatures, initBlur, thresh, lowestScale}\n"
           "varargout = {homography, inlier_ratio}\n"
           "Variable arguments are optional, and always have the same order above.\n"
           "Args:\n"
           "  img1, img2: 2D real single matrices\n"
           "  numFeatures: positive integer\n"
           "  initBlur: non-negative real scalar\n"
           "  thresh: non-negative real scalar\n"
           "  lowestScale: non-negative real scalar\n";
}


void isValidInputs(int nrhs, const mxArray *prhs[]);


// mexFunction:  Entry point for the MEX-file.
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Usage:
    // [img1p, img2p, varargout] = imcoregmx(img1, img2, varargin)
    // varargin = {numFeatures, initBlur, thresh, lowestScale}
    // varargout = {homography, inlier_ratio}
    //
    // Variable arguments are optional, and always have the same order above.

    // Check for proper number of input and output arguments
    // Number of input arguments must be 2-6
    // Number of output arguments must be 2-4

    isValidInputs(nrhs, prhs);

    if (nlhs < 2 || nlhs > 4)
    {
        mexErrMsgIdAndTxt("imcoregmx:invalidNumOutputs", usage());
    }

    //
    // Outputs:
    //  img1p and img2p are 2D real single matrices
    //  homography is a 3x3 real single matrix
    //  inlier_ratio is a real scalar
    //

    float *img1, *img2;
    int height1, width1, height2, width2;

    img1 = (float *)mxGetPr(prhs[0]);   // img1, swap rows and cols
    width1 = mxGetM(prhs[0]);
    height1 = mxGetN(prhs[0]);

    img2 = (float *)mxGetPr(prhs[1]);   // img2
    width2 = mxGetM(prhs[1]);
    height2 = mxGetN(prhs[1]);

    // These are device pointers, and will be allocated in the GPU
    float *d_img1, *d_img2;
    std::vector<float> homography;
    float inlier_ratio;

    // Variable arguments
    int numFeatures = 4000;
    float initBlur = 1.0f;
    float thresh = 3.0f;
    float lowestScale = 0.0f;

    if (nrhs > 2) numFeatures = (int)mxGetScalar(prhs[2]);
    if (nrhs > 3) initBlur = (float)mxGetScalar(prhs[3]);
    if (nrhs > 4) thresh = (float)mxGetScalar(prhs[4]);
    if (nrhs > 5) lowestScale = (float)mxGetScalar(prhs[5]);

    if (numFeatures <= 0 || initBlur <= 0 || thresh <= 0 || lowestScale < 0)
    {
        mexErrMsgIdAndTxt("imcoregmx:invalidInputValue", usage());
    }

    // Call the CUDA SIFT function
    int result = cudaSift(
        img1, width1, height1,
        img2, width2, height2,
        homography, &inlier_ratio,
        &d_img1, &d_img2,
        numFeatures, initBlur, thresh, lowestScale);

    if (result != CUSIFT_ERROR_NONE)
    {
        mexErrMsgIdAndTxt("imcoregmx:cudaSiftError", "Error in cudaSift function");
    }

    // Get the meshgrid
    float *d_X, *d_Y;
    int width, height;
    int result2 = imwback_get_meshgrid(&d_X, &d_Y, &width, &height, homography.data(), width1, height1, width2, height2);
    if (result2 != 0)
    {
        cudaFree((void*)d_img1);
        cudaFree((void*)d_img2);
        cudaDeviceReset();
        mexErrMsgIdAndTxt("imcoregmx:imwbackError", "Error in imwback_get_meshgrid function");
    }

    float *img1p, *img2p;
    int result3 = imwback_warpImage(d_img1, width1, height1, &img1p, width, height, d_X, d_Y);
    if (result3 != 0)
    {
        cudaFree((void*)d_img1);
        cudaFree((void*)d_img2);
        cudaFree((void*)d_X);
        cudaFree((void*)d_Y);
        cudaDeviceReset();
        mexErrMsgIdAndTxt("imcoregmx:imwbackError", "Error in imwback_warpImage function");
    }
    int result4 = imwback_inv_meshgrid(d_X, d_Y, width, height, homography.data());
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
    int result5 = imwback_warpImage(d_img2, width2, height2, &img2p, width, height, d_X, d_Y);
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

    // Allocate new gpu memory for the output matrices, transposed
    float *img1p_T, *img2p_T;
    int result6 = imwback_transpose(&img1p_T, &img2p_T, img1p, img2p, width, height);
    if (result6 != 0)
    {
        cudaFree((void*)img1p);
        cudaFree((void*)img2p);
        cudaDeviceReset();
        mexErrMsgIdAndTxt("imcoregmx:imwbackError", "Error in imwback_transpose function");
    }

    cudaFree((void*)img1p);
    cudaFree((void*)img2p);

    // Create the output matrices
    plhs[0] = mxCreateNumericMatrix(height, width, mxSINGLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(height, width, mxSINGLE_CLASS, mxREAL);

    // Copy the data to the output matrices
    float *img1p_out = (float *)mxGetPr(plhs[0]);
    float *img2p_out = (float *)mxGetPr(plhs[1]);

    cudaError_t err = cudaMemcpy(img1p_out, img1p_T, height * width * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cudaFree((void*)img1p_T);
        cudaFree((void*)img2p_T);
        cudaDeviceReset();
        mexErrMsgIdAndTxt("imcoregmx:cudaMemcpyError", "Error in cudaMemcpy function");
    }

    err = cudaMemcpy(img2p_out, img2p_T, height * width * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cudaFree((void*)img1p_T);
        cudaFree((void*)img2p_T);
        cudaDeviceReset();
        mexErrMsgIdAndTxt("imcoregmx:cudaMemcpyError", "Error in cudaMemcpy function");
    }

    cudaFree((void*)img1p_T);
    cudaFree((void*)img2p_T);
    cudaDeviceReset();

    // Return the homography matrix and inlier ratio, if requested
    if (nlhs > 2)
    {
        plhs[2] = mxCreateNumericMatrix(3, 3, mxSINGLE_CLASS, mxREAL);
        float *homography_out = (float *)mxGetPr(plhs[2]);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                // Matlab is column-major order, we are using row-major order
                homography_out[i + j * 3] = homography[i * 3 + j];
            }
        }
    }

    if (nlhs > 3)
    {
        plhs[3] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
        float *inlier_ratio_out = (float *)mxGetPr(plhs[3]);
        inlier_ratio_out[0] = inlier_ratio;
    }
}



void isValidInputs(int nrhs, const mxArray *prhs[])
{
    // Check for proper input argument types
    if (nrhs < 2 || nrhs > 6)
    {
        mexErrMsgIdAndTxt("imcoregmx:invalidNumInputs", usage());
    }

    // Check for proper input argument types
    // Inputs:
    //  img1 and img2 must be 2D real single matrices
    //  numFeatures must be a positive integer
    //  initBlur must be a non-negative real scalar
    //  thresh must be a non-negative real scalar
    //  lowestScale must be a non-negative real scalar

    if (!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1]))
    {
        mexErrMsgIdAndTxt("imcoregmx:invalidInputType", usage());
    }

    if (mxGetNumberOfDimensions(prhs[0]) != 2 || mxGetNumberOfDimensions(prhs[1]) != 2)
    {
        mexErrMsgIdAndTxt("imcoregmx:invalidInputType", usage());
    }

    if (nrhs > 2)
    {
        if (!mxIsDouble(prhs[2]) || mxGetNumberOfElements(prhs[2]) != 1)
        {
            mexErrMsgIdAndTxt("imcoregmx:invalidInputType", usage());
        }
    }

    if (nrhs > 3)
    {
        if (!mxIsDouble(prhs[3]) || mxGetNumberOfElements(prhs[3]) != 1)
        {
            mexErrMsgIdAndTxt("imcoregmx:invalidInputType", usage());
        }
    }

    if (nrhs > 4)
    {
        if (!mxIsDouble(prhs[4]) || mxGetNumberOfElements(prhs[4]) != 1)
        {
            mexErrMsgIdAndTxt("imcoregmx:invalidInputType", usage());
        }
    }

    if (nrhs > 5)
    {
        if (!mxIsDouble(prhs[5]) || mxGetNumberOfElements(prhs[5]) != 1)
        {
            mexErrMsgIdAndTxt("imcoregmx:invalidInputType", usage());
        }
    }
}



