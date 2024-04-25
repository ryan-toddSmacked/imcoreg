
#include "imwback.h"

#include <cuda_runtime.h>
#include <lapacke.h>
#include <cblas.h>
#include <cublas_v2.h>

#define MIN2(a,b) ((a) < (b) ? (a) : (b))
#define MAX2(a,b) ((a) > (b) ? (a) : (b))
#define MIN3(a,b,c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))
#define MAX3(a,b,c) ((a) > (b) ? ((a) > (c) ? (a) : (c)) : ((b) > (c) ? (b) : (c)))

#define BLOCK_DIM 16

int solve_Hx(const float *H, float *x, const float *b);
float *arange(float start, float stop, int* length);
int meshgrid(float **d_X, float **d_Y, const float *d_x, const float *d_y, int lengthX, int lengthY);

__global__ void arange_kernel(float *d_arr, float start, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        d_arr[i] = start + (float)i;
    }
}

__global__ void meshgrid_kernel(float *d_X, float *d_Y, const float *d_x, const float *d_y, int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < width && j < height)
    {
        d_X[j*width + i] = d_x[i];
        d_Y[j*width + i] = d_y[j];
    }
}

__global__ void warpImage_kernel(const float *d_img, int width, int height, float *d_imgW, int widthW, int heightW, const float *d_X, const float *d_Y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    const float xMin = 0.0f;
    const float xMax = width-1;
    const float yMin = 0.0f;
    const float yMax = height-1;

    const float NaN = nanf("");
    
    if (i >= widthW || j >= heightW)
    {
        return;
    }

    int index = j*widthW + i;

    float x = d_X[index];
    float y = d_Y[index];
    float z = NaN;

    if (x >= xMin && x <= xMax && y >= yMin && y <= yMax)
    {
        int x0 = (int)x;
        int y0 = (int)y;
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        const float *P = d_img + y0*width + x0;
        
        // Weights
        float wx0 = x0 + 1;
        float wy0 = y0 + 1;
        float wx1 = x1 + 1;
        float wy1 = y1 + 1;

        float wx = (x - x0) / (x1 - x0);
        float wy = (y - y0) / (y1 - y0);

        // Load all possible neighbors
        float z00 = 0.0f;
        float z01 = 0.0f;
        float z10 = 0.0f;
        float z11 = 0.0f;

        z00 = *P;
        if (x1 < width)
        {
            z01 = *(P + 1);
        }

        // Increment P to the next row
        P += width;

        if (y1 < height)
        {
            z10 = *P;
            if (x1 < width)
            {
                z11 = *(P + 1);
            }
        }

        // Interpolate
        z =
            (1 - wy) * ((1-wx) * z00 + wx * z01) +
            wy * ((1-wx) * z10 + wx * z11);
    }

    d_imgW[index] = z;
}

__global__ void inv_meshgrid_kernel(float *X, float *Y, const float *H, int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height)
    {
        return;
    }

    int index = j*width + i;

    float z = H[6]*X[index] + H[7]*Y[index] + H[8];

    float x = (H[0]*X[index] + H[1]*Y[index] + H[2]) / z;
    float y = (H[3]*X[index] + H[4]*Y[index] + H[5]) / z;

    X[index] = x;
    Y[index] = y;
}

__global__ void transpose_kernel(float *odata1, const float *idata1, float *odata2, const float *idata2, int width, int height)
{
	__shared__ float block1[BLOCK_DIM][BLOCK_DIM+1];
    __shared__ float block2[BLOCK_DIM][BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block1[threadIdx.y][threadIdx.x] = idata1[index_in];
        block2[threadIdx.y][threadIdx.x] = idata2[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata1[index_out] = block1[threadIdx.x][threadIdx.y];
        odata2[index_out] = block2[threadIdx.x][threadIdx.y];
	}
}

int imwback_get_meshgrid(float **d_X, float **d_Y, int *width, int *height, const float *H, int width1, int height1, int width2, int height2)
{
    // box [3x4]
    float box[12] = {
        0, width2-1,  width2-1,         0,
        0,        0, height2-1, height2-1,
        1,        1,         1,         1
    };

    // Solve H*x = box
    float box2[12];
    solve_Hx(H, box2, box);

    box2[0] /= box2[6];
    box2[1] /= box2[7];
    box2[2] /= box2[8];
    box2[3] /= box2[6];
    box2[4] /= box2[7];
    box2[5] /= box2[8];

    const float minX = MIN2(0.0f, MIN3(box2[0], box2[1], box2[2]));
    const float maxX = MAX2((float)(width1-1), MAX3(box2[0], box2[1], box2[2]));
    const float minY = MIN2(0.0f, MIN3(box2[3], box2[4], box2[5]));
    const float maxY = MAX2((float)(height1-1), MAX3(box2[3], box2[4], box2[5]));

    int lengthX, lengthY;
    float *d_x = arange(minX, maxX+1, &lengthX);
    float *d_y = arange(minY, maxY+1, &lengthY);

    if (d_x == NULL || d_y == NULL)
    {
        return -1;
    }

    float *d_X_, *d_Y_;
    int ret = meshgrid(&d_X_, &d_Y_, d_x, d_y, lengthX, lengthY);
    if (ret != 0)
    {
        cudaFree((void*)d_x);
        cudaFree((void*)d_y);
        return -1;
    }

    cudaFree((void*)d_x);
    cudaFree((void*)d_y);

    *d_X = d_X_;
    *d_Y = d_Y_;
    *height = lengthY;
    *width = lengthX;
    return 0;
}


int imwback_warpImage(const float *d_img, int width, int height, float **d_imgW, int widthW, int heightW, const float *d_X, const float *d_Y)
{
    float *d_imgW_;
    cudaError_t err;
    err = cudaMalloc((void**)&d_imgW_, widthW * heightW * sizeof(float));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed with ERRORcode %d\n", err);
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        return -1;
    }

    dim3 blockSize(16, 16);
    dim3 numBlocks((widthW + blockSize.x - 1) / blockSize.x, (heightW + blockSize.y - 1) / blockSize.y);
    warpImage_kernel<<<numBlocks, blockSize>>>(d_img, width, height, d_imgW_, widthW, heightW, d_X, d_Y);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "warpImage_kernel failed with error code %d\n", err);
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        cudaFree((void*)d_imgW_);
        return -1;
    }

    *d_imgW = d_imgW_;

    return 0;
}


int imwback_inv_meshgrid(float *d_X, float *d_Y, int width, int height, const float *H)
{
    float *d_H;
    cudaError_t err;
    err = cudaMalloc((void**)&d_H, 9 * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed with error code %d\n", err);
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(d_H, H, 9 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed with error code %d\n", err);
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        cudaFree((void*)d_H);
        return -1;
    }

    dim3 blockSize(16, 16);
    dim3 numBlocks((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    inv_meshgrid_kernel<<<numBlocks, blockSize>>>(d_X, d_Y, d_H, width, height);

    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "inv_meshgrid_kernel failed with error code %d\n", err);
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        cudaFree((void*)d_H);
        return -1;
    }

    cudaFree((void*)d_H);
    return 0;
}

int imwback_transpose(float **d_1out, float **d_2out, float *d_1src, const float *d_2src, int width, int height)
{
    float *d_1out_, *d_2out_;
    cudaError_t err;
    err = cudaMalloc((void**)&d_1out_, width * height * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed with error code %d\n", err);
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc((void**)&d_2out_, width * height * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed with error code %d\n", err);
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        cudaFree((void*)d_1out_);
        return -1;
    }

    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    dim3 numBlocks((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    transpose_kernel<<<numBlocks, blockSize>>>(d_1out_, d_1src, d_2out_, d_2src, width, height);

    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "transpose_kernel failed with error code %d\n", err);
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        cudaFree((void*)d_1out_);
        cudaFree((void*)d_2out_);
        return -1;
    }

    *d_1out = d_1out_;
    *d_2out = d_2out_;
    return 0;
}



inline void inv_3x3(float *H)
{

// const float A = (H[4] * H[8] - H[5] * H[7]);
//double det = m(0, 0) * (A) -
//             m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
//             m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
//
//double invdet = 1 / det;
//
//Matrix33d minv; // inverse of matrix m
//minv(0, 0) = (A) * invdet;
//minv(0, 1) = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * invdet;
//minv(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * invdet;
//minv(1, 0) = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) * invdet;
//minv(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * invdet;
//minv(1, 2) = (m(1, 0) * m(0, 2) - m(0, 0) * m(1, 2)) * invdet;
//minv(2, 0) = (m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1)) * invdet;
//minv(2, 1) = (m(2, 0) * m(0, 1) - m(0, 0) * m(2, 1)) * invdet;
//minv(2, 2) = (m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1)) * invdet;

    const float A = (H[4] * H[8] - H[5] * H[7]);
    const float det =
        H[0] * A -
        H[1] * (H[3] * H[8] - H[5] * H[6]) +
        H[2] * (H[3] * H[7] - H[4] * H[6]);
    const float invdet = 1.0f / det;

    float H_[9];
    H_[0] = A * invdet;
    H_[1] = (H[2] * H[7] - H[1] * H[8]) * invdet;
    H_[2] = (H[1] * H[5] - H[2] * H[4]) * invdet;
    H_[3] = (H[5] * H[6] - H[3] * H[8]) * invdet;
    H_[4] = (H[0] * H[8] - H[2] * H[6]) * invdet;
    H_[5] = (H[2] * H[3] - H[0] * H[5]) * invdet;
    H_[6] = (H[3] * H[7] - H[4] * H[6]) * invdet;
    H_[7] = (H[1] * H[6] - H[0] * H[7]) * invdet;
    H_[8] = (H[0] * H[4] - H[1] * H[3]) * invdet;

    H[0] = H_[0];
    H[1] = H_[1];
    H[2] = H_[2];
    H[3] = H_[3];
    H[4] = H_[4];
    H[5] = H_[5];
    H[6] = H_[6];
    H[7] = H_[7];
    H[8] = H_[8];
}


int solve_Hx(const float *H, float *x, const float *b)
{
    // Performing the folling operation
    // x = inv(H)*b
    // H - [3x3]
    // b - [3x4]
    // x - [3x4]

    // Create a copy of H
    float H_[9];
    cblas_scopy(9, H, 1, H_, 1);

    // Invert H
    lapack_int IPIV[3];
    lapack_int LWORK = 9;
    float WORK[9];
    lapack_int INFO;

    inv_3x3(H_);

    // Multiply H^-1*b, sgemm is used to multiply two matrices
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 4, 3, 1.0f, H_, 3, b, 4, 0.0f, x, 4);

    return 0;
}

float *arange(float start, float stop, int* length)
{
    // Create a range of numbers from start to stop [start, start+1, ..., stop)
    // incrementing by 1
    // The range is returned as a 1D array

    int n = (int)ceilf(stop - start);
    float *d_arr;
    cudaError_t err;
    err = cudaMalloc((void**)&d_arr, n * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed with error code %d\n", err);
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        return NULL;
    }

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    arange_kernel<<<numBlocks, blockSize>>>(d_arr, start, n);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "arange_kernel failed with error code %d\n", err);
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        cudaFree((void*)d_arr);
        return NULL;
    }

    *length = n;
    return d_arr;
}

int meshgrid(float **d_X, float **d_Y, const float *d_x, const float *d_y, int lengthX, int lengthY)
{
    float *d_X_, *d_Y_;
    cudaError_t err;
    err = cudaMalloc((void**)&d_X_, lengthX * lengthY * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed with error code %d\n", err);
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc((void**)&d_Y_, lengthX * lengthY * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed with error code %d\n", err);
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        cudaFree((void*)d_X_);
        return -1;
    }

    dim3 blockSize(16, 16);
    dim3 numBlocks((lengthX + blockSize.x - 1) / blockSize.x, (lengthY + blockSize.y - 1) / blockSize.y);
    meshgrid_kernel<<<numBlocks, blockSize>>>(d_X_, d_Y_, d_x, d_y, lengthX, lengthY);

    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "meshgrid_kernel failed with error code %d\n", err);
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
        cudaFree((void*)d_X_);
        cudaFree((void*)d_Y_);
        return -1;
    }

    *d_X = d_X_;
    *d_Y = d_Y_;
    return 0;
}

