
#include "geomFuncs_blas.hpp"
#include <lapacke.h>
#include <cblas.h>


void solve_8x8(const double *M, double *A, const double *X);

inline void printMatrix(const double *M, int rows, int cols)
{
    double val;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            val = M[i * cols + j];
            if (val <= 0.0)
            {
                printf(" %.7e", val);
            }
            else
            {
                printf("  %.7e", val);
            }
        }
        printf("\n");
    }
}

int ImproveHomography_Mat(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh)
{
    if (data.h_data == NULL)
        return 0;

    SiftPoint *mpts = data.h_data;
    const double limit = thresh * thresh;

    const int numPts = data.numPts;
    const double zeros64[64] = {0.0f};
    const double zeros8[8] = {0.0f};
    double M[64];
    double A[8];
    double X[8];
    double Y[8];

    for (int i = 0; i < 8; i++)
        A[i] = homography[i] / homography[8];

    for (int loop = 0; loop < numLoops; loop++)
    {
        cblas_dcopy(64, zeros64, 1, M, 1);
        cblas_dcopy(8, zeros8, 1, X, 1);

        for (int i = 0; i < numPts; i++)
        {
            SiftPoint &pt = mpts[i];
            if (pt.score < minScore || pt.ambiguity > maxAmbiguity)
                continue;
            double den = A[6] * pt.xpos + A[7] * pt.ypos + 1.0f;
            double dx = (A[0] * pt.xpos + A[1] * pt.ypos + A[2]) / den - pt.match_xpos;
            double dy = (A[3] * pt.xpos + A[4] * pt.ypos + A[5]) / den - pt.match_ypos;
            double err = dx * dx + dy * dy;
            double wei = (err < limit ? 1.0f : 0.0f); // limit / (err + limit);
            Y[0] = pt.xpos;
            Y[1] = pt.ypos;
            Y[2] = 1.0;
            Y[3] = Y[4] = Y[5] = 0.0;
            Y[6] = -pt.xpos * pt.match_xpos;
            Y[7] = -pt.ypos * pt.match_xpos;
            for (int j = 0; j < 64; j++)
                M[j] += (Y[j % 8] * Y[j / 8] * wei);
            cblas_daxpy(8, pt.match_xpos * wei, Y, 1, X, 1);
            Y[0] = Y[1] = Y[2] = 0.0;
            Y[3] = pt.xpos;
            Y[4] = pt.ypos;
            Y[5] = 1.0;
            Y[6] = -pt.xpos * pt.match_ypos;
            Y[7] = -pt.ypos * pt.match_ypos;
            for (int j = 0; j < 64; j++)
                M[j] += (Y[j % 8] * Y[j / 8] * wei);
            cblas_daxpy(8, pt.match_ypos * wei, Y, 1, X, 1);
        }
        solve_8x8(M, A, X);
    }
    int numfit = 0;
    for (int i = 0; i < numPts; i++)
    {
        SiftPoint &pt = mpts[i];
        double den = A[6] * pt.xpos + A[7] * pt.ypos + 1.0;
        double dx = (A[0] * pt.xpos + A[1] * pt.ypos + A[2]) / den - pt.match_xpos;
        double dy = (A[3] * pt.xpos + A[4] * pt.ypos + A[5]) / den - pt.match_ypos;
        double err = dx * dx + dy * dy;
        if (err < limit)
            numfit++;
        pt.match_error = sqrt(err);
    }

    for (int i = 0; i < 8; i++)
        homography[i] = A[i];
    homography[8] = 1.0f;
    return numfit;
}


void solve_8x8(const double *M, double *A, const double *X)
{
    // Solve the least squares problem M * A = X
    // M is a matrix of size 8 x 8
    // A is a matrix of size 8 x 1
    // X is a matrix of size 8 x 1
    //
    // A is the output matrix
    // lapacke is used to solve the least squares problem
    // The following function call solves the least squares problem

    // single precision real valued least squares solution: sgels
    lapack_int m = 8;
    lapack_int n = 8;
    lapack_int nrhs = 1;
    lapack_int lda = 8;
    lapack_int ldb = 1;
    lapack_int info;

    // Copy X to A
    cblas_dcopy(8, X, 1, A, 1);

    // Create a copy of M, it is only 64 elements, try the stack
    double M_copy[64];
    cblas_dcopy(64, M, 1, M_copy, 1);

    info = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', m, n, nrhs, M_copy, lda, A, ldb);

    if (info != 0)
    {
        fprintf(stderr, "LAPACK sgels failed with error code %d\n", info);
    }
}
