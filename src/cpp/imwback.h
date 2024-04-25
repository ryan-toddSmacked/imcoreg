#ifndef __IMWBACK_H__
#define __IMWBACK_H__

#ifdef __cplusplus
extern "C" {
#endif


int imwback_get_meshgrid(float **d_X, float **d_Y, int *width, int *height, const float *H, int width1, int height1, int width2, int height2);

int imwback_warpImage(const float *d_img, int width, int height, float **d_imgW, int widthW, int heightW, const float *d_X, const float *d_Y);

int imwback_inv_meshgrid(float *d_X, float *d_Y, int width, int height, const float *H);

int imwback_transpose(float **d_1out, float **d_2out, float *d_1src, const float *d_2src, int width, int height);

#ifdef __cplusplus
}
#endif

#endif // __IMWBACK_H__
