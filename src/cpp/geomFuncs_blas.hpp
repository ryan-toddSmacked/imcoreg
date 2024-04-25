#pragma once
#ifndef __GEOMFUNCS_MAT_HPP__
#define __GEOMFUNCS_MAT_HPP__

#ifdef __cplusplus
extern "C" {
#endif

#include "cudaSift.h"

int ImproveHomography_Mat(SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);




#ifdef __cplusplus
}
#endif

#endif // __GEOMFUNCS_MAT_HPP__
