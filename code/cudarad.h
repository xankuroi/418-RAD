#ifndef __CUDARAD_H_
#define __CUDARAD_H_

#include <vector>

#include "bsp.h"
#include "cudabsp.h"

namespace CUDARAD {
    struct FaceInfo {
        BSP::DFace face;
        BSP::DPlane plane;
        BSP::TexInfo texInfo;
        CUDAMatrix::CUDAMatrix<double, 3, 3> Ainv;
        float3 faceNorm;
        float3 totalLight;
        float3 avgLight;
        size_t faceIndex;
        size_t lightmapWidth;
        size_t lightmapHeight;
        size_t lightmapSize;
        size_t lightmapStartIndex;

        __device__ FaceInfo();
        __device__ FaceInfo(CUDABSP::CUDABSP& cudaBSP, size_t faceInfex);

        __device__ float3 xyz_from_st(float s, float t);
    };
}

namespace DirectLighting {
    __device__ float3 sample_at(
        CUDABSP::CUDABSP& cudaBSP, CUDARAD::FaceInfo& faceInfo,
        float s, float t
    );
}

namespace CUDARAD {
    void init(BSP::BSP& bsp);
    void cleanup(void);

    void compute_direct_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP);
    void antialias_direct_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP);

    void bounce_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP);
}

#endif
