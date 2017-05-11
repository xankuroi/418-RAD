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
        __device__ FaceInfo(CUDABSP::CUDABSP& cudaBSP, size_t faceIndex);

        __device__ float3 xyz_from_st(float s, float t);
    };

    struct PatchInfo{
      FaceInfo face_info;
      float3 initialLight;
      float3 totalLight;
      float3 brightness;
      float3 reflectivity;
      float3 receivedLight;
      float3* vertices;
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
    void bounce_lighting_fly(CUDABSP::CUDABSP* pCudaBSP);
    __global__ void bounce_iteration(CUDABSP::CUDABSP* pCudaBSP, PatchInfo* patches, int totalNumPatches);
    void generate_face_info(CUDABSP::CUDABSP* pCudaBSP, CUDARAD::FaceInfo& faces);
    void generate_patch_info(CUDABSP::CUDABSP* pCudaBSP, CUDARAD::FaceInfo* faces,
                             size_t num_faces, CUDARAD::PatchInfo& patches);

    float3 center(PatchInfo patch);
    static __device__ inline float distance(float3 p1, float3 p2);
}

#endif
