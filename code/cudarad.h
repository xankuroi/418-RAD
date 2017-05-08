#ifndef __CUDARAD_H_
#define __CUDARAD_H_

#include <vector>

#include "bsp.h"
#include "cudabsp.h"

__device__ bool intersects(
    const float3& vertex1, const float3& vertex2, const float3& vertex3,
    const float3& startPos, const float3& endPos
);

namespace CUDARAD {
    void init(BSP::BSP& bsp);
    void cleanup(void);

    void compute_direct_lighting(
        BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP,
        std::vector<BSP::RGBExp32>& lightSamples
    );

    void bounce_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP);
}

#endif
