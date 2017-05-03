#ifndef __CUDARAD_H_
#define __CUDARAD_H_

#include <vector>

#include "bsp.h"
#include "cudabsp.h"

namespace CUDARAD {
    void compute_direct_lighting(
        BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP,
        std::vector<BSP::RGBExp32>& lightSamples
    );

    void bounce_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP);
}

#endif
