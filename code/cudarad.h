#ifndef __CUDARAD_H_
#define __CUDARAD_H_

#include <vector>

#include "bsp.h"
#include "cudabsp.h"

namespace CUDARAD {
    void init(BSP::BSP& bsp);
    void cleanup(void);

    void compute_direct_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP);
    void antialias_direct_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP);

    void bounce_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP);
}

#endif
