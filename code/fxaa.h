#ifndef __FXAA_H_
#define __FXAA_H_

#include "bsp.h"
#include "cudabsp.h"

namespace CUDAFXAA {
    void antialias_lightsamples(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP);
}

#endif
