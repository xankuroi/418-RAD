#ifndef __CUDABSP_H_
#define __CUDABSP_H_

#include <cstdint>

#include "cuda_runtime.h"
#include "gmtl/Matrix.h"

#include "bsp.h"

#include "cudamatrix.h"


namespace CUDABSP {
    const uint32_t TAG = 0xdeadbeef;

    struct CUDABSP {
        uint32_t tag;

        BSP::DModel* models;
        BSP::DPlane* planes;
        float3* vertices;
        BSP::DEdge* edges;
        int32_t* surfEdges;
        BSP::DFace* faces;
        CUDAMatrix::CUDAMatrix<double, 3, 3>* xyzMatrices;
        float3* lightSamples;
        BSP::RGBExp32* rgbExp32LightSamples;
        BSP::TexInfo* texInfos;
        BSP::DTexData* texDatas;
        BSP::DLeaf* leaves;
        BSP::DWorldLight* worldLights;

        size_t numModels;
        size_t numPlanes;
        size_t numVertices;
        size_t numEdges;
        size_t numSurfEdges;
        size_t numFaces;
        size_t numLightSamples;
        size_t numTexInfos;
        size_t numTexDatas;
        size_t numLeaves;
        size_t numWorldLights;
    };

    /** Creates a new CUDABSP on the device, and returns a pointer to it. */
    CUDABSP* make_cudabsp(const BSP::BSP& bsp);

    /** Destroys the given CUDABSP located on the devices. */
    void destroy_cudabsp(CUDABSP* pCudaBSP);

    /** Convert lightsamples from float3 to RGBExp32 format. */
    void convert_lightsamples(CUDABSP* pCudaBSP);

    /**
    * Updates the given BSP using the information contained in the given
    * CUDABSP (which should be on the device).
    */
    void update_bsp(BSP::BSP& bsp, CUDABSP* pCudaBSP);
}


#endif
