#include <iostream>
#include <fstream>

#include <string>
#include <vector>
#include <random>
#include <memory>

#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <cmath>

#include <gmtl/LineSeg.h>
#include <gmtl/Plane.h>
#include <gmtl/Intersection.h>

#include "cuda.h"
#include "bsp.h"
#include "cudabsp.h"
#include "cudarad.h"
#include "cudautils.h"


void print_cudainfo(void) {
    int device;
    CUDA_CHECK_ERROR(cudaGetDevice(&device));

    cudaDeviceProp deviceProps;
    CUDA_CHECK_ERROR(cudaGetDeviceProperties(&deviceProps, device));

    std::cout << "CUDA Device: " << deviceProps.name << std::endl;
    std::cout << "    Device Memory: "
        << deviceProps.totalGlobalMem << std::endl;
    std::cout << "    Max Threads/Block: "
        << deviceProps.maxThreadsPerBlock << std::endl;
    std::cout << "    Max Block Dim X: "
        << deviceProps.maxThreadsDim[0] << std::endl;
    std::cout << "    Max Block Dim Y: "
        << deviceProps.maxThreadsDim[1] << std::endl;
    std::cout << "    Max Block Dim Z: "
        << deviceProps.maxThreadsDim[2] << std::endl;
    std::cout << "    Max Grid Size X: "
        << deviceProps.maxGridSize[0] << std::endl;
    std::cout << "    Max Grid Size Y: "
        << deviceProps.maxGridSize[1] << std::endl;
    std::cout << "    Max Grid Size Z: "
        << deviceProps.maxGridSize[2] << std::endl;
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Invalid arguments." << std::endl;
        return 1;
    }
    
    std::cout << "418RAD -- 15-418 Radiosity Simulator" << std::endl;

    const std::string filename(argv[1]);
    std::ifstream f(filename, std::ios::binary);
    
    std::unique_ptr<BSP::BSP> pBSP;

    try {
        pBSP = std::unique_ptr<BSP::BSP>(new BSP::BSP(filename));
    }
    catch (BSP::InvalidBSP e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    pBSP->build_worldlights();

    print_cudainfo();

    CUDA_CHECK_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));

    std::cout << "Copy BSP to device memory..." << std::endl;
    CUDABSP::CUDABSP* pCudaBSP = CUDABSP::make_cudabsp(*pBSP);
    
    std::cout << "Initialize radiosity subsystem..." << std::endl;
    CUDARAD::init(*pBSP);

    std::cout << "Start RAD!" << std::endl;
    
    std::cout << "Compute direct lighting..." << std::endl;
    std::vector<BSP::RGBExp32> directLighting;
    CUDARAD::compute_direct_lighting(*pBSP, pCudaBSP, directLighting);

    std::cout << "Compute light bounces..." << std::endl;
    CUDARAD::bounce_lighting(*pBSP, pCudaBSP);

    CUDABSP::update_bsp(*pBSP, pCudaBSP);

    CUDABSP::destroy_cudabsp(pCudaBSP);

    /*
     * Mark the BSP as non-fullbright.
     *
     * This tells the engine that there is actually lighting information 
     * embedded in the map.
     */
    pBSP->set_fullbright(false);
    
    pBSP->write("out.bsp");
    
    std::cout << "Wrote to file \"out.bsp\"." << std::endl;

    /* Tear down the radiosity subsystem. */
    CUDARAD::cleanup();
    
    return 0;
}
