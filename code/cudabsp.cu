#include <iostream>

#include <cstdio>
#include <cstdint>
#include <cassert>

#include "cuda_runtime.h"

#include "cudabsp.h"
#include "cudamatrix.h"

#include "cudautils.h"


static inline __device__ __host__ void print_face(
        BSP::DFace* pFace,
        size_t faceIndex
        ) {

    char* c = reinterpret_cast<char*>(pFace);

    printf(
        "Face %u: \n"
        //"\t first 8 bytes: %x %x %x %x %x %x %x %x\n"
        "\t addr: %p\n"
        "\t firstEdge addr: %p\n"
        "\t planeNum: %u\n"
        "\t side: %u\n"
        "\t onNode: %u\n"
        "\t firstEdge: %d\n"
        "\t numEdges: %d\n"
        "\t texInfo: %d\n"
        "\t dispInfo: %d\n"
        "\t surfaceFogVolumeID: %d\n"
        "\t styles: %x, %x, %x, %x\n"
        "\t lightOffset: %d\n"
        "\t area: %f\n"
        "\t mins: (%d, %d)\n"
        "\t size: %d x %d\n"
        "\t origFace: %d\n"
        "\t numPrims: %u\n"
        "\t firstPrimID: %u\n"
        "\t smoothingGroups: %x\n",
        static_cast<unsigned int>(faceIndex),
        //static_cast<int>(c[0]),
        //static_cast<int>(c[1]),
        //static_cast<int>(c[2]),
        //static_cast<int>(c[3]),
        //static_cast<int>(c[4]),
        //static_cast<int>(c[5]),
        //static_cast<int>(c[6]),
        //static_cast<int>(c[7]),
        pFace,
        &pFace->firstEdge,
        static_cast<unsigned int>(pFace->planeNum),
        static_cast<unsigned int>(pFace->side),
        static_cast<unsigned int>(pFace->onNode),
        static_cast<int>(pFace->firstEdge),
        static_cast<int>(pFace->numEdges),
        static_cast<int>(pFace->texInfo),
        static_cast<int>(pFace->dispInfo),
        static_cast<int>(pFace->surfaceFogVolumeID),
        static_cast<int>(pFace->styles[0]),
        static_cast<int>(pFace->styles[1]),
        static_cast<int>(pFace->styles[2]),
        static_cast<int>(pFace->styles[3]),
        static_cast<int>(pFace->lightOffset),
        pFace->area,
        static_cast<int>(pFace->lightmapTextureMinsInLuxels[0]),
        static_cast<int>(pFace->lightmapTextureMinsInLuxels[1]),
        static_cast<int>(pFace->lightmapTextureSizeInLuxels[0]),
        static_cast<int>(pFace->lightmapTextureSizeInLuxels[1]),
        static_cast<int>(pFace->origFace),
        static_cast<unsigned int>(pFace->numPrims),
        static_cast<unsigned int>(pFace->firstPrimID),
        static_cast<int>(pFace->smoothingGroups)
    );
}


namespace CUDABSP {
    CUDABSP* make_cudabsp(const BSP::BSP& bsp) {
        CUDABSP cudaBSP;
        
        // To detect corruption.
        cudaBSP.tag = TAG;

        /* Compute the sizes of all the necessary arrays. */
        cudaBSP.numModels = bsp.get_models().size();
        cudaBSP.numPlanes = bsp.get_planes().size();
        cudaBSP.numVertices = bsp.get_vertices().size();
        cudaBSP.numEdges = bsp.get_edges().size();
        cudaBSP.numSurfEdges = bsp.get_surfedges().size();
        cudaBSP.numFaces = bsp.get_dfaces().size();
        cudaBSP.numLightSamples = bsp.get_lightsamples().size();
        cudaBSP.numTexInfos = bsp.get_texinfos().size();
        cudaBSP.numTexDatas = bsp.get_texdatas().size();
        cudaBSP.numLeaves = bsp.get_leaves().size();
        cudaBSP.numWorldLights = bsp.get_worldlights().size();

        size_t modelsSize = sizeof(BSP::DModel) * cudaBSP.numModels;
        size_t planesSize = sizeof(BSP::DPlane) * cudaBSP.numPlanes;
        size_t verticesSize = sizeof(float3) * cudaBSP.numVertices;
        size_t edgesSize = sizeof(BSP::DEdge) * cudaBSP.numEdges;
        size_t surfEdgesSize = sizeof(int32_t) * cudaBSP.numSurfEdges;
        size_t facesSize = sizeof(BSP::DFace) * cudaBSP.numFaces;
        size_t lightSamplesSize
            = sizeof(BSP::RGBExp32) * cudaBSP.numLightSamples;
        size_t texInfosSize = sizeof(BSP::TexInfo) * cudaBSP.numTexInfos;
        size_t texDatasSize = sizeof(BSP::DTexData) * cudaBSP.numTexDatas;
        size_t leavesSize = sizeof(BSP::DLeaf) * cudaBSP.numLeaves;
        size_t worldLightsSize
            = sizeof(BSP::DWorldLight) * cudaBSP.numWorldLights;

        /* Copy the BSP's data to device memory. */
        CUDA_CHECK_ERROR(cudaMalloc(&cudaBSP.models, modelsSize));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                cudaBSP.models, bsp.get_models().data(), modelsSize,
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK_ERROR(cudaMalloc(&cudaBSP.planes, planesSize));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                cudaBSP.planes, bsp.get_planes().data(), planesSize,
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK_ERROR(cudaMalloc(&cudaBSP.vertices, verticesSize));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                cudaBSP.vertices, bsp.get_vertices().data(),
                verticesSize,
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK_ERROR(cudaMalloc(&cudaBSP.edges, edgesSize));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                cudaBSP.edges, bsp.get_edges().data(),
                edgesSize,
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK_ERROR(cudaMalloc(&cudaBSP.surfEdges, surfEdgesSize));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                cudaBSP.surfEdges, bsp.get_surfedges().data(),
                surfEdgesSize,
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK_ERROR(cudaMalloc(&cudaBSP.faces, facesSize));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                cudaBSP.faces, bsp.get_dfaces().data(),
                facesSize,
                cudaMemcpyHostToDevice
            )
        );

        /* Special routine for st/xyz matrices */
        std::vector<CUDAMatrix::CUDAMatrix<double, 3, 3>> xyzMatrices;

        size_t xyzMatricesSize
            = sizeof(CUDAMatrix::CUDAMatrix<double, 3, 3>) * cudaBSP.numFaces;

        for (size_t i=0; i<cudaBSP.numFaces; i++) {
            const BSP::Face& face = bsp.get_faces()[i];

            gmtl::Matrix<double, 3, 3> xyzMatrix;
            face.make_st_xyz_matrix(xyzMatrix);

            xyzMatrices.push_back(CUDAMatrix::CUDAMatrix<double, 3, 3>());

            for (int row=0; row<3; row++) {
                for (int col=0; col<3; col++) {
                    xyzMatrices.back()[row][col] = xyzMatrix[row][col];
                }
            }
        }

        assert(
            xyzMatricesSize ==
            xyzMatrices.size() * sizeof(CUDAMatrix::CUDAMatrix<double, 3, 3>)
        );

        CUDA_CHECK_ERROR(cudaMalloc(&cudaBSP.xyzMatrices, xyzMatricesSize));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                cudaBSP.xyzMatrices, xyzMatrices.data(),
                xyzMatricesSize,
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK_ERROR(
            cudaMalloc(
                &cudaBSP.lightSamples,
                lightSamplesSize
            )
        );
        // Don't need to copy light samples since we're computing them.

        CUDA_CHECK_ERROR(cudaMalloc(&cudaBSP.texInfos, texInfosSize));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                cudaBSP.texInfos, bsp.get_texinfos().data(),
                texInfosSize,
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK_ERROR(cudaMalloc(&cudaBSP.texDatas, texDatasSize));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                cudaBSP.texDatas, bsp.get_texdatas().data(),
                texDatasSize,
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK_ERROR(cudaMalloc(&cudaBSP.leaves, leavesSize));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                cudaBSP.leaves, bsp.get_leaves().data(),
                leavesSize,
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK_ERROR(cudaMalloc(&cudaBSP.worldLights, worldLightsSize));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                cudaBSP.worldLights, bsp.get_worldlights().data(),
                worldLightsSize,
                cudaMemcpyHostToDevice
            )
        );

        /* Copy the CUDABSP structure to device memory. */
        CUDABSP* pCudaBSP;
        CUDA_CHECK_ERROR(cudaMalloc(&pCudaBSP, sizeof(CUDABSP)));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                pCudaBSP, &cudaBSP, sizeof(CUDABSP),
                cudaMemcpyHostToDevice
            )
        );

        return pCudaBSP;
    }

    void destroy_cudabsp(CUDABSP* pCudaBSP) {
        /* We need a host copy of this to access the internal pointers. */
        CUDABSP cudaBSP;
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                &cudaBSP, pCudaBSP, sizeof(CUDABSP),
                cudaMemcpyDeviceToHost
            )
        );

        /* Free the internal arrays. */
        cudaFree(cudaBSP.models);
        cudaFree(cudaBSP.planes);
        cudaFree(cudaBSP.vertices);
        cudaFree(cudaBSP.edges);
        cudaFree(cudaBSP.surfEdges);
        cudaFree(cudaBSP.faces);
        cudaFree(cudaBSP.lightSamples);
        cudaFree(cudaBSP.texInfos);
        cudaFree(cudaBSP.texDatas);
        cudaFree(cudaBSP.leaves);
        cudaFree(cudaBSP.worldLights);

        /* Free the device pointer itself. */
        cudaFree(pCudaBSP);
    }

    void update_bsp(BSP::BSP& bsp, CUDABSP* pCudaBSP) {
        CUDABSP cudaBSP;
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                &cudaBSP, pCudaBSP, sizeof(CUDABSP),
                cudaMemcpyDeviceToHost
            )
        );

        //std::vector<BSP::RGBExp32> lightSamples(cudaBSP.numLightSamples);

        CUDA_CHECK_ERROR(
            cudaMemcpy(
                //lightSamples.data(), cudaBSP.lightSamples,
                bsp.get_lightsamples().data(), cudaBSP.lightSamples,
                sizeof(BSP::RGBExp32) * cudaBSP.numLightSamples,
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK_ERROR(
            cudaMemcpy(
                bsp.get_dfaces().data(), cudaBSP.faces,
                sizeof(BSP::DFace) * cudaBSP.numFaces,
                cudaMemcpyDeviceToHost
            )
        );

        //std::cout << cudaBSP.numLightSamples << std::endl;
        //std::cout << bsp.get_lightsamples().size() << std::endl;

        //for (BSP::RGBExp32& sample : bsp.get_lightsamples()) {
        //    std::cout << "(" 
        //        << static_cast<int>(sample.r) << ", "
        //        << static_cast<int>(sample.g) << ", "
        //        << static_cast<int>(sample.b) << ") * 2^"
        //        << static_cast<int>(sample.exp) << std::endl;
        //}
    }
}
