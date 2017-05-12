#include <iostream>

#include <cstdio>
#include <cstdint>
#include <cassert>

#include "cuda_runtime.h"

#include "cudabsp.h"
#include "cudamatrix.h"

#include "cudautils.h"


//static inline __device__ __host__ void print_face(
//        BSP::DFace* pFace,
//        size_t faceIndex
//        ) {
//
//    //char* c = reinterpret_cast<char*>(pFace);
//
//    printf(
//        "Face %u: \n"
//        //"\t first 8 bytes: %x %x %x %x %x %x %x %x\n"
//        "\t addr: %p\n"
//        "\t firstEdge addr: %p\n"
//        "\t planeNum: %u\n"
//        "\t side: %u\n"
//        "\t onNode: %u\n"
//        "\t firstEdge: %d\n"
//        "\t numEdges: %d\n"
//        "\t texInfo: %d\n"
//        "\t dispInfo: %d\n"
//        "\t surfaceFogVolumeID: %d\n"
//        "\t styles: %x, %x, %x, %x\n"
//        "\t lightOffset: %d\n"
//        "\t area: %f\n"
//        "\t mins: (%d, %d)\n"
//        "\t size: %d x %d\n"
//        "\t origFace: %d\n"
//        "\t numPrims: %u\n"
//        "\t firstPrimID: %u\n"
//        "\t smoothingGroups: %x\n",
//        static_cast<unsigned int>(faceIndex),
//        //static_cast<int>(c[0]),
//        //static_cast<int>(c[1]),
//        //static_cast<int>(c[2]),
//        //static_cast<int>(c[3]),
//        //static_cast<int>(c[4]),
//        //static_cast<int>(c[5]),
//        //static_cast<int>(c[6]),
//        //static_cast<int>(c[7]),
//        pFace,
//        &pFace->firstEdge,
//        static_cast<unsigned int>(pFace->planeNum),
//        static_cast<unsigned int>(pFace->side),
//        static_cast<unsigned int>(pFace->onNode),
//        static_cast<int>(pFace->firstEdge),
//        static_cast<int>(pFace->numEdges),
//        static_cast<int>(pFace->texInfo),
//        static_cast<int>(pFace->dispInfo),
//        static_cast<int>(pFace->surfaceFogVolumeID),
//        static_cast<int>(pFace->styles[0]),
//        static_cast<int>(pFace->styles[1]),
//        static_cast<int>(pFace->styles[2]),
//        static_cast<int>(pFace->styles[3]),
//        static_cast<int>(pFace->lightOffset),
//        pFace->area,
//        static_cast<int>(pFace->lightmapTextureMinsInLuxels[0]),
//        static_cast<int>(pFace->lightmapTextureMinsInLuxels[1]),
//        static_cast<int>(pFace->lightmapTextureSizeInLuxels[0]),
//        static_cast<int>(pFace->lightmapTextureSizeInLuxels[1]),
//        static_cast<int>(pFace->origFace),
//        static_cast<unsigned int>(pFace->numPrims),
//        static_cast<unsigned int>(pFace->firstPrimID),
//        static_cast<int>(pFace->smoothingGroups)
//    );
//}


namespace CUDABSP {
    __device__ BSP::RGBExp32 rgbexp32_from_float3(float3 color) {
        if ((color.x < 1.0 || color.y < 1.0 || color.z < 1.0)
                && color.x > 1e-3 && color.y > 1e-3 && color.z > 1e-3) {
            int8_t exp = 0;

            while ((color.x < 1.0 || color.y < 1.0 || color.z < 1.0)
                    && color.x > 1e-3 && color.y > 1e-3 && color.z > 1e-3) {
                color *= 2.0;
                exp--;
            }

            return BSP::RGBExp32 {
                static_cast<uint8_t>(color.x),
                static_cast<uint8_t>(color.y),
                static_cast<uint8_t>(color.z),
                exp,
            };
        }
        else {
            uint64_t r = static_cast<uint64_t>(color.x);
            uint64_t g = static_cast<uint64_t>(color.y);
            uint64_t b = static_cast<uint64_t>(color.z);

            int8_t exp = 0;

            while (r > 255 || g > 255 || b > 255) {
                r >>= 1;
                g >>= 1;
                b >>= 1;

                exp++;
            }

            return BSP::RGBExp32 {
                static_cast<uint8_t>(r),
                static_cast<uint8_t>(g),
                static_cast<uint8_t>(b),
                exp
            };
        }
    }

    __global__ void map_lightsamples(
            float3* lightSamples,
            BSP::RGBExp32* rgbExp32LightSamples,
            size_t numLightSamples
            ) {

        size_t index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= numLightSamples) {
            return;
        }

        float3 sample = lightSamples[index];

        rgbExp32LightSamples[index] = rgbexp32_from_float3(sample);
    }

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
        cudaBSP.numAmbientLightSamples = bsp.get_ambient_samples().size();
        cudaBSP.numWorldLights = bsp.get_worldlights().size();

        size_t modelsSize = sizeof(BSP::DModel) * cudaBSP.numModels;
        size_t planesSize = sizeof(BSP::DPlane) * cudaBSP.numPlanes;
        size_t verticesSize = sizeof(float3) * cudaBSP.numVertices;
        size_t edgesSize = sizeof(BSP::DEdge) * cudaBSP.numEdges;
        size_t surfEdgesSize = sizeof(int32_t) * cudaBSP.numSurfEdges;
        size_t facesSize = sizeof(BSP::DFace) * cudaBSP.numFaces;
        size_t lightSamplesSize
            = sizeof(float3) * cudaBSP.numLightSamples;
        size_t rgbExp32LightSamplesSize
            = sizeof(BSP::RGBExp32) * cudaBSP.numLightSamples;
        size_t texInfosSize = sizeof(BSP::TexInfo) * cudaBSP.numTexInfos;
        size_t texDatasSize = sizeof(BSP::DTexData) * cudaBSP.numTexDatas;
        size_t leavesSize = sizeof(BSP::DLeaf) * cudaBSP.numLeaves;
        size_t ambientIndicesSize
            = sizeof(BSP::DLeafAmbientIndex) * cudaBSP.numLeaves;
        size_t ambientLightSamplesSize
            = sizeof(BSP::DLeafAmbientLighting)
                * cudaBSP.numAmbientLightSamples;
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

        for (const BSP::Face& face : bsp.get_faces()) {
            const gmtl::Matrix<double, 3, 3>& xyzMatrix
                = face.get_st_xyz_matrix();

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


        CUDA_CHECK_ERROR(
            cudaMalloc(
                &cudaBSP.rgbExp32LightSamples,
                rgbExp32LightSamplesSize
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

        CUDA_CHECK_ERROR(
            cudaMalloc(&cudaBSP.ambientIndices, ambientIndicesSize)
        );
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                cudaBSP.ambientIndices, bsp.get_ambient_indices().data(),
                ambientIndicesSize,
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK_ERROR(
            cudaMalloc(&cudaBSP.ambientLightSamples, ambientLightSamplesSize)
        );
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                cudaBSP.ambientLightSamples, bsp.get_ambient_samples().data(),
                ambientLightSamplesSize,
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
        CUDA_CHECK_ERROR(cudaFree(cudaBSP.models));
        CUDA_CHECK_ERROR(cudaFree(cudaBSP.planes));
        CUDA_CHECK_ERROR(cudaFree(cudaBSP.vertices));
        CUDA_CHECK_ERROR(cudaFree(cudaBSP.edges));
        CUDA_CHECK_ERROR(cudaFree(cudaBSP.surfEdges));
        CUDA_CHECK_ERROR(cudaFree(cudaBSP.faces));
        CUDA_CHECK_ERROR(cudaFree(cudaBSP.xyzMatrices));
        CUDA_CHECK_ERROR(cudaFree(cudaBSP.lightSamples));
        CUDA_CHECK_ERROR(cudaFree(cudaBSP.rgbExp32LightSamples));
        CUDA_CHECK_ERROR(cudaFree(cudaBSP.texInfos));
        CUDA_CHECK_ERROR(cudaFree(cudaBSP.texDatas));
        CUDA_CHECK_ERROR(cudaFree(cudaBSP.leaves));
        CUDA_CHECK_ERROR(cudaFree(cudaBSP.ambientIndices));
        CUDA_CHECK_ERROR(cudaFree(cudaBSP.ambientLightSamples));
        CUDA_CHECK_ERROR(cudaFree(cudaBSP.worldLights));

        /* Free the device pointer itself. */
        CUDA_CHECK_ERROR(cudaFree(pCudaBSP));
    }

    void convert_lightsamples(CUDABSP* pCudaBSP) {
        CUDABSP cudaBSP;

        CUDA_CHECK_ERROR(
            cudaMemcpy(
                &cudaBSP, pCudaBSP, sizeof(CUDABSP),
                cudaMemcpyDeviceToHost
            )
        );

        size_t BLOCK_WIDTH = 1024;
        size_t numBlocks = div_ceil(cudaBSP.numLightSamples, BLOCK_WIDTH);

        KERNEL_LAUNCH(
            map_lightsamples,
            numBlocks, BLOCK_WIDTH,
            cudaBSP.lightSamples, cudaBSP.rgbExp32LightSamples,
            cudaBSP.numLightSamples
        );

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }

    void update_bsp(BSP::BSP& bsp, CUDABSP* pCudaBSP) {
        CUDABSP cudaBSP;
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                &cudaBSP, pCudaBSP, sizeof(CUDABSP),
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK_ERROR(
            cudaMemcpy(
                bsp.get_lightsamples().data(), cudaBSP.rgbExp32LightSamples,
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

        CUDA_CHECK_ERROR(
            cudaMemcpy(
                bsp.get_ambient_samples().data(), cudaBSP.ambientLightSamples,
                sizeof(BSP::DLeafAmbientLighting)
                    * cudaBSP.numAmbientLightSamples,
                cudaMemcpyDeviceToHost
            )
        );
    }
}
