#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include <iostream>
#include <thread>
#include <chrono>

#include "bsp.h"

#include "cudarad.h"

#include "cudabsp.h"
#include "cudamatrix.h"
#include "raytracer.h"

#include "cudautils.h"


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
};


namespace CUDARAD {
    static std::unique_ptr<RayTracer::CUDARayTracer> g_pRayTracer;
    static __device__ RayTracer::CUDARayTracer* g_pDeviceRayTracer;
}


static __device__ inline float attenuate(
        BSP::DWorldLight& light,
        float dist
        ) {
        
    float c = light.constantAtten;
    float l = light.linearAtten;
    float q = light.quadraticAtten;

    return c + l * dist + q * dist * dist;
}


static __device__ float3 xyz_from_st(FaceInfo& faceInfo, size_t s, size_t t) {
    BSP::DFace& face = faceInfo.face;
    BSP::DPlane& plane = faceInfo.plane;
    BSP::TexInfo& texInfo = faceInfo.texInfo;
    CUDAMatrix::CUDAMatrix<double, 3, 3>& Ainv = faceInfo.Ainv;

    float sOffset = texInfo.lightmapVecs[0][3];
    float tOffset = texInfo.lightmapVecs[1][3];

    float sMin = face.lightmapTextureMinsInLuxels[0];
    float tMin = face.lightmapTextureMinsInLuxels[1];

    CUDAMatrix::CUDAMatrix<double, 3, 1> B;

    B[0][0] = s - sOffset + sMin;
    B[1][0] = t - tOffset + tMin;
    B[2][0] = plane.dist;

    CUDAMatrix::CUDAMatrix<double, 3, 1> result = Ainv * B;

    return make_float3(result[0][0], result[1][0], result[2][0]);
}


static __device__ BSP::RGBExp32 lightsample_from_rgb(float3 color) {
    uint32_t r = static_cast<uint32_t>(color.x);
    uint32_t g = static_cast<uint32_t>(color.y);
    uint32_t b = static_cast<uint32_t>(color.z);

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
        exp,
    };
}


namespace DirectLighting {
    __device__ float3 sample_at(
            CUDABSP::CUDABSP& cudaBSP,
            FaceInfo& faceInfo,
            size_t s, size_t t
            ) {
        
        //const float EPSILON = 1e-6;

        float3 samplePos = xyz_from_st(faceInfo, s, t);

        samplePos += faceInfo.faceNorm * 1e-3;

        float3 result = make_float3(0.0, 0.0, 0.0);

        for (size_t lightIndex=0;
                 lightIndex<cudaBSP.numWorldLights;
                 lightIndex++
                 ) {
                
            BSP::DWorldLight& light = cudaBSP.worldLights[lightIndex];

            float3 lightPos = make_float3(
                light.origin.x,
                light.origin.y,
                light.origin.z
            );

            float3 diff = samplePos - lightPos;

            /*
             * This light is on the wrong side of the current face.
             * There's no way it could possibly light this sample.
             */
            if (dot(diff, faceInfo.faceNorm) >= 0.0) {
                continue;
            }

            bool lightBlocked = CUDARAD::g_pDeviceRayTracer->LOS_blocked(
                lightPos, samplePos
            );

            if (lightBlocked) {
                // This light can't be seen from the position of the sample.
                // Ignore it.
                continue;
            }

            /* I CAN SEE THE LIGHT */

            float dist = len(diff);
            float attenuation = attenuate(light, dist);

            result.x += light.intensity.x * 255.0 / attenuation;    // r
            result.y += light.intensity.y * 255.0 / attenuation;    // g
            result.z += light.intensity.z * 255.0 / attenuation;    // b
        }

        //printf(
        //    "Sample at (%u, %u) for Face %u: (%f, %f, %f)\n",
        //    static_cast<unsigned int>(s),
        //    static_cast<unsigned int>(t),
        //    static_cast<unsigned int>(faceIndex),
        //    result.x, result.y, result.z
        //);

        return result;
    }

    __global__ void map_faces(
            CUDABSP::CUDABSP* pCudaBSP,
            size_t* pFacesCompleted
            ) {
            
        bool primaryThread = (threadIdx.x == 0 && threadIdx.y == 0);

        if (pCudaBSP->tag != CUDABSP::TAG) {
            if (primaryThread) {
                printf("Invalid CUDABSP Tag: %x\n", pCudaBSP->tag);
            }
            return;
        }

        __shared__ FaceInfo faceInfo;

        if (primaryThread) {
            // Map block numbers to faces.
            faceInfo.faceIndex = blockIdx.x;

            faceInfo.face = pCudaBSP->faces[faceInfo.faceIndex];
            faceInfo.plane = pCudaBSP->planes[faceInfo.face.planeNum];
            faceInfo.texInfo = pCudaBSP->texInfos[faceInfo.face.texInfo];
            faceInfo.Ainv = pCudaBSP->xyzMatrices[faceInfo.faceIndex];

            BSP::DFace& face = faceInfo.face;
            BSP::DPlane& plane = faceInfo.plane;
            
            faceInfo.faceNorm = make_float3(
                plane.normal.x,
                plane.normal.y,
                plane.normal.z
            );

            faceInfo.lightmapWidth = face.lightmapTextureSizeInLuxels[0] + 1;
            faceInfo.lightmapHeight = face.lightmapTextureSizeInLuxels[1] + 1;

            size_t& lightmapWidth = faceInfo.lightmapWidth;
            size_t& lightmapHeight = faceInfo.lightmapHeight;

            faceInfo.lightmapSize = lightmapWidth * lightmapHeight;
            faceInfo.lightmapStartIndex
                = face.lightOffset / sizeof(BSP::RGBExp32);
            faceInfo.totalLight = make_float3(0.0, 0.0, 0.0);

            //printf(
            //    "Processing Face %u...\n",
            //    static_cast<unsigned int>(faceInfo.faceIndex)
            //);
        }

        __syncthreads();

        /* Take a sample at each lightmap luxel. */
        for (size_t i=0; i<faceInfo.lightmapHeight; i+=blockDim.y) {
            size_t t = i + threadIdx.y;

            if (t >= faceInfo.lightmapHeight) {
                continue;
            }

            for (size_t j=0; j<faceInfo.lightmapWidth; j+=blockDim.x) {
                size_t s = j + threadIdx.x;
                
                if (s >= faceInfo.lightmapWidth) {
                    continue;
                }

                float3 color = sample_at(*pCudaBSP, faceInfo, s, t);

                size_t& lightmapStartIndex = faceInfo.lightmapStartIndex;
                size_t sampleIndex = t * faceInfo.lightmapWidth + s;

                pCudaBSP->lightSamples[lightmapStartIndex + sampleIndex]
                    = lightsample_from_rgb(color);

                atomicAdd(&faceInfo.totalLight.x, color.x);
                atomicAdd(&faceInfo.totalLight.y, color.y);
                atomicAdd(&faceInfo.totalLight.z, color.z);
            }
        }

        __syncthreads();

        if (primaryThread) {
            faceInfo.avgLight = faceInfo.totalLight;

            faceInfo.avgLight.x /= faceInfo.lightmapSize;
            faceInfo.avgLight.y /= faceInfo.lightmapSize;
            faceInfo.avgLight.z /= faceInfo.lightmapSize;

            pCudaBSP->lightSamples[faceInfo.lightmapStartIndex - 1]
                = lightsample_from_rgb(faceInfo.avgLight);

            // Still have no idea how this works. But if we don't do this,
            // EVERYTHING becomes a disaster...
            faceInfo.face.styles[0] = 0x00;
            faceInfo.face.styles[1] = 0xFF;
            faceInfo.face.styles[2] = 0xFF;
            faceInfo.face.styles[3] = 0xFF;

            /* Copy our changes back to the CUDABSP. */
            pCudaBSP->faces[faceInfo.faceIndex] = faceInfo.face;

            atomicAdd(reinterpret_cast<unsigned int*>(pFacesCompleted), 1);
            __threadfence_system();
        }

        //printf(
        //    "Lightmap offset for face %u: %u\n",
        //    static_cast<unsigned int>(faceIndex),
        //    static_cast<unsigned int>(lightmapStartIndex)
        //);

        //printf("%u\n", static_cast<unsigned int>(*pFacesCompleted));
    }
}


namespace CUDARAD {
    void init(BSP::BSP& bsp) {
        std::cout << "Setting up ray-trace acceleration structure... "
            << std::flush;

        using Clock = std::chrono::high_resolution_clock;

        auto start = Clock::now();

        g_pRayTracer = std::unique_ptr<RayTracer::CUDARayTracer>(
            new RayTracer::CUDARayTracer()
        );

        std::vector<RayTracer::Triangle> triangles;

        /* Put all of the BSP's face triangles into the ray-tracer. */
        int i = 0;
        for (const BSP::Face& face : bsp.get_faces()) {
            std::vector<BSP::Edge>::const_iterator pEdge
                = face.get_edges().begin();

            BSP::Vec3<float> vertex1 = (pEdge++)->vertex1;
            BSP::Vec3<float> vertex2;
            BSP::Vec3<float> vertex3 = (pEdge++)->vertex1;

            do {
                vertex2 = vertex3;
                vertex3 = (pEdge++)->vertex1;

                RayTracer::Triangle tri {
                    {
                        make_float3(vertex1.x, vertex1.y, vertex1.z),
                        make_float3(vertex2.x, vertex2.y, vertex2.z),
                        make_float3(vertex3.x, vertex3.y, vertex3.z),
                    },
                };

                triangles.push_back(tri);

            } while (pEdge != face.get_edges().end());

            i++;
        }

        g_pRayTracer->add_triangles(triangles);

        auto end = Clock::now();
        std::chrono::milliseconds ms
            = std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start
            );

        std::cout << "Done! (" << ms.count() << "ms)" << std::endl;

        std::cout << "Moving ray-tracer to device..." << std::endl;

        RayTracer::CUDARayTracer* pDeviceRayTracer;

        CUDA_CHECK_ERROR(
            cudaMalloc(&pDeviceRayTracer, sizeof(RayTracer::CUDARayTracer))
        );
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                pDeviceRayTracer, g_pRayTracer.get(),
                sizeof(RayTracer::CUDARayTracer),
                cudaMemcpyHostToDevice
            )
        );
        CUDA_CHECK_ERROR(
            cudaMemcpyToSymbol(
                g_pDeviceRayTracer, &pDeviceRayTracer,
                sizeof(RayTracer::CUDARayTracer*), 0,
                cudaMemcpyHostToDevice
            )
        );
    }

    void cleanup(void) {
        CUDA_CHECK_ERROR(cudaFree(g_pDeviceRayTracer));
        g_pRayTracer = nullptr;
    }

    void compute_direct_lighting(
            BSP::BSP& bsp,
            CUDABSP::CUDABSP* pCudaBSP,
            std::vector<BSP::RGBExp32>& lightSamples
            ) {

        volatile size_t* pFacesCompleted;
        CUDA_CHECK_ERROR(
            cudaHostAlloc(
                &pFacesCompleted, sizeof(size_t),
                cudaHostAllocMapped
            )
        );

        *pFacesCompleted = 0;

        volatile size_t* pDeviceFacesCompleted;
        CUDA_CHECK_ERROR(
            cudaHostGetDevicePointer(
                const_cast<size_t**>(&pDeviceFacesCompleted),
                const_cast<size_t*>(pFacesCompleted),
                0
            )
        );

        const size_t BLOCK_WIDTH = 16;
        const size_t BLOCK_HEIGHT = 16;

        size_t numFaces = bsp.get_faces().size();

        dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);

        std::cout << "Launching "
            << numFaces * BLOCK_WIDTH * BLOCK_HEIGHT << " threads ("
            << numFaces << " faces)..."
            << std::endl;

        cudaEvent_t startEvent;
        cudaEvent_t stopEvent;

        CUDA_CHECK_ERROR(cudaEventCreate(&startEvent));
        CUDA_CHECK_ERROR(cudaEventCreate(&stopEvent));

        CUDA_CHECK_ERROR(cudaEventRecord(startEvent));

        KERNEL_LAUNCH(
            DirectLighting::map_faces,
            numFaces, blockDim,
            pCudaBSP, const_cast<size_t*>(pDeviceFacesCompleted)
        );

        CUDA_CHECK_ERROR(cudaPeekAtLastError());

        flush_wddm_queue();

        size_t lastFacesCompleted = 0;
        size_t facesCompleted;

        /* Progress notification logic */
        do {
            CUDA_CHECK_ERROR(cudaPeekAtLastError());

            facesCompleted = *pFacesCompleted;

            if (facesCompleted > lastFacesCompleted) {
                std::cout << facesCompleted << "/"
                    << numFaces
                    << " faces processed..." << std::endl;
            }

            lastFacesCompleted = facesCompleted;

            std::this_thread::sleep_for(std::chrono::milliseconds(5));

        } while (facesCompleted < numFaces);

        CUDA_CHECK_ERROR(cudaEventRecord(stopEvent));
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        float time;
        CUDA_CHECK_ERROR(cudaEventElapsedTime(&time, startEvent, stopEvent));

        std::cout << "Done! (" << time << "ms)" << std::endl;

        cudaFreeHost(const_cast<size_t*>(pFacesCompleted));
    }

    void bounce_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP) {
        std::cerr << "Radiosity light bounces not implemented!" << std::endl;
    }
}
