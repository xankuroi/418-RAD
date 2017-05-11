#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include <iostream>
#include <thread>
#include <chrono>

#include "cudarad.h"

#include "bsp.h"

#include "cudabsp.h"
#include "cudamatrix.h"
#include "raytracer.h"

#include "cudautils.h"

#define MAX_ITER 100


namespace CUDARAD {
    static std::unique_ptr<RayTracer::CUDARayTracer> g_pRayTracer;
    static __device__ RayTracer::CUDARayTracer* g_pDeviceRayTracer;

    __device__ FaceInfo::FaceInfo() {};

    __device__ FaceInfo::FaceInfo(
            CUDABSP::CUDABSP& cudaBSP,
            size_t faceIndex
            ) :
            faceIndex(faceIndex),
            face(cudaBSP.faces[faceIndex]),
            plane(cudaBSP.planes[face.planeNum]),
            texInfo(cudaBSP.texInfos[face.texInfo]),
            Ainv(cudaBSP.xyzMatrices[faceIndex]),
            faceNorm(
                make_float3(plane.normal.x, plane.normal.y, plane.normal.z)
            ),
            lightmapWidth(face.lightmapTextureSizeInLuxels[0] + 1),
            lightmapHeight(face.lightmapTextureSizeInLuxels[1] + 1),
            lightmapSize(lightmapWidth * lightmapHeight),
            lightmapStartIndex(face.lightOffset / sizeof(BSP::RGBExp32)),
            totalLight(make_float3(0.0, 0.0, 0.0)) {}

    __device__ float3 FaceInfo::xyz_from_st(float s, float t) {
        float sOffset = this->texInfo.lightmapVecs[0][3];
        float tOffset = this->texInfo.lightmapVecs[1][3];

        float sMin = this->face.lightmapTextureMinsInLuxels[0];
        float tMin = this->face.lightmapTextureMinsInLuxels[1];

        CUDAMatrix::CUDAMatrix<double, 3, 1> B;

        B[0][0] = s - sOffset + sMin;
        B[1][0] = t - tOffset + tMin;
        B[2][0] = this->plane.dist;

        CUDAMatrix::CUDAMatrix<double, 3, 1> result = this->Ainv * B;

        return make_float3(result[0][0], result[1][0], result[2][0]);
    }
}


namespace DirectLighting {
    static __device__ inline float attenuate(
            BSP::DWorldLight& light,
            float dist
            ) {

        float c = light.constantAtten;
        float l = light.linearAtten;
        float q = light.quadraticAtten;

        return c + l * dist + q * dist * dist;
    }

    static __device__ const float INV_GAMMA = 1.0 / 2.2;

    static __device__ inline float perceptual_from_linear(float linear) {
        return powf(linear, INV_GAMMA);
    }

    static __device__ float intensity(float3 rgb) {
        return perceptual_from_linear(
            dot(
                rgb / 255.0,
                make_float3(1.0, 1.0, 1.0)
                //make_float3(0.299, 0.587, 0.114)
            )
        );
    }

    __device__ float3 sample_at(
            CUDABSP::CUDABSP& cudaBSP,
            float3 samplePos,
            float3 sampleNormal=make_float3(0.0, 0.0, 0.0)
            ) {
<<<<<<< HEAD

        float3 samplePos = faceInfo.xyz_from_st(s, t);
=======
>>>>>>> 533c851478d777627bedebfc84a55e37b8d792c2

        //samplePos += faceInfo.faceNorm * 1e-3;

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
             * This light is on the wrong side of the current sample.
             * There's no way it could possibly light it.
             */
            if (len(sampleNormal) > 0.0 && dot(diff, sampleNormal) >= 0.0) {
                continue;
            }

            float dist = len(diff);
            float3 dir = diff / dist;

            float penumbraScale = 1.0;

            if (light.type == BSP::EMIT_SPOTLIGHT) {
                float3 lightNorm = make_float3(
                    light.normal.x,
                    light.normal.y,
                    light.normal.z
                );

                float lightDot = dot(dir, lightNorm);

                if (lightDot < light.stopdot2) {
                    /* This sample is outside the spotlight cone. */
                    continue;
                }
                else if (lightDot < light.stopdot) {
                    /* This sample is within the spotlight's penumbra. */
                    penumbraScale = (
                        (lightDot - light.stopdot2)
                        / (light.stopdot - light.stopdot2)
                    );
                    //penumbraScale = 100.0;
                }

                //if (lightIndex == cudaBSP.numWorldLights - 1) {
                //    printf(
                //        "(%f, %f, %f) is within spotlight!\n"
                //        "Pos: (%f, %f, %f)\n"
                //        "Norm: <%f, %f, %f> (<%f, %f, %f>)\n"
                //        "stopdot: %f; stopdot2: %f\n"
                //        "Dot between light and sample: %f\n",
                //        samplePos.x, samplePos.y, samplePos.z,
                //        lightPos.x, lightPos.y, lightPos.z,
                //        lightNorm.x, lightNorm.y, lightNorm.z,
                //        light.normal.x, light.normal.y, light.normal.z,
                //        light.stopdot, light.stopdot2,
                //        lightDot
                //    );
                //}
            }

            const float EPSILON = 1e-3;

            // Nudge the sample position towards the light slightly, to avoid
            // colliding with triangles that directly contain the sample
            // position.
            samplePos -= dir * EPSILON;

            bool lightBlocked = CUDARAD::g_pDeviceRayTracer->LOS_blocked(
                lightPos, samplePos
            );

            if (lightBlocked) {
                // This light can't be seen from the position of the sample.
                // Ignore it.
                continue;
            }

            /* I CAN SEE THE LIGHT */
            float attenuation = attenuate(light, dist);

            float3 lightContribution = make_float3(
                light.intensity.x,  // r
                light.intensity.y,  // g
                light.intensity.z   // b
            );

            lightContribution *= penumbraScale * 255.0 / attenuation;

            result += lightContribution;
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

    __device__ float3 sample_at(
            CUDABSP::CUDABSP& cudaBSP,
            CUDARAD::FaceInfo& faceInfo,
            float s, float t
            ) {

        float3 samplePos = faceInfo.xyz_from_st(s, t);
        return sample_at(cudaBSP, samplePos, faceInfo.faceNorm);
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

        __shared__ CUDARAD::FaceInfo faceInfo;

        if (primaryThread) {
            // Map block numbers to faces.
            faceInfo = CUDARAD::FaceInfo(*pCudaBSP, blockIdx.x);

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

                float3 color = sample_at(
                    *pCudaBSP, faceInfo,
                    static_cast<float>(s),
                    static_cast<float>(t)
                );

                size_t& lightmapStart = faceInfo.lightmapStartIndex;
                size_t sampleIndex = t * faceInfo.lightmapWidth + s;

                pCudaBSP->lightSamples[lightmapStart + sampleIndex] = color;

                atomicAdd(&faceInfo.totalLight.x, color.x);
                atomicAdd(&faceInfo.totalLight.y, color.y);
                atomicAdd(&faceInfo.totalLight.z, color.z);
            }
        }

        __syncthreads();

        if (primaryThread) {
            faceInfo.avgLight = faceInfo.totalLight;
            faceInfo.avgLight /= static_cast<float>(faceInfo.lightmapSize);

            pCudaBSP->lightSamples[faceInfo.lightmapStartIndex - 1]
                = faceInfo.avgLight;

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

    static __device__ const float MIN_AA_GRADIENT = 0.0625;    // 1/16

    __global__ void map_faces_AA(CUDABSP::CUDABSP* pCudaBSP) {
        bool primaryThread = (threadIdx.x == 0 && threadIdx.y == 0);

        __shared__ CUDARAD::FaceInfo faceInfo;

        __shared__ size_t lightmapStart;
        __shared__ size_t width;
        __shared__ size_t height;

        __shared__ float3* results;

        if (primaryThread) {
            // Map block numbers to faces.
            faceInfo = CUDARAD::FaceInfo(*pCudaBSP, blockIdx.x);

            lightmapStart = faceInfo.lightmapStartIndex;
            width = faceInfo.lightmapWidth;
            height = faceInfo.lightmapHeight;

            results = new float3[width * height];
        }

        __syncthreads();

        for (size_t i=0; i<height; i+=blockDim.y) {
            size_t t = i + threadIdx.y;

            if (t >= height) {
                continue;
            }

            for (size_t j=0; j<width; j+=blockDim.x) {
                size_t s = j + threadIdx.x;

                if (s >= width) {
                    continue;
                }

                size_t sampleIndex = t * width + s;

                float3 sampleColor
                    = pCudaBSP->lightSamples[lightmapStart + sampleIndex];

                float sampleIntensity = intensity(sampleColor);

                /* Calculate the maximum gradient of this luxel. */
                float gradient = 0.0;

                for (int tOffset=-1; tOffset<=1; tOffset++) {
                    int neighborT = t + tOffset;

                    if (!(0 <= neighborT && neighborT < height)) {
                        continue;
                    }

                    for (int sOffset=-1; sOffset<=1; sOffset++) {
                        if (sOffset == 0 && tOffset == 0) {
                            continue;
                        }

                        int neighborS = s + sOffset;

                        if (!(0 <= neighborS && neighborS < width)) {
                            continue;
                        }

                        int neighborIndex
                            = neighborT * width + neighborS;

                        float neighborIntensity = intensity(
                            pCudaBSP->lightSamples[
                                lightmapStart + neighborIndex
                            ]
                        );

                        gradient = fmaxf(
                            gradient,
                            fabsf(neighborIntensity - sampleIntensity)
                        );
                    }
                }

                /*
                 * Don't bother antialiasing this sample if the gradient is
                 * low enough.
                 */
                if (gradient < MIN_AA_GRADIENT) {
                    results[sampleIndex] = sampleColor;
                    continue;
                }

                /*
                 * Box blur!
                 * Really stupid and potentially ugly, but really fast!
                 */

                float3 color = make_float3(0.0, 0.0, 0.0);

<<<<<<< HEAD
                //for (int tOffset=-1; tOffset<=1; tOffset++) {
                //    float blurT = static_cast<float>(t) + tOffset;

                //    for (int sOffset=-1; sOffset<=1; sOffset++) {
                //        float blurS = static_cast<float>(s) + sOffset;

                //        float3 blurColor;

                //        /*
                //         * Out of range!
                //         * We have no choice but to actually take a sample.
                //         */
                //        if (!(0 <= blurS && blurS < width)
                //                || !(0 <= blurT && blurT < height)) {
                //            blurColor = sample_at(
                //                *pCudaBSP, faceInfo,
                //                blurS, blurT
                //            );
                //        }
                //        else {
                //            size_t i = static_cast<size_t>(
                //                blurT * width + blurS
                //            );
                //            blurColor = pCudaBSP->lightSamples[
                //                lightmapStart + i
                //            ];
                //        }

                //        color += blurColor;
                //    }
                //}
=======
                for (int tOffset=-1; tOffset<=1; tOffset++) {
                    float blurT = static_cast<float>(t) + tOffset;
>>>>>>> 533c851478d777627bedebfc84a55e37b8d792c2

                    for (int sOffset=-1; sOffset<=1; sOffset++) {
                        float blurS = static_cast<float>(s) + sOffset;

                        float3 blurColor;

                        /*
                         * Out of range!
                         * We have no choice but to actually take a sample.
                         */
                        if (!(0 <= blurS && blurS < width)
                                || !(0 <= blurT && blurT < height)) {
                            blurColor = sample_at(
                                *pCudaBSP, faceInfo,
                                blurS, blurT
                            );
                        }
                        else {
                            size_t i = static_cast<size_t>(
                                blurT * width + blurS
                            );
                            blurColor = pCudaBSP->lightSamples[
                                lightmapStart + i
                            ];
                        }

                        color += blurColor;
                    }
                }

                /* Take the average of the box blur samples. */
                color /= 9.0;

                ///* Perform supersampling at this point. */
                //const size_t SUPERSAMPLE_WIDTH = 4;

                //float sStep = 2.0 / static_cast<float>(SUPERSAMPLE_WIDTH);
                //float tStep = 2.0 / static_cast<float>(SUPERSAMPLE_WIDTH);

                //float3 color = make_float3(0.0, 0.0, 0.0);

                //for (size_t ssi=0; ssi<SUPERSAMPLE_WIDTH; ssi++) {
                //    float tOffset = tStep * ssi - 1.0;

                //    for (size_t ssj=0; ssj<SUPERSAMPLE_WIDTH; ssj++) {
                //        float sOffset = sStep * ssj - 1.0;

                //        color += sample_at(
                //            *pCudaBSP, faceInfo,
                //            s + sOffset, t + tOffset
                //        );
                //    }
                //}

                //color /= SUPERSAMPLE_WIDTH * SUPERSAMPLE_WIDTH;

                results[sampleIndex] = color;
                //pCudaBSP->lightSamples[lightmapStart + sampleIndex] = color;
            }
        }

        __syncthreads();

        if (primaryThread) {
            /* Move the results back to the light samples array. */
            memcpy(
                pCudaBSP->lightSamples + faceInfo.lightmapStartIndex,
                results,
                sizeof(float3) * faceInfo.lightmapSize
            );

            delete[] results;
        }
    }
}


namespace BouncedLighting {
    static __device__ const float PI = 3.14159265358979323846264;
    static __device__ const float INV_PI = 0.31830988618379067153715;

    /**
     * Computes the form factor from a differential patch to a convex
     * polygonal patch.
     *
     * Thankfully, Source's polygons are always convex.
     *
     * Formula graciously stolen from Formula 81 of this book:
     * https://people.cs.kuleuven.be/~philip.dutre/GI/TotalCompendium.pdf
     *
     * ... and Formula 4.16 of this one:
     * https://books.google.com/books?id=zALK286TFXgC&lpg=PP1&pg=PA72#v=onepage&q&f=false
     */
    static __device__ float ff_diff_poly(
            float3 diffPos, float3 diffNorm,
            float3* vertices, size_t numVertices
            ) {

        float result = 0.0;

        for (size_t i=0; i<numVertices; i++) {
            float3 vertex1 = vertices[i] - diffPos;
            float3 vertex2 = vertices[(i + 1) % numVertices] - diffPos;

            float3 vertexCross = cross(vertex1, vertex2);
            float crossLen = len(vertexCross);

            vertexCross /= crossLen;

            float v1Len = len(vertex1);
            float v2Len = len(vertex2);

            float theta = asinf(crossLen / (v1Len * v2Len));

            result += dot(diffNorm, vertexCross) * theta;
        }

        result *= 0.5 * INV_PI;

        return result;
    }

    /** Computes the form factor between two differential patches. */
    static __device__ float ff_diff_diff(
            float3 diff1Pos, float3 diff1Norm,
            float3 diff2Pos, float3 diff2Norm
            ) {

        float3 delta = diff2Pos - diff1Pos;
        float invDist = 1.0 / len(delta);

        float3 dir = delta * invDist;

        return (
            dot(diff1Norm, dir) * -dot(diff2Norm, dir)
            * INV_PI * invDist * invDist
        );
    }
}


namespace AmbientLighting {
    static __device__ const float AMBIENT_SCALE = 0.0078125;    // 1/128

    __global__ void map_leaves(CUDABSP::CUDABSP* pCudaBSP) {
        size_t leafIndex = blockIdx.x;

        if (leafIndex >= pCudaBSP->numLeaves) {
            return;
        }

        BSP::DLeaf& leaf = pCudaBSP->leaves[leafIndex];

        if (leaf.contents & BSP::CONTENTS_SOLID) {
            return;
        }

        BSP::DLeafAmbientIndex& ambientIndex
            = pCudaBSP->ambientIndices[leafIndex];

        BSP::DLeafAmbientLighting* ambientSamples
            = &pCudaBSP->ambientLightSamples[ambientIndex.firstAmbientSample];

        for (size_t i=threadIdx.x;
                i<ambientIndex.ambientSampleCount;
                i+=blockDim.x) {

            if (i >= ambientIndex.ambientSampleCount) {
                return;
            }

            BSP::DLeafAmbientLighting& sample = ambientSamples[i];

            float3 leafMins = make_float3(
                leaf.mins[0], leaf.mins[1], leaf.mins[2]
            );

            float3 leafMaxs = make_float3(
                leaf.maxs[0], leaf.maxs[1], leaf.maxs[2]
            );

            float3 leafSize = leafMaxs - leafMins;

            float3 samplePos = leafMins + make_float3(
                leafSize.x * static_cast<float>(sample.x) / 255.0,
                leafSize.y * static_cast<float>(sample.y) / 255.0,
                leafSize.z * static_cast<float>(sample.z) / 255.0
            );

            //sample.cube.color[0] = BSP::RGBExp32 {1, 1, 1, -3};
            //sample.cube.color[1] = BSP::RGBExp32 {1, 1, 1, -3};
            //sample.cube.color[2] = BSP::RGBExp32 {1, 1, 1, -3};
            //sample.cube.color[3] = BSP::RGBExp32 {1, 1, 1, -3};
            //sample.cube.color[4] = BSP::RGBExp32 {1, 1, 1, -3};
            //sample.cube.color[5] = BSP::RGBExp32 {1, 1, 1, -3};

            // +X
            sample.cube.color[0] = CUDABSP::rgbexp32_from_float3(
                DirectLighting::sample_at(
                    *pCudaBSP,
                    samplePos,
                    make_float3(1.0, 0.0, 0.0)
                ) * AMBIENT_SCALE
            );

            // -X
            sample.cube.color[1] = CUDABSP::rgbexp32_from_float3(
                DirectLighting::sample_at(
                    *pCudaBSP,
                    samplePos,
                    make_float3(-1.0, 0.0, 0.0)
                ) * AMBIENT_SCALE
            );

            // +Y
            sample.cube.color[2] = CUDABSP::rgbexp32_from_float3(
                DirectLighting::sample_at(
                    *pCudaBSP,
                    samplePos,
                    make_float3(0.0, 1.0, 0.0)
                ) * AMBIENT_SCALE
            );

            // -Y
            sample.cube.color[3] = CUDABSP::rgbexp32_from_float3(
                DirectLighting::sample_at(
                    *pCudaBSP,
                    samplePos,
                    make_float3(0.0, -1.0, 0.0)
                ) * AMBIENT_SCALE
            );

            // +Z
            sample.cube.color[4] = CUDABSP::rgbexp32_from_float3(
                DirectLighting::sample_at(
                    *pCudaBSP,
                    samplePos,
                    make_float3(0.0, 0.0, 1.0)
                ) * AMBIENT_SCALE
            );

            // -Z
            sample.cube.color[5] = CUDABSP::rgbexp32_from_float3(
                DirectLighting::sample_at(
                    *pCudaBSP,
                    samplePos,
                    make_float3(0.0, 0.0, -1.0)
                ) * AMBIENT_SCALE
            );
        }
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
        for (const BSP::Face& face : bsp.get_faces()) {
            if (face.get_texinfo().flags & BSP::SURF_TRANS) {
                // Skip translucent faces.
                continue;
            }

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
        RayTracer::CUDARayTracer* pDeviceRayTracer;

        CUDA_CHECK_ERROR(
            cudaMemcpyFromSymbol(
                &pDeviceRayTracer, g_pDeviceRayTracer,
                sizeof(RayTracer::CUDARayTracer*), 0,
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK_ERROR(cudaFree(pDeviceRayTracer));

        g_pRayTracer = nullptr;
    }

    void compute_direct_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP) {
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

        KERNEL_LAUNCH_DEVICE(
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

    void antialias_direct_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP) {
        cudaEvent_t startEvent;
        cudaEvent_t stopEvent;

        CUDA_CHECK_ERROR(cudaEventCreate(&startEvent));
        CUDA_CHECK_ERROR(cudaEventCreate(&stopEvent));

        const size_t BLOCK_WIDTH = 16;
        const size_t BLOCK_HEIGHT = 16;

        size_t numFaces = bsp.get_faces().size();

        dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);

        CUDA_CHECK_ERROR(cudaEventRecord(startEvent));

        KERNEL_LAUNCH_DEVICE(
            DirectLighting::map_faces_AA,
            numFaces, blockDim,
            pCudaBSP
        );

        CUDA_CHECK_ERROR(cudaEventRecord(stopEvent));

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        float time;
        CUDA_CHECK_ERROR(cudaEventElapsedTime(&time, startEvent, stopEvent));

        std::cout << "Done! (" << time << "ms)" << std::endl;
    }

    __global__ void generate_face_info(CUDABSP::CUDABSP* pCudaBSP, CUDARAD::FaceInfo* const faces)
    {
      int face_index = threadIdx.x + threadIdx.y;
      if(face_index+1 > pCudaBSP->numFaces)
        return;

      bool primaryThread = (threadIdx.x == 0 && threadIdx.y == 0);

      if (pCudaBSP->tag != CUDABSP::TAG) {
        if (primaryThread) {
          printf("Invalid CUDABSP Tag: %x\n", pCudaBSP->tag);
        }
        return;
      }

      faces[face_index] = FaceInfo(*pCudaBSP, face_index);
    }

    __global__ void generate_patch_info(CUDABSP::CUDABSP* pCudaBSP,
                                        CUDARAD::FaceInfo* faces, size_t num_faces,
                                        CUDARAD::PatchInfo* patches)
    {
      int i = threadIdx.x + threadIdx.y * blockDim.x;
      if(i > (int) num_faces - 1)
        return;

      CUDARAD::FaceInfo face = faces[i];

      int offset0 = face.lightmapStartIndex; // TODO
      for(int j = 1; j < face.lightmapHeight; j++)
      {
        int offset1 = (j - 1)*face.lightmapHeight;
        float3 reflectivity = make_float3(
            pCudaBSP->texDatas[face.face.texInfo].reflectivity.x,
            pCudaBSP->texDatas[face.face.texInfo].reflectivity.y,
            pCudaBSP->texDatas[face.face.texInfo].reflectivity.z);
        for(int k = 1; k < face.lightmapWidth; k++)
        {
          // Calculate patch light
          int index = offset0 + offset1 + k - 1;
          patches[offset0 + offset1 + k - 1].initialLight = (
              pCudaBSP->lightSamples[offset0 + offset1 + face.lightmapHeight + k - 1] +
              pCudaBSP->lightSamples[offset0 + offset1 + k - 1] +
              pCudaBSP->lightSamples[offset0 + offset1 + face.lightmapHeight + k] +
              pCudaBSP->lightSamples[offset0 + offset1 + k]
              ) / 4;
          // Calculate vertices
          patches[index].vertices[0] = faces[i].xyz_from_st(j-1, k-1);
          patches[index].vertices[1] = faces[i].xyz_from_st(j, k-1);
          patches[index].vertices[2] = faces[i].xyz_from_st(j, k);
          patches[index].vertices[3] = faces[i].xyz_from_st(j-1, k);
          // Fill in remaining data
          patches[index].reflectivity = reflectivity;
          patches[index].receivedLight = make_float3(0, 0, 0);
          patches[index].totalLight = make_float3(0, 0, 0);
          patches[index].brightness = faces[i].totalLight;
          patches[index].face_info = faces[i];
        }
      }

    }



    void bounce_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP) {
      // TODO set up pcudabsp
      bound_lighting_fly(pCudaBSP);

    }

    size_t count_patches(CUDABSP::CUDABSP* pCudaBSP, CUDARAD::FaceInfo* faces)
    {
      size_t count = 0;
      for(int i = 0; i < pCudaBSP->numFaces; i++)
      {
        count += faces[i].lightmapSize;
      }
      return count;
    }

    void bounce_lighting_fly(CUDABSP::CUDABSP* pCudaBSP){
      std::cout << "Beginning bounce lightning" << std::endl;
      // Calculate patches
      FaceInfo* faces;
      CUDA_CHECK_ERROR(
          cudaMalloc(&faces, sizeof(CUDARAD::FaceInfo) * pCudaBSP->numFaces)
          );
      std::cout << "Generate FaceInfo for all faces" << std::endl;
      const size_t BLOCK_WIDTH = 16;
      const size_t BLOCK_HEIGHT = 16;
      dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT); // TODO figure out something better
      KERNEL_LAUNCH(
          generate_face_info,
          pCudaBSP->numFaces, blockDim,
          pCudaBSP, faces
          );

      std::cout << "Generate PatchInfo" << std::endl;
      size_t totalPatches = count_patches(pCudaBSP, faces);
      PatchInfo* patches;
      CUDA_CHECK_ERROR(
        cudaMalloc(&patches, sizeof(CUDARAD::PatchInfo) * totalPatches)
        );

      KERNEL_LAUNCH(
          generate_patch_info,
          pCudaBSP->numFaces, blockDim,
          pCudaBSP, faces, pCudaBSP->numFaces, patches
          );


      std::cout << "Begin iteration" << std::endl;

      KERNEL_LAUNCH(
          bounce_iteration,
          pCudaBSP->numFaces, blockDim,
          pCudaBSP, patches, totalPatches
          );

      std::cout << "Complete iterations" << std::endl;
      // TODO copy data somewhere
    }

    __global__ void bounce_iteration(CUDABSP::CUDABSP* pCudaBSP,
        PatchInfo* patches, int totalNumPatches)
    {
      bool primaryThread = (threadIdx.x == 0 && threadIdx.y == 0);

      if (pCudaBSP->tag != CUDABSP::TAG) {
        if (primaryThread) {
          printf("Invalid CUDABSP Tag: %x\n", pCudaBSP->tag);
        }
        return;
      }


      int index = threadIdx.x + threadIdx.y * blockIdx.x;
      PatchInfo receiver = patches[index];
      float3 reflectivity = receiver.reflectivity;
      for (int i = 0; i < MAX_ITER; i++)
      {
        float3 received_light = make_float3(0, 0, 0);
        for(int j = 0; j < totalNumPatches; j++)
        {
          if(j == index)
            continue;
          PatchInfo patch = patches[j];
          /* is it worth doing the memcmp?
          if(memcmp(receiver.face_info, patch.face_info, sizeof(ptr)) == 0)
            continue;
            */
          float ff;
          float3 center1 = center(receiver);
          float3 center2 = center(patch);
          float dist = distance(center1, center2);
          // Determine whether to diff->diff or diff->poly
          //  then calculate form factor
          if(dist < 10) // distance threshold
          {
            ff = BouncedLighting::ff_diff_poly(center2, patch.face_info.faceNorm,
                                               receiver.vertices, 4);
            // float3 diffPos, float3 diffNOrm, float3* vertices, size_t numVertices
          }
          else
          {
            ff = BouncedLighting::ff_diff_diff(center1, receiver.face_info.faceNorm,
                              center2, patch.face_info.faceNorm);
            //float3 diff1Pos, float3 diff1Norm, float3 diff2Pos, float3 diff2Norm
          }
          received_light += patches[j].brightness * ff;
        }
        __syncthreads();
        // Update array with information
        receiver.brightness = receiver.brightness + received_light; // TODO shuffle values
        __syncthreads();
        float const EPSILON = 0.0001; // convergeance threshold
        if(received_light.x < EPSILON && received_light.y < EPSILON && received_light.z < EPSILON)
        {
          break;
        }
      }

    }

    __device__ float3 center(PatchInfo patch)
    {
      float3 p = patch.vertices[0] + patch.vertices[1] + patch.vertices[2] + patch.vertices[3];
      return p/4;
    }

    static __device__ inline float distance(float3 p1, float3 p2)
    {
      // Calculate the distance between the centers of the two patches
      float diff_x = p1.x - p2.x;
      float diff_y = p1.y - p2.y;
      float diff_z = p1.z - p2.z;
      return sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);
    }

    __global__ void avg_light(CUDARAD::FaceInfo* faces, int num_faces,
                              CUDARAD::PatchInfo* patches, int num_patches)
    {
      // Get average lighting for all patches in the face

      int face_index = threadIdx.x + threadIdx.y * blockDim.x;
      if(face_index > num_faces)
        return;

      float3 sum = make_float3(0, 0, 0);
      for(int i = 0; i < num_patches; i++)
      {
        sum += patches[i].brightness; // TODO probaly not correct
      }

      faces[face_index].avgLight = sum / num_patches;
    }

    void compute_ambient_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP) {
        const size_t BLOCK_WIDTH = 32;

        size_t numLeaves = bsp.get_leaves().size();

        KERNEL_LAUNCH(
            AmbientLighting::map_leaves,
            numLeaves, BLOCK_WIDTH,
            pCudaBSP
        );

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }
}
