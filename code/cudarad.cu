#include "cub/device/device_scan.cuh"

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

    __device__ float3 sample_at(
            CUDABSP::CUDABSP& cudaBSP,
            float3 samplePos,
            float3 sampleNormal=make_float3(0.0, 0.0, 0.0)
            ) {

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
}


namespace AA {
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

    static __device__ const float MIN_AA_GRADIENT = 0.0625;    // 1/16

    __global__ void map_face_samples(
            CUDABSP::CUDABSP* pCudaBSP,
            /* output */ int* facesForSamples,
            /* output */ int2* coordsForSamples
            ) {

        int faceIndex = blockIdx.x * blockDim.x + threadIdx.x;

        if (faceIndex >= pCudaBSP->numFaces) {
            return;
        }

        CUDARAD::FaceInfo faceInfo(*pCudaBSP, faceIndex);

        for (size_t i=0; i<faceInfo.lightmapSize; i++) {
            size_t sampleIndex = faceInfo.lightmapStartIndex + i;

            facesForSamples[sampleIndex] = faceIndex;
            coordsForSamples[sampleIndex] = make_int2(
                i % faceInfo.lightmapWidth,
                i / faceInfo.lightmapWidth
            );
        }
    }

    __global__ void map_select_targets(
            CUDABSP::CUDABSP* pCudaBSP,
            int* facesForSamples, int2* coordsForSamples,
            /* output */ int* targets
            ) {

        size_t sampleIndex = blockIdx.x * blockDim.x + threadIdx.x;

        if (sampleIndex >= pCudaBSP->numLightSamples) {
            return;
        }

        int faceIndex = facesForSamples[sampleIndex];
        int2 coords = coordsForSamples[sampleIndex];

        if (faceIndex == -1 || coords.x == -1 || coords.y == -1) {
            return;
        }

        int s = coords.x;
        int t = coords.y;

        CUDARAD::FaceInfo faceInfo(*pCudaBSP, faceIndex);

        float3* samples = &pCudaBSP->lightSamples[faceInfo.lightmapStartIndex];

        size_t width = faceInfo.lightmapWidth;
        size_t height = faceInfo.lightmapHeight;

        float3 sample = samples[t * width + s];

        float gradient = 0.0;

        for (int offsetT=-1; offsetT<=1; offsetT++) {
            int neighborT = t + offsetT;

            if (!(0 <= neighborT && neighborT < height)) {
                continue;
            }

            for (int offsetS=-1; offsetS<=1; offsetS++) {
                if (offsetS == 0 && offsetT == 0) {
                    continue;
                }

                int neighborS = s + offsetS;

                if (!(0 <= neighborS && neighborS < width)) {
                    continue;
                }

                size_t neighborIndex = neighborT * width + neighborS;

                float3 neighbor = samples[neighborIndex];

                gradient = fmaxf(
                    gradient,
                    fabsf(intensity(neighbor) - intensity(sample))
                );
            }
        }

        targets[sampleIndex] = static_cast<int>(gradient >= MIN_AA_GRADIENT);
    }

    __global__ void gather_target_coords(
            CUDABSP::CUDABSP* pCudaBSP,
            /* output */ int2* finalCoords,
            /* output */ int* finalFacesForCoords,
            int2* coordsForSamples, int* facesForSamples,
            int* targetsScanned, int* targets
            ) {

        size_t index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= pCudaBSP->numLightSamples) {
            return;
        }

        if (targets[index]) {
            size_t destIndex = targetsScanned[index] - 1;

            finalCoords[destIndex] = coordsForSamples[index];
            finalFacesForCoords[destIndex] = facesForSamples[index];
        }
    }

    __global__ void antialias_coords(
            CUDABSP::CUDABSP* pCudaBSP,
            int2* coords, int* facesForCoords,
            size_t numCoords
            ) {

        size_t index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= numCoords) {
            return;
        }

        int2 samplePos = coords[index];
        size_t faceIndex = facesForCoords[index];

        CUDARAD::FaceInfo faceInfo(*pCudaBSP, faceIndex);

        int s = samplePos.x;
        int t = samplePos.y;

        /* Perform supersampling at this point. */
        const size_t SUPERSAMPLE_WIDTH = 4;

        float sStep = 2.0 / static_cast<float>(SUPERSAMPLE_WIDTH);
        float tStep = 2.0 / static_cast<float>(SUPERSAMPLE_WIDTH);

        float3 color = make_float3(0.0, 0.0, 0.0);

        for (size_t ssi=0; ssi<SUPERSAMPLE_WIDTH; ssi++) {
            float tOffset = tStep * ssi - 1.0;

            for (size_t ssj=0; ssj<SUPERSAMPLE_WIDTH; ssj++) {
                float sOffset = sStep * ssj - 1.0;

                color += DirectLighting::sample_at(
                    *pCudaBSP, faceInfo,
                    s + sOffset, t + tOffset
                );
            }
        }

        color /= SUPERSAMPLE_WIDTH * SUPERSAMPLE_WIDTH;

        size_t startIndex = faceInfo.lightmapStartIndex;
        size_t sampleIndex = t * faceInfo.lightmapWidth + s;

        pCudaBSP->lightSamples[startIndex + sampleIndex] = color;
    }

    __global__ void map_faces_AA(CUDABSP::CUDABSP* pCudaBSP) {
        bool primaryThread = (threadIdx.x == 0 && threadIdx.y == 0);

        __shared__ CUDARAD::FaceInfo faceInfo;

        __shared__ size_t lightmapStart;
        __shared__ size_t width;
        __shared__ size_t height;

        //__shared__ float3* results;

        if (primaryThread) {
            // Map block numbers to faces.
            faceInfo = CUDARAD::FaceInfo(*pCudaBSP, blockIdx.x);

            lightmapStart = faceInfo.lightmapStartIndex;
            width = faceInfo.lightmapWidth;
            height = faceInfo.lightmapHeight;

            //results = new float3[width * height];
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
                    //results[sampleIndex] = sampleColor;
                    continue;
                }

                ///*
                // * Box blur!
                // * Really stupid and potentially ugly, but really fast!
                // */

                //float3 color = make_float3(0.0, 0.0, 0.0);

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

                ///* Take the average of the box blur samples. */
                //color /= 9.0;

                /* Perform supersampling at this point. */
                const size_t SUPERSAMPLE_WIDTH = 4;

                float sStep = 2.0 / static_cast<float>(SUPERSAMPLE_WIDTH);
                float tStep = 2.0 / static_cast<float>(SUPERSAMPLE_WIDTH);

                float3 color = make_float3(0.0, 0.0, 0.0);

                for (size_t ssi=0; ssi<SUPERSAMPLE_WIDTH; ssi++) {
                    float tOffset = tStep * ssi - 1.0;

                    for (size_t ssj=0; ssj<SUPERSAMPLE_WIDTH; ssj++) {
                        float sOffset = sStep * ssj - 1.0;

                        color += DirectLighting::sample_at(
                            *pCudaBSP, faceInfo,
                            s + sOffset, t + tOffset
                        );
                    }
                }

                color /= SUPERSAMPLE_WIDTH * SUPERSAMPLE_WIDTH;

                //results[sampleIndex] = color;
                pCudaBSP->lightSamples[lightmapStart + sampleIndex] = color;
            }
        }

        //__syncthreads();

        //if (primaryThread) {
        //    /* Move the results back to the light samples array. */
        //    memcpy(
        //        pCudaBSP->lightSamples + faceInfo.lightmapStartIndex,
        //        results,
        //        sizeof(float3) * faceInfo.lightmapSize
        //    );

        //    delete[] results;
        //}
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
    static __device__ float ff_diff_poly( // Assumes perfect squares
            float3 diffPos, float3 diffNorm,
            CUDARAD::PatchInfo patch, size_t numVertices
            ) {

        float result = 0.0;
        float3 v1, v2, v3, v4, vx1, vx2, vx3, vx4;
        v1 = patch.face_info.xyz_from_st(patch.s,   patch.t)   - diffPos;
        v2 = patch.face_info.xyz_from_st(patch.s,   patch.t+1) - diffPos;
        v3 = patch.face_info.xyz_from_st(patch.s+1, patch.t+1) - diffPos;
        v4 = patch.face_info.xyz_from_st(patch.s+1, patch.t)   - diffPos;

        vx1 = cross(v1, v2);
        vx2 = cross(v2, v3);
        vx3 = cross(v3, v4);
        vx4 = cross(v4, v1);

        float xlen1 = len(vx1);
        float xlen2 = len(vx2);
        float xlen3 = len(vx3);
        float xlen4 = len(vx4);

        vx1 /= xlen1;
        vx2 /= xlen2;
        vx3 /= xlen3;
        vx4 /= xlen4;

        float vlen1 = len(v1);
        float vlen2 = len(v2);
        float vlen3 = len(v3);
        float vlen4 = len(v4);

        float t1 = asinf(xlen1 / (vlen1 * vlen2));
        float t2 = asinf(xlen2 / (vlen2 * vlen3));
        float t3 = asinf(xlen3 / (vlen3 * vlen4));
        float t4 = asinf(xlen4 / (vlen4 * vlen1));

        result += dot(diffNorm, vx1) * t1;
        result += dot(diffNorm, vx2) * t2;
        result += dot(diffNorm, vx3) * t3;
        result += dot(diffNorm, vx4) * t4;



/*
        for (size_t i=0; i<4; i++) {
            float3 vertex1 = vertices[i] - diffPos;
            float3 vertex2 = vertices[(i + 1) % numVertices] - diffPos;
            float3 vertexCross = cross(vertex1, vertex2);
            float crossLen = len(vertexCross);

            vertexCross /= crossLen;

            float v1Len = len(vertex1);
            float v2Len = len(vertex2);

            float theta =  asinf(crossLen / (v1Len * v2Len));

            result += dot(diffNorm, vertexCross) * theta;
        }
        */

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

        CUDA_CHECK_ERROR(cudaEventRecord(startEvent));

        size_t numSamples = bsp.get_lightsamples().size();

        int* facesForSamples;

        CUDA_CHECK_ERROR(
            cudaMalloc(&facesForSamples, sizeof(int) * numSamples)
        );
        CUDA_CHECK_ERROR(
            cudaMemset(facesForSamples, -1, sizeof(int) * numSamples)
        );

        int2* coordsForSamples;

        CUDA_CHECK_ERROR(
            cudaMalloc(&coordsForSamples, sizeof(int2) * numSamples)
        );
        CUDA_CHECK_ERROR(
            cudaMemset(coordsForSamples, -1, sizeof(int2) * numSamples)
        );

        int* targets;
        CUDA_CHECK_ERROR(cudaMalloc(&targets, sizeof(int) * numSamples));
        CUDA_CHECK_ERROR(cudaMemset(targets, 0, sizeof(int) * numSamples));

        size_t blockWidth = 1024;
        size_t numBlocks = div_ceil(bsp.get_faces().size(), blockWidth);

        KERNEL_LAUNCH(
            AA::map_face_samples,
            numBlocks, blockWidth,
            pCudaBSP,
            facesForSamples, coordsForSamples
        );

        numBlocks = div_ceil(numSamples, blockWidth);

        KERNEL_LAUNCH(
            AA::map_select_targets,
            numBlocks, blockWidth,
            pCudaBSP,
            facesForSamples, coordsForSamples,
            targets
        );

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        int* targetsScanned;
        CUDA_CHECK_ERROR(
            cudaMalloc(&targetsScanned, sizeof(int) * numSamples)
        );

        void* dTempStorage = nullptr;
        size_t tempStorageSize = 0;
        cub::DeviceScan::InclusiveSum(
            dTempStorage, tempStorageSize,
            targets, targetsScanned,
            numSamples
        );

        CUDA_CHECK_ERROR(cudaMalloc(&dTempStorage, tempStorageSize));

        cub::DeviceScan::InclusiveSum(
            dTempStorage, tempStorageSize,
            targets, targetsScanned,
            numSamples
        );

        CUDA_CHECK_ERROR(cudaFree(dTempStorage));

        int numTargets;
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                &numTargets, &targetsScanned[numSamples - 1], sizeof(int),
                cudaMemcpyDeviceToHost
            )
        );

        if (numTargets <= 0) {
            // TODO: Free all cudaMalloc()'d memory!
            return;
        }

        std::cout << "numTargets: " << numTargets << std::endl;

        int2* finalCoords;
        CUDA_CHECK_ERROR(
            cudaMalloc(&finalCoords, sizeof(int2) * numTargets)
        );

        int* finalFacesForCoords;
        CUDA_CHECK_ERROR(
            cudaMalloc(&finalFacesForCoords, sizeof(int) * numTargets)
        );

        blockWidth = 128;
        numBlocks = div_ceil(numSamples, blockWidth);

        KERNEL_LAUNCH(
            AA::gather_target_coords,
            numBlocks, blockWidth,
            pCudaBSP,
            finalCoords, finalFacesForCoords,
            coordsForSamples, facesForSamples,
            targetsScanned, targets
        );

        numBlocks = div_ceil(numTargets, blockWidth);

        std::cout << "numBlocks: " << numBlocks << std::endl;

        KERNEL_LAUNCH(
            AA::antialias_coords,
            numBlocks, blockWidth,
            pCudaBSP,
            finalCoords, finalFacesForCoords,
            static_cast<size_t>(numTargets)
        );

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        CUDA_CHECK_ERROR(cudaFree(finalFacesForCoords));
        CUDA_CHECK_ERROR(cudaFree(finalCoords));
        CUDA_CHECK_ERROR(cudaFree(targetsScanned));
        CUDA_CHECK_ERROR(cudaFree(targets));
        CUDA_CHECK_ERROR(cudaFree(coordsForSamples));
        CUDA_CHECK_ERROR(cudaFree(facesForSamples));

        //const size_t BLOCK_WIDTH = 16;
        //const size_t BLOCK_HEIGHT = 16;

        //size_t numFaces = bsp.get_faces().size();

        //dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);

        //KERNEL_LAUNCH(
        //    DirectLighting::map_faces_AA,
        //    numFaces, blockDim,
        //    pCudaBSP
        //);

        CUDA_CHECK_ERROR(cudaEventRecord(stopEvent));

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        float time;
        CUDA_CHECK_ERROR(cudaEventElapsedTime(&time, startEvent, stopEvent));

        std::cout << "Done! (" << time << "ms)" << std::endl;
    }

    __global__ void generate_face_info(CUDABSP::CUDABSP* pCudaBSP, CUDARAD::FaceInfo* const faces)
    {
      //int face_index = threadIdx.x + threadIdx.y * blockIdx.x;
      int face_index = blockIdx.x * blockDim.x + threadIdx.x;
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

    __global__ void map_patch_to_face(CUDABSP::CUDABSP* pCudaBSP,
        CUDARAD::FaceInfo* faces, size_t num_faces)
    {
      if(!(threadIdx.x == 0 || threadIdx.y == 0))
        return;
      size_t count = 0;
      for(int i = 0; i < num_faces; i++)
      {
        faces[i].patchStartIndex = count;
        count += faces[i].lightmapSize;
      }
    }

    __global__ void gatherLight(CUDABSP::CUDABSP* pCudaBSP,
        CUDARAD::FaceInfo* faces, size_t num_faces, PatchInfo* patches)
    {
      int face_index = blockIdx.x * blockDim.x + threadIdx.x;
      if(face_index >= num_faces)
        return;
      FaceInfo face = faces[face_index];
      int start = face.lightmapStartIndex;
      int patch_index = face.patchStartIndex;
      int width = face.lightmapWidth;
      int height = face.lightmapHeight;
      float3 totalLight = make_float3(0, 0, 0);
      printf("width %d | height %d | start %d | patch_index %d", width, height, start, patch_index);
      for(int j = 0; j < height; j++)
      {
        int offset1 = j*width;
        for(int k = 0; k < width; k++)
        {
//          printf("patch: %d || lightsample: %d\n", (int)(patch_index + offset1 + k),
  //            (int)(start + offset1 + k));
 //         printf("\t j: %d, k: %d\n", j, k);
          float3 light = patches[patch_index + offset1 + k].totalLight;
          totalLight += light;
          pCudaBSP->lightSamples[start + offset1 + k] = make_float3(light.x, light.y, light.z);
          /*
      if(face_index == num_faces-1)
            printf("%d R: %f, G: %f, B: %f\n", patch_index+offset1+k, (float)light.x, (float)light.y, (float)light.z);
          pCudaBSP->lightSamples[start + offset1 + width + k] = (
               patches[patch_index + offset1 + k].totalLight +
               patches[patch_index + offset1 + width + k + 1].totalLight +
               patches[patch_index + offset1 + width + k].totalLight +
               patches[patch_index + offset1 + k + 1].totalLight
               )/4; */
        }
        //totalLight += patches[patch_index + offset1 + width-1].totalLight;
      }
      // Extrapolate
      /*
      if(height > 2)
      {
        for(int i = 0; i < width; i++)
        {
          float3 diff = pCudaBSP->lightSamples[start + width + i] -
                          pCudaBSP->lightSamples[start + 2*width + i];
          pCudaBSP->lightSamples[start + i] = pCudaBSP->lightSamples[start + width + i] + diff;
        }
      }
      else
      {
        for(int i = 0; i < width; i++)
        {
          pCudaBSP->lightSamples[start + i] = pCudaBSP->lightSamples[start + width + i];
        }
      }*/
      face.avgLight = totalLight / face.lightmapSize;
      face.totalLight = totalLight;
      pCudaBSP->lightSamples[start-1] = face.avgLight;
    }

    __global__ void generate_patch_info(CUDABSP::CUDABSP* pCudaBSP,
        CUDARAD::FaceInfo* faces, size_t num_faces,
        CUDARAD::PatchInfo* patches, size_t num_patches)
    {
      int face_index = blockIdx.x * blockDim.x + threadIdx.x;
      if(face_index >= num_faces)
        return;
      CUDARAD::FaceInfo face = faces[face_index];
      int patch_index = face.patchStartIndex;

      int start = face.lightmapStartIndex;
      float3 reflectivity = make_float3(
          pCudaBSP->texDatas[face.texInfo.texData].reflectivity.x,
          pCudaBSP->texDatas[face.texInfo.texData].reflectivity.y,
          pCudaBSP->texDatas[face.texInfo.texData].reflectivity.z);
      for(int j = 0; j < face.lightmapHeight; j++)
      {
        int offset1 = j*face.lightmapWidth;
        for(int k = 0; k < face.lightmapWidth; k++)
        {
          // Calculate patch light
          PatchInfo patch = patches[patch_index + offset1 + k];
          patch.reflectivity = reflectivity;
          patch.face_info = face;
          patch.totalLight = pCudaBSP->lightSamples[start + offset1 + k];//]make_float3(pCudaBSP->lightSamples[start + offset1 + k].x, pCudaBSP->lightSamples[start + offset1 + k].y, pCudaBSP->lightSamples[start + offset1 + k].z);
          patch.brightness = patch.totalLight;
          patches[patch_index + offset1 + k] = patch;
          /*
          if(face_index == num_faces - 1)
          {
            printf("%d R: %f, G: %f, B: %f\n", patch_index+offset1+k, patch.totalLight.x, patch.totalLight.y, patch.totalLight.z);
          }
          (
              pCudaBSP->lightSamples[start + offset1 + face.lightmapWidth + k] +
              pCudaBSP->lightSamples[start + offset1 + k] +
              pCudaBSP->lightSamples[start + offset1 + face.lightmapWidth + k + 1] +
              pCudaBSP->lightSamples[start + offset1 + k + 1]
              ) / 4;
          // Calculate vertices
             CUDA_CHECK_ERROR_DEVICE(cudaMalloc(&patch.vertices, sizeof(float3)*4));
             patch.vertices[0] = face.xyz_from_st((j + 1), (k + 1));
             printf("DOOT\n");
             patch.vertices[1] = face.xyz_from_st(j, (k + 1));
             printf("DOOTDOOT\n");
             patch.vertices[2] = face.xyz_from_st(j, k);
             printf("DOOT\n");
             patch.vertices[3] = face.xyz_from_st((j + 1), k);
           */
          // Fill in remaining data
          //           printf("%d\n", patch_index);
          patch.s = j;
          patch.t = k;
        }
      }
    }


    void bounce_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP) {
      // TODO timing
      //pCudaBSP = make_cudabsp(bsp);
      using Clock = std::chrono::high_resolution_clock;

      auto start = Clock::now();
      bounce_lighting_fly(bsp, pCudaBSP);
      auto end = Clock::now();
      std::chrono::milliseconds ms
        = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start
            );

      std::cout << "Done! (" << ms.count() << "ms)" << std::endl;
      //update_bsp(bsp, pCudabsp);
    }

//    size_t count_patches(CUDABSP::CUDABSP* pCudaBSP, CUDARAD::FaceInfo* faces)
    size_t count_patches(BSP::BSP& bsp)
    {
      size_t count = 0;
      std::vector<BSP::Face> faces = bsp.get_faces();
      for(int i = 0; i < faces.size(); i++)
      {
        count += faces[i].get_lightmap_size();
      }
      return count;
    }

    void bounce_lighting_fly(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP){
      size_t numFaces = bsp.get_faces().size();
      std::cout << "Beginning bounce lightning" << std::endl;
      // Calculate patches
      FaceInfo* faces;
      CUDA_CHECK_ERROR(
          cudaMalloc(&faces, sizeof(CUDARAD::FaceInfo) * numFaces)
          );
      std::cout << "Generate FaceInfo for all faces" << std::endl;
      const size_t BLOCK_WIDTH = 16;
      const size_t BLOCK_HEIGHT = 16;
      dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);
      KERNEL_LAUNCH(
          generate_face_info,
          div_ceil(numFaces, BLOCK_WIDTH), BLOCK_WIDTH,
          pCudaBSP, faces
          );

      CUDA_CHECK_ERROR(cudaDeviceSynchronize());
      size_t totalPatches = count_patches(bsp);
      KERNEL_LAUNCH(
          map_patch_to_face, 1, 1,
          pCudaBSP, faces, numFaces
          );

      CUDA_CHECK_ERROR(cudaDeviceSynchronize());
      std::cout << "Generate PatchInfo; " << totalPatches << " patches" << std::endl;
      PatchInfo* patches;
      CUDA_CHECK_ERROR(
        cudaMalloc(&patches, sizeof(CUDARAD::PatchInfo) * totalPatches)
        );


      KERNEL_LAUNCH(
          generate_patch_info,
          div_ceil(numFaces, BLOCK_WIDTH), BLOCK_WIDTH,
          pCudaBSP, faces, numFaces, patches, totalPatches
          );
      //generate_patch_info(pCudaBSP, faces, numFaces, patches);

      CUDA_CHECK_ERROR(cudaDeviceSynchronize());

      std::cout << "Begin iteration" << std::endl;

      const size_t BLOCK_WIDTH2 = 64;
      /*
      KERNEL_LAUNCH(
          bounce_iteration,
          div_ceil(totalPatches, BLOCK_WIDTH2), BLOCK_WIDTH2,
          pCudaBSP, patches, totalPatches
          );
          */

      CUDA_CHECK_ERROR(cudaDeviceSynchronize());
      std::cout << "Complete iterations" << std::endl;
      KERNEL_LAUNCH(
          gatherLight,
          div_ceil(numFaces, BLOCK_WIDTH), BLOCK_WIDTH,
          pCudaBSP, faces, numFaces, patches
          );
      CUDA_CHECK_ERROR(cudaDeviceSynchronize());

      std::cout << "Gathered light" << std::endl;
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


      int index = blockIdx.x * blockDim.x + threadIdx.x;
      //printf("%d, %d, %d\n", received_light.x, received_light.y, received_light.z);
      if(index >= totalNumPatches)
        return;
      PatchInfo receiver = patches[index];
      float3 reflectivity = receiver.reflectivity;
      for (int i = 0; i < MAX_ITER; i++)
      {
        float3 received_light = make_float3(0, 0, 0);
        for(int j = 0; j < totalNumPatches; j++)
        {
          if(j == index)
            continue;
          PatchInfo emitter = patches[j];
          /* is it worth doing the memcmp?
          if(memcmp(receiver.face_info, patch.face_info, sizeof(ptr)) == 0)
            continue;
            */
          float ff;
          float3 center1 = center(receiver);
          float3 center2 = center(emitter);
          float dist = distance(center1, center2);
          // Determine whether to diff->diff or diff->poly
          //  then calculate form factor
          if(dist < 10) // distance threshold
          {
            ff = BouncedLighting::ff_diff_poly(center2, receiver.face_info.faceNorm,
                                               receiver, 4);
          }
          else
          {
            ff = BouncedLighting::ff_diff_diff(center1, receiver.face_info.faceNorm,
                              center2, emitter.face_info.faceNorm);
          }
          received_light += patches[j].brightness * ff;
        }
        __syncthreads();
        // Update array with information
        receiver.totalLight += element_wise_mult(reflectivity, received_light);
        receiver.brightness = make_float3(0, 0, 0);//received_light;
        __syncthreads();
        const float EPSILON = 1e-3;
        if(received_light.x < EPSILON && received_light.y < EPSILON && received_light.z < EPSILON)
        {
          break;
        }
      }

    }

    __device__ void patch_to_face(CUDABSP::CUDABSP* pCudaBSP,
        CUDARAD::FaceInfo* faces, size_t num_faces,
        CUDARAD::PatchInfo* patches, size_t num_patches)
    {
      if(!(threadIdx.x == 0 && threadIdx.y == 0))
        return;
      int patch_index = 0;
      for(int i = 0; i < num_faces; i++)
      {
        CUDARAD::FaceInfo face = faces[i];
        for(int j = 0; j < face.lightmapHeight - 1; j++)
        {
          for(int k = 0; k < face.lightmapWidth - 1; k++)
          {
          }
        }
      }
    }


    static __device__ inline float3 element_wise_mult(float3 f1, float3 f2)
    {
      return make_float3(f1.x * f2.x, f1.y * f2.y, f1.z * f2.z);
    }

    __device__ float3 center(PatchInfo patch)
    {
      float3 p1 = patch.face_info.xyz_from_st(patch.s, patch.t);
      float3 p2 = patch.face_info.xyz_from_st(patch.s+1, patch.t);
      float3 p3 = patch.face_info.xyz_from_st(patch.s+1, patch.t+1);
      float3 p4 = patch.face_info.xyz_from_st(patch.s, patch.t+1);
      float3 p = p1 + p2 + p3 + p4;
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

    /*
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
    */

    void compute_ambient_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP) {
        const size_t BLOCK_WIDTH = 32;

        size_t numLeaves = bsp.get_leaves().size();
        std::cout << "Begin computations" << std::endl;

        KERNEL_LAUNCH(
            AmbientLighting::map_leaves,
            numLeaves, BLOCK_WIDTH,
            pCudaBSP
        );
        std::cout << "Complete mapping" << std::endl;

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }
}
