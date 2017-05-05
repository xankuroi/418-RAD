#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include <iostream>
#include <thread>

#include "bsp.h"

#include "cudarad.h"

#include "cudabsp.h"
#include "cudamatrix.h"

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


static __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}


static __device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}


static __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}


static __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}


static __device__ inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}


static __device__ inline float3 operator*(const float3& v, float c) {
    return make_float3(v.x * c, v.y * c, v.z * c);
}


static __device__ inline float3 operator*(float c, const float3& v) {
    return v * c;
}


static __device__ inline float3 operator/(const float3& v, float c) {
    return v * (1.0 / c);
}


static __device__ inline float dist(const float3& a, const float3& b) {
    float3 diff = b - a;
    return sqrt(dot(diff, diff));
}


static __device__ inline float len(const float3& v) {
    return dist(make_float3(0.0, 0.0, 0.0), v);
}


static __device__ inline float3 normalized(const float3& v) {
    return v / len(v);
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


static __device__ void make_points(
        CUDABSP::CUDABSP& cudaBSP,
        /* output */ float3* points,
        BSP::DFace& face
        ) {

    for (size_t i=0; i<face.numEdges; i++) {
        int32_t surfEdge = cudaBSP.surfEdges[face.firstEdge + i];

        bool firstToSecond = (surfEdge >= 0);

        if (!firstToSecond) {
            surfEdge *= -1;
        }

        BSP::DEdge& edge = cudaBSP.edges[surfEdge];

        if (firstToSecond) {
            points[i] = cudaBSP.vertices[edge.vertex1];
        }
        else {
            points[i] = cudaBSP.vertices[edge.vertex2];
        }
    }
}


/* Implements the M-T ray-triangle intersection algorithm. */
static __device__ bool intersects(
        const float3& vertex1, const float3& vertex2, const float3& vertex3,
        const float3& startPos, const float3& endPos
        ) {

    const float EPSILON = 1e-6;

    float3 diff = endPos - startPos;
    float dist = len(diff);
    float3 dir = diff / dist;

    float3 edge1 = vertex2 - vertex1;
    float3 edge2 = vertex3 - vertex1;

    float3 pVec = cross(dir, edge2);

    float det = dot(edge1, pVec);

    if (det < EPSILON) {
        return false;
    }

    float3 tVec = startPos - vertex1;

    float u = dot(tVec, pVec);
    if (u < 0.0 || u > det) {
        return false;
    }

    float3 qVec = cross(tVec, edge1);

    float v = dot(dir, qVec);

    if (v < 0.0 || u + v > det) {
        return false;
    }

    float t = dot(edge2, qVec) / det;

    return (0.0 < t && t < dist);
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
    __global__ void map_faces_LOS(
            CUDABSP::CUDABSP* pCudaBSP,
            /* output */ bool* pLightBlocked,
            size_t faceIndex,
            float3 samplePos,
            float3 lightPos
            ) {

        size_t otherFaceIndex = blockIdx.x * blockDim.x + threadIdx.x;

        if (otherFaceIndex == faceIndex
                || otherFaceIndex >= pCudaBSP->numFaces) {
            return;
        }

        BSP::DFace& otherFace = pCudaBSP->faces[otherFaceIndex];

        float3 vertex1;
        float3 vertex2;
        float3 vertex3;

        for (size_t i=0; i<otherFace.numEdges; i++) {
            int32_t surfEdge = pCudaBSP->surfEdges[otherFace.firstEdge + i];

            bool firstToSecond = (surfEdge >= 0);

            if (!firstToSecond) {
                surfEdge *= -1;
            }

            BSP::DEdge& edge = pCudaBSP->edges[surfEdge];

            if (i == 0) {
                if (firstToSecond) {
                    vertex1 = pCudaBSP->vertices[edge.vertex1];
                }
                else {
                    vertex1 = pCudaBSP->vertices[edge.vertex2];
                }
            }
            else if (i == 1) {
                if (firstToSecond) {
                    vertex3 = pCudaBSP->vertices[edge.vertex1];
                }
                else {
                    vertex3 = pCudaBSP->vertices[edge.vertex2];
                }
            }
            else {
                vertex2 = vertex3;

                if (firstToSecond) {
                    vertex3 = pCudaBSP->vertices[edge.vertex1];
                }
                else {
                    vertex3 = pCudaBSP->vertices[edge.vertex2];
                }

                bool lightBlocked = intersects(
                    vertex3, vertex2, vertex1,
                    lightPos, samplePos
                );

                if (lightBlocked) {
                    *pLightBlocked = true;
                    return;
                }
            }
        }
    }

    __device__ float3 sample_at(
            CUDABSP::CUDABSP& cudaBSP,
            FaceInfo& faceInfo,
            size_t s, size_t t
            ) {
        
        const float EPSILON = 1e-6;

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

            float3 diff = lightPos - samplePos;

            /*
             * This light is on the wrong side of the current face.
             * There's no way it could possibly light this sample.
             */
            if (dot(diff, faceInfo.faceNorm) < 0.0) {
                continue;
            }

            bool* pLightBlocked = new bool;

            CUDA_CHECK_ERROR_DEVICE(cudaPeekAtLastError());

            *pLightBlocked = false;

            const size_t BLOCK_WIDTH = 8;

            size_t numBlocks = div_ceil(cudaBSP.numFaces, BLOCK_WIDTH);

            KERNEL_LAUNCH_DEVICE(
                map_faces_LOS,
                numBlocks, BLOCK_WIDTH,
                &cudaBSP, pLightBlocked, faceInfo.faceIndex,
                samplePos, lightPos
            );

            CUDA_CHECK_ERROR_DEVICE(cudaDeviceSynchronize());

            bool lightBlocked = *pLightBlocked;

            delete pLightBlocked;

            //bool lightBlocked = false;

            //int x = 0;

            //for (size_t otherFaceIndex=0;
            //        otherFaceIndex<cudaBSP.numFaces;
            //        otherFaceIndex++) {

            //    if (otherFaceIndex == faceInfo.faceIndex) {
            //        continue;
            //    }

            //    BSP::DFace& otherFace = cudaBSP.faces[otherFaceIndex];
            //    BSP::DPlane& otherPlane = cudaBSP.planes[otherFace.planeNum];

            //    float3 vertex1;
            //    float3 vertex2;
            //    float3 vertex3;

            //    size_t startEdge = otherFace.firstEdge;
            //    size_t endEdge = startEdge + otherFace.numEdges;

            //    for (size_t i=startEdge; i<endEdge; i++) {
            //        int32_t surfEdge = cudaBSP.surfEdges[i];

            //        bool firstToSecond = (surfEdge >= 0);

            //        if (!firstToSecond) {
            //            surfEdge *= -1;
            //        }

            //        BSP::DEdge& edge = cudaBSP.edges[surfEdge];

            //        if (i == startEdge) {
            //            if (firstToSecond) {
            //                vertex1 = cudaBSP.vertices[edge.vertex1];
            //            }
            //            else {
            //                vertex1 = cudaBSP.vertices[edge.vertex2];
            //            }
            //        }
            //        else if (i == startEdge + 1) {
            //            if (firstToSecond) {
            //                vertex3 = cudaBSP.vertices[edge.vertex1];
            //            }
            //            else {
            //                vertex3 = cudaBSP.vertices[edge.vertex2];
            //            }
            //        }
            //        else {
            //            vertex2 = vertex3;

            //            if (firstToSecond) {
            //                vertex3 = cudaBSP.vertices[edge.vertex1];
            //            }
            //            else {
            //                vertex3 = cudaBSP.vertices[edge.vertex2];
            //            }

            //            lightBlocked = intersects(
            //                vertex3, vertex2, vertex1,
            //                lightPos, samplePos
            //            );

            //            //printf(
            //            //    "Light blocked: %d\n",
            //            //    static_cast<int>(lightBlocked)
            //            //);

            //            if (lightBlocked) {
            //                //*pLightBlocked = true;
            //                break;
            //            }
            //        }
            //    }

            //    if (lightBlocked) {
            //        break;
            //    }
            //}

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
            volatile size_t* pFacesCompleted
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

        //for (size_t i=0; i<lightmapSize; i++) {
        //    size_t s = i % lightmapWidth;
        //    size_t t = i / lightmapWidth;

        //    float3 color = sample_at(*pCudaBSP, faces, faceIndex, s, t);

        //    pCudaBSP->lightSamples[lightmapStartIndex + i]
        //        = lightsample_from_rgb(color);

        //    totalLight += color;
        //}

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

            atomicAdd(
                reinterpret_cast<unsigned int*>(
                    const_cast<size_t*>(pFacesCompleted)
                ), 1
            );
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
            pCudaBSP, pDeviceFacesCompleted
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
