#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include <iostream>

#include "bsp.h"

#include "cudarad.h"

#include "cudabsp.h"
#include "cudamatrix.h"

#include "cudautils.h"


static __device__ inline float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}


static __device__ inline float attenuate(
        BSP::DWorldLight& light, float dist
        ) {
    float c = light.constantAtten;
    float l = light.linearAtten;
    float q = light.quadraticAtten;

    return c + l * dist + q * dist * dist;
}


//static __device__ void mult(
//        CUDAMatrix::CUDAMatrix<double, 3, 1>& result,
//        CUDAMatrix::CUDAMatrix<double, 3, 3>& A,
//        CUDAMatrix::CUDAMatrix<double, 3, 1>& B
//        ) {
//
//    for (int resultRow=0; resultRow<3; resultRow++) {
//        double sum = 0.0;
//
//        for (int i=0; i<3; i++) {
//            sum += A[resultRow][i] * B[i][0];
//        }
//
//        result[resultRow][0] = sum;
//    }
//}


static __device__ float3 xyz_from_st(
        CUDABSP::CUDABSP cudaBSP, size_t faceIndex,
        size_t s, size_t t
        ) {

    BSP::DFace& face = cudaBSP.faces[faceIndex];
    BSP::DPlane& plane = cudaBSP.planes[face.planeNum];
    BSP::TexInfo& texInfo = cudaBSP.texInfos[face.texInfo];
    CUDAMatrix::CUDAMatrix<double, 3, 3> Ainv = cudaBSP.xyzMatrices[faceIndex];

    float sOffset = texInfo.lightmapVecs[0][3];
    float tOffset = texInfo.lightmapVecs[1][3];

    float sMin = face.lightmapTextureMinsInLuxels[0];
    float tMin = face.lightmapTextureMinsInLuxels[1];

    CUDAMatrix::CUDAMatrix<double, 3, 1> B;

    B[0][0] = s - sOffset + sMin;
    B[1][0] = t - tOffset + tMin;
    B[2][0] = plane.dist;

    //mult(result, Ainv, B);

    //printf(
    //    "Face %u Ainv:\n"
    //    "\t%f %f %f\n"
    //    "\t%f %f %f\n"
    //    "\t%f %f %f\n",
    //    static_cast<unsigned int>(faceIndex),
    //    Ainv[0][0], Ainv[0][1], Ainv[0][2],
    //    Ainv[1][0], Ainv[1][1], Ainv[1][2],
    //    Ainv[2][0], Ainv[2][1], Ainv[2][2]
    //);

    CUDAMatrix::CUDAMatrix<double, 3, 1> result = Ainv * B;

    //CUDAMatrix::CUDAMatrix<double, 3, 1> result;
    //result[0][0]
    //    = Ainv[0][0] * B[0][0] + Ainv[0][1] * B[1][0] + Ainv[0][2] * B[2][0];
    //result[1][0]
    //    = Ainv[1][0] * B[0][0] + Ainv[1][1] * B[1][0] + Ainv[1][2] * B[2][0];
    //result[2][0]
    //    = Ainv[2][0] * B[0][0] + Ainv[2][1] * B[1][0] + Ainv[2][2] * B[2][0];

    float x = static_cast<float>(result[0][0]);
    float y = static_cast<float>(result[1][0]);
    float z = static_cast<float>(result[2][0]);

    return make_float3(x, y, z);
}


static __device__ void make_points(
        CUDABSP::CUDABSP& cudaBSP,
        BSP::DFace& face,
        float3*& points
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
    //__global__ void map_faces_LOS(
    //        CUDABSP::CUDABSP* pCudaBSP,
    //        size_t faceIndex
    //        ) {

    //    size_t otherFaceIndex = blockIdx.x * blockDim.x + threadIdx.x;

    //    BSP::DFace& face = pCudaBSP->faces[faceIndex];

    //    float3* points;
    //    cudaMalloc(&points, sizeof(float3) * face.numEdges);

    //    make_points(*pCudaBSP, face, points);



    //    cudaFree(points);
    //}

    __global__ void map_worldlights(
            CUDABSP::CUDABSP* pCudaBSP,
            /* output */ float3* lightContributions,
            size_t faceIndex,
            float3 samplePos
            ) {

        size_t lightIndex = blockIdx.x * blockDim.x + threadIdx.x;

        if (lightIndex >= pCudaBSP->numWorldLights) {
            return;
        }

        //printf(
        //    "map lights for Face %u at (%f, %f, %f)\n",
        //    static_cast<unsigned int>(faceIndex),
        //    samplePos.x,
        //    samplePos.y,
        //    samplePos.z
        //);

        BSP::DFace& face = pCudaBSP->faces[faceIndex];
        BSP::DPlane& plane = pCudaBSP->planes[face.planeNum];

        float3 faceNorm = make_float3(
            plane.normal.x,
            plane.normal.y,
            plane.normal.z
        );

        BSP::DWorldLight& light = pCudaBSP->worldLights[lightIndex];

        float3 lightPos = make_float3(
            light.origin.x,
            light.origin.y,
            light.origin.z
        );

        float3 diff = make_float3(
            lightPos.x - samplePos.x,
            lightPos.y - samplePos.y,
            lightPos.z - samplePos.z
        );

        float diffDotNormal = dot(diff, faceNorm);

        /*
         * This light is on the wrong side of the current face.
         * There's no way it could possibly light this sample.
         */
        if (diffDotNormal < 0.0) {
            return;
        }

        //const size_t BLOCK_WIDTH = 1024;

        //size_t numFaces = pCudaBSP->numFaces;
        //size_t numBlocks = div_ceil(numFaces, BLOCK_WIDTH);

        //KERNEL_LAUNCH(
        //    map_faces_LOS,
        //    numBlocks, numFaces,
        //    pCudaBSP, faceIndex
        //);

        //cudaDeviceSynchronize();

        //for (int i=0; i<pCudaBSP->numFaces; i++) {
        //    BSP::DFace& otherFace = pCudaBSP->faces[i];

        //    if (i == faceIndex) {
        //        continue;
        //    }

        //    float3* points;
        //    cudaMalloc(&points, sizeof(float3) *otherFace.numEdges);

        //    make_points(*pCudaBSP, otherFace, points);

        //    float3* pVertex = points + otherFace.numEdges - 1;

        //    float3 vertex1 = *pVertex--;
        //    float3 vertex2;
        //    float3 vertex3 = *pVertex--;



        //}

        float dist = sqrt(
            diff.x * diff.x +
            diff.y * diff.y +
            diff.z * diff.z
        );

        float attenuation = attenuate(light, dist);

        float3 color;

        color.x = light.intensity.x / attenuation;
        color.y = light.intensity.y / attenuation;
        color.z = light.intensity.z / attenuation;

        lightContributions[lightIndex] = color;
    }

    __global__ void map_supersamples(CUDABSP::CUDABSP* pCudaBSP) {

    }

    __global__ void map_luxels(
            CUDABSP::CUDABSP* pCudaBSP,
            /* output */ float3* pTotalLight,
            size_t faceIndex
            ) {

        size_t s = blockIdx.x * blockDim.x + threadIdx.x;
        size_t t = blockIdx.y * blockDim.y + threadIdx.y;

        BSP::DFace& face = pCudaBSP->faces[faceIndex];

        size_t lightmapWidth = face.lightmapTextureSizeInLuxels[0] + 1;
        size_t lightmapHeight = face.lightmapTextureSizeInLuxels[1] + 1;

        if (s >= lightmapWidth || t >= lightmapHeight) {
            //printf(
            //    "Early return for Face %u at (%u, %u)\n",
            //    static_cast<unsigned int>(faceIndex),
            //    static_cast<unsigned int>(s),
            //    static_cast<unsigned int>(t)
            //);
            return;
        }

        //printf(
        //    "map luxels for Face %u at (%u, %u)\n",
        //    static_cast<unsigned int>(faceIndex),
        //    static_cast<unsigned int>(s),
        //    static_cast<unsigned int>(t)
        //);

        BSP::DPlane& plane = pCudaBSP->planes[face.planeNum];

        float3 faceNorm = make_float3(
            plane.normal.x,
            plane.normal.y,
            plane.normal.z
        );

        float3 samplePos = xyz_from_st(*pCudaBSP, faceIndex, s, t);

        //printf(
        //    "(%u) Sampling light for Face %u (%u, %u) at <%f, %f, %f> "
        //    "(normal: <%f, %f, %f>, plane: %u)\n",
        //    static_cast<unsigned int>(t * lightmapWidth + s),
        //    static_cast<unsigned int>(faceIndex),
        //    static_cast<unsigned int>(s),
        //    static_cast<unsigned int>(t),
        //    samplePos.x,
        //    samplePos.y,
        //    samplePos.z,
        //    pCudaBSP->planes[face.planeNum].normal.x,
        //    pCudaBSP->planes[face.planeNum].normal.y,
        //    pCudaBSP->planes[face.planeNum].normal.z,
        //    static_cast<unsigned int>(face.planeNum)
        //);

        //float3* lightContributions = new float3[pCudaBSP->numWorldLights];

        //CUDA_CHECK_ERROR_DEVICE(cudaPeekAtLastError());

        //const size_t BLOCK_WIDTH = 1;
        //size_t numLights = pCudaBSP->numWorldLights;
        //size_t numBlocks = div_ceil(numLights, BLOCK_WIDTH);

        //printf(
        //    "Number of world lights: %u\n",
        //    static_cast<unsigned int>(pCudaBSP->numWorldLights)
        //);

        //KERNEL_LAUNCH(
        //    map_worldlights,
        //    numBlocks, numLights,
        //    pCudaBSP, lightContributions, faceIndex, samplePos
        //);

        //CUDA_CHECK_ERROR_DEVICE(cudaPeekAtLastError());

        //cudaDeviceSynchronize();

        //CUDA_CHECK_ERROR_DEVICE(cudaPeekAtLastError());

        float3 result = make_float3(0.0, 0.0, 0.0);

        for (size_t i=0; i<pCudaBSP->numWorldLights; i++) {
            BSP::DWorldLight& light = pCudaBSP->worldLights[i];

            //printf(
            //    "Light %u -- rgb: (%f, %f, %f), c: %f, l: %f, q: %f\n",
            //    static_cast<unsigned int>(i),
            //    light.intensity.x,
            //    light.intensity.y,
            //    light.intensity.z,
            //    light.constantAtten,
            //    light.linearAtten,
            //    light.quadraticAtten
            //);

            float3 diff = make_float3(
                light.origin.x - samplePos.x,
                light.origin.y - samplePos.y,
                light.origin.z - samplePos.z
            );

            float diffDotNormal = dot(diff, faceNorm);

            /*
             * This light is on the wrong side of the current face.
             * There's no way it could possibly light this sample.
             */
            if (diffDotNormal < 0.0) {
                //printf(
                //    "Face %u skipped light %u due to normal <%f, %f, %f>!\n",
                //    static_cast<unsigned int>(faceIndex),
                //    static_cast<unsigned int>(i),
                //    faceNorm.x,
                //    faceNorm.y,
                //    faceNorm.z
                //);
                return;
            }

            float dist = sqrt(
                diff.x * diff.x +
                diff.y * diff.y +
                diff.z * diff.z
            );

            float attenuation = attenuate(light, dist);

            //printf("dist: %f; attenuation: %f\n", dist, attenuation);

            result.x += light.intensity.x * 255.0 / attenuation;
            result.y += light.intensity.y * 255.0 / attenuation;
            result.z += light.intensity.z * 255.0 / attenuation;
        }

        size_t lightmapStartIndex = face.lightOffset / sizeof(BSP::RGBExp32);
        size_t lightSampleIndex = t * lightmapWidth + s;

        //printf(
        //    "Start Index for Face %u: %u\n",
        //    static_cast<unsigned int>(faceIndex),
        //    static_cast<unsigned int>(lightmapStartIndex)
        //);

        pCudaBSP->lightSamples[lightmapStartIndex + lightSampleIndex]
            = lightsample_from_rgb(result);

        // Don't worry. This will compile just fine.
        atomicAdd(&pTotalLight->x, result.x);
        atomicAdd(&pTotalLight->y, result.y);
        atomicAdd(&pTotalLight->z, result.z);

        //delete[] lightContributions;
    }

    __global__ void map_faces(CUDABSP::CUDABSP* pCudaBSP) {
        if (pCudaBSP->tag != CUDABSP::TAG) {
            printf("Invalid CUDABSP Tag: %x\n", pCudaBSP->tag);
            return;
        }

        // Map to faces.
        size_t faceIndex = blockIdx.x * blockDim.x + threadIdx.x;

        if (faceIndex >= pCudaBSP->numFaces) {
            return;
        }

        printf("Processing Face %u...\n", faceIndex);

        BSP::DFace& face = pCudaBSP->faces[faceIndex];

        size_t lightmapWidth = face.lightmapTextureSizeInLuxels[0] + 1;
        size_t lightmapHeight = face.lightmapTextureSizeInLuxels[1] + 1;
        size_t lightmapSize = lightmapWidth * lightmapHeight;

        //printf(
        //    "Face %u: \n"
        //    //"\t first 8 bytes: %x %x %x %x %x %x %x %x\n"
        //    "\t addr: %p\n"
        //    "\t firstEdge addr: %p\n"
        //    "\t planeNum: %u\n"
        //    "\t side: %u\n"
        //    "\t onNode: %u\n"
        //    "\t firstEdge: %d\n"
        //    "\t numEdges: %d\n"
        //    "\t texInfo: %d\n"
        //    "\t dispInfo: %d\n"
        //    "\t surfaceFogVolumeID: %d\n"
        //    "\t styles: %x, %x, %x, %x\n"
        //    "\t lightOffset: %d\n"
        //    "\t area: %f\n"
        //    "\t mins: (%d, %d)\n"
        //    "\t size: %d x %d\n"
        //    "\t origFace: %d\n"
        //    "\t numPrims: %u\n"
        //    "\t firstPrimID: %u\n"
        //    "\t smoothingGroups: %x\n",
        //    static_cast<unsigned int>(faceIndex),
        //    //static_cast<int>(c[0]),
        //    //static_cast<int>(c[1]),
        //    //static_cast<int>(c[2]),
        //    //static_cast<int>(c[3]),
        //    //static_cast<int>(c[4]),
        //    //static_cast<int>(c[5]),
        //    //static_cast<int>(c[6]),
        //    //static_cast<int>(c[7]),
        //    pFace,
        //    &pFace->firstEdge,
        //    static_cast<unsigned int>(pFace->planeNum),
        //    static_cast<unsigned int>(pFace->side),
        //    static_cast<unsigned int>(pFace->onNode),
        //    static_cast<int>(pFace->firstEdge),
        //    static_cast<int>(pFace->numEdges),
        //    static_cast<int>(pFace->texInfo),
        //    static_cast<int>(pFace->dispInfo),
        //    static_cast<int>(pFace->surfaceFogVolumeID),
        //    static_cast<int>(pFace->styles[0]),
        //    static_cast<int>(pFace->styles[1]),
        //    static_cast<int>(pFace->styles[2]),
        //    static_cast<int>(pFace->styles[3]),
        //    static_cast<int>(pFace->lightOffset),
        //    pFace->area,
        //    static_cast<int>(pFace->lightmapTextureMinsInLuxels[0]),
        //    static_cast<int>(pFace->lightmapTextureMinsInLuxels[1]),
        //    static_cast<int>(pFace->lightmapTextureSizeInLuxels[0]),
        //    static_cast<int>(pFace->lightmapTextureSizeInLuxels[1]),
        //    static_cast<int>(pFace->origFace),
        //    static_cast<unsigned int>(pFace->numPrims),
        //    static_cast<unsigned int>(pFace->firstPrimID),
        //    static_cast<int>(pFace->smoothingGroups)
        //);

        float3* pTotalLight = new float3;

        CUDA_CHECK_ERROR_DEVICE(cudaPeekAtLastError());

        const size_t BLOCK_WIDTH = 4;
        const size_t BLOCK_HEIGHT = 4;

        dim3 gridDim(
            div_ceil(lightmapWidth, BLOCK_WIDTH),
            div_ceil(lightmapHeight, BLOCK_HEIGHT)
        );
        dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);

        //printf(
        //    "Face %u/%u:\n"
        //    "\t Lightmap Dimensions: %u x %u\n"
        //    "\t gridDim: %u x %u\n",
        //    static_cast<unsigned int>(faceIndex),
        //    static_cast<unsigned int>(pCudaBSP->numFaces),
        //    static_cast<unsigned int>(lightmapWidth),
        //    static_cast<unsigned int>(lightmapHeight),
        //    static_cast<unsigned int>(gridDim.x),
        //    static_cast<unsigned int>(gridDim.y)
        //);

        //printf(
        //    "Face %u launching kernel...\n",
        //    static_cast<unsigned int>(faceIndex)
        //);

        KERNEL_LAUNCH_DEVICE(
            map_luxels,
            gridDim, blockDim,
            pCudaBSP, pTotalLight, faceIndex
        );

        cudaDeviceSynchronize();

        CUDA_CHECK_ERROR_DEVICE(cudaPeekAtLastError());

        float3 avgLight = *pTotalLight;

        avgLight.x /= lightmapSize;
        avgLight.y /= lightmapSize;
        avgLight.z /= lightmapSize;

        size_t lightmapStartIndex = face.lightOffset / sizeof(BSP::RGBExp32);

        //printf(
        //    "Lightmap offset for face %u: %u\n",
        //    static_cast<unsigned int>(faceIndex),
        //    static_cast<unsigned int>(lightmapStartIndex)
        //);

        pCudaBSP->lightSamples[lightmapStartIndex - 1]
            = lightsample_from_rgb(avgLight);

        // Still have no idea how this works. But if we don't do this,
        // EVERYTHING becomes a disaster...
        face.styles[0] = 0x00;
        face.styles[1] = 0xFF;
        face.styles[2] = 0xFF;
        face.styles[3] = 0xFF;

        delete pTotalLight;
    }
}


namespace CUDARAD {
    void compute_direct_lighting(
            BSP::BSP& bsp,
            CUDABSP::CUDABSP* pCudaBSP,
            std::vector<BSP::RGBExp32>& lightSamples
            ) {

        const size_t BLOCK_WIDTH = 8;

        size_t numFaces = bsp.get_faces().size();
        unsigned int numBlocks = div_ceil(
            numFaces,
            BLOCK_WIDTH
        );

        std::cout << "Launching "
            << numBlocks * BLOCK_WIDTH << " threads ("
            << numFaces << " faces)..."
            << std::endl;

        KERNEL_LAUNCH(
            DirectLighting::map_faces,
            numBlocks, BLOCK_WIDTH,
            pCudaBSP
        );

        cudaDeviceSynchronize();

        CUDA_CHECK_ERROR(cudaPeekAtLastError());
    }

    void bounce_lighting(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP) {
        std::cerr << "Radiosity light bounces not implemented!" << std::endl;
    }
}
