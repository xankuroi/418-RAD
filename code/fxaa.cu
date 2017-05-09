#include "fxaa.h"

#include "bsp.h"
#include "cudabsp.h"
#include "cudarad.h"

#include "cudautils.h"


static __device__ inline float luma_from_rgb(float3 rgb) {
    return sqrt(dot(rgb / 255.0, make_float3(0.299, 0.587, 0.114)));
}


static __device__ inline float clamp(float x, float lower, float upper) {
    return fmaxf(lower, fminf(upper, x));
}


static __device__ float3 subsample(
        float3* samples, size_t width, size_t height,
        float s, float t
        ) {

    //float3 zero = make_float3(0.0, 0.0, 0.0);

    int s0 = static_cast<int>(floorf(s));
    int t0 = static_cast<int>(floorf(t));
    int s1 = s0 + 1;
    int t1 = t0 + 1;

    if (s0 < 0) {
        s0 = 0;
    }

    if (t0 < 0) {
        t0 = 0;
    }

    if (s1 >= width) {
        s1 = width - 1;
    }

    if (t1 >= height) {
        t1 = height - 1;
    }

    float rWeight = s - floorf(s);
    float lWeight = 1.0 - rWeight;

    float dWeight = t - floorf(t);
    float uWeight = 1.0 - dWeight;

    float3 sampleUL = samples[t0 * width + s0];
    float3 sampleUR = samples[t0 * width + s1];
    float3 sampleDL = samples[t1 * width + s0];
    float3 sampleDR = samples[t1 * width + s1];

    float3 sampleU = lWeight * sampleUL + rWeight * sampleUR;
    float3 sampleD = lWeight * sampleDL + rWeight * sampleDR;

    return uWeight * sampleU + dWeight * sampleD;
}


static __device__ const float EDGE_THRESHOLD = 0.125;           // 1/8
static __device__ const float EDGE_THRESHOLD_MIN = 0.03125;     // 1/32
static __device__ const size_t MAX_ITERATIONS = 12;
static __device__ const float SUBPIXEL_QUALITY = 0.75;


/**
 * CUDA FXAA implementation based on shader code at:
 *  http://blog.simonrodriguez.fr/articles/30-07-2016_implementing_fxaa.html
 * and also:
 *  http://developer.download.nvidia.com/assets/gamedev/files/sdk/11/FXAA_WhitePaper.pdf
 */
__global__ void map_samples_fxaa(
        float3* samplesIn,
        /* output */ float3* samplesOut,
        size_t width, size_t height
        ) {

    size_t s = blockIdx.x * blockDim.x + threadIdx.x;
    size_t t = blockIdx.y * blockDim.y + threadIdx.y;

    if (s >= width || t >= height) {
        return;
    }

    float3 sample = samplesIn[t * width + s];

    float lumaCenter = luma_from_rgb(sample);

    /* Grab the lumas of our four direct neighbors. */
    float lumaUp = luma_from_rgb(
        samplesIn[((t > 0) ? (t - 1) : t) * width + s]
    );
    float lumaDown = luma_from_rgb(
        samplesIn[((t < height - 1) ? (t + 1) : t) * width + s]
    );
    float lumaLeft = luma_from_rgb(
        samplesIn[t * width + ((s > 0) ? (s - 1) : s)]
    );
    float lumaRight = luma_from_rgb(
        samplesIn[t * width + ((s < width - 1) ? (s + 1) : s)]
    );

    /* Determine the color contrast between ourselves and our neighbors. */
    float lumaMin = fminf(
        lumaCenter,
        fminf(
            fminf(lumaUp, lumaDown),
            fminf(lumaLeft, lumaRight)
        )
    );

    float lumaMax = fmaxf(
        lumaCenter,
        fmaxf(
            fmaxf(lumaUp, lumaDown),
            fmaxf(lumaLeft, lumaRight)
        )
    );

    float lumaRange = lumaMax - lumaMin;

    /*
     * Luma contrast too low (or this is a really dark spot).
     * Don't perform AA.
     */
    if (lumaRange < fmaxf(EDGE_THRESHOLD_MIN, lumaMax * EDGE_THRESHOLD)) {
        samplesOut[t * width + s] = sample;
        return;
    }
    //else {
    //    samplesOut[t * width + s] = make_float3(255.0, 0.0, 0.0);
    //    return;
    //}

    /* Grab the lumas of our remaining corner neighbors. */
    float lumaUL = luma_from_rgb(
        (t > 0 && s > 0) ?
        samplesIn[(t - 1) * width + s - 1] :
        samplesIn[t * width + s]
    );

    float lumaUR = luma_from_rgb(
        (t > 0 && s < width - 1) ?
        samplesIn[(t - 1) * width + s + 1] :
        samplesIn[t * width + s]
    );

    float lumaDL = luma_from_rgb(
        (t < height - 1 && s > 0) ?
        samplesIn[(t + 1) * width + s - 1] :
        samplesIn[t * width + s]
    );

    float lumaDR = luma_from_rgb(
        (t < height - 1 && s < width - 1) ?
        samplesIn[(t + 1) * width + s + 1] :
        samplesIn[t * width + s]
    );

    /* Combine the edge lumas. */
    float lumaUD = lumaUp + lumaDown;
    float lumaLR = lumaLeft + lumaRight;

    /* Combine the corner lumas. */
    float lumaULUR = lumaUL + lumaUR;
    float lumaDLDR = lumaDL + lumaDR;
    float lumaULDL = lumaUL + lumaDL;
    float lumaURDR = lumaUR + lumaDR;

    /* Estimate horizontal and vertical gradients. */
    float gradientHoriz = (
        fabsf(-2.0 * lumaLeft + lumaULDL)
        + fabsf(-2.0 * lumaCenter + lumaUD) * 2.0
        + fabsf(-2.0 * lumaRight + lumaURDR)
    );

    float gradientVerti = (
        fabsf(-2.0 * lumaUp + lumaULUR)
        + fabsf(-2.0 * lumaCenter + lumaLR) * 2.0
        + fabsf(-2.0 * lumaDown + lumaDLDR)
    );

    /* Are we at a horizontal or vertical edge? */
    bool isHoriz = (gradientHoriz >= gradientVerti);

    //if (isHoriz) {
    //    samplesOut[t * width + s] = make_float3(255.0, 0.0, 0.0);
    //}
    //else {
    //    samplesOut[t * width + s] = make_float3(0.0, 255.0, 0.0);
    //}
    //return;

    /* Choose two lumas in the direction opposite of the edge. */
    float luma1 = isHoriz ? lumaUp : lumaLeft;
    float luma2 = isHoriz ? lumaDown : lumaRight;

    /* Compute their gradients. */
    float gradient1 = luma1 - lumaCenter;
    float gradient2 = luma2 - lumaCenter;

    /* Choose the steeper gradient. */
    bool grad1Steeper = fabsf(gradient1) >= fabsf(gradient2);

    /* Normalize the gradients. */
    float gradientNorm = 0.25 * fmaxf(fabsf(gradient1), fabsf(gradient2));

    /* Determine directional luma average. */
    float lumaLocalAvg;

    if (grad1Steeper) {
        lumaLocalAvg = 0.5 * (luma1 + lumaCenter);
    }
    else {
        lumaLocalAvg = 0.5 * (luma2 + lumaCenter);
    }

    /* Subsample locations for each iteration. */
    float iteration1S = static_cast<float>(s);
    float iteration1T = static_cast<float>(t);
    float iteration2S = iteration1S;
    float iteration2T = iteration1T;

    /* Offset our sample locations toward the edge by half a pixel. */
    if (isHoriz) {
        iteration1T += grad1Steeper ? -0.5 : 0.5;
        iteration2T += grad1Steeper ? -0.5 : 0.5;
    }
    else {
        iteration1S += grad1Steeper ? -0.5 : 0.5;
        iteration2S += grad1Steeper ? -0.5 : 0.5;
    }
    
    /* Determine iteration offsets. */
    size_t offsetS = isHoriz ? 1 : 0;
    size_t offsetT = isHoriz ? 0 : 1;

    iteration1S -= offsetS;
    iteration1T -= offsetT;

    iteration2S += offsetS;
    iteration2T += offsetT;

    /* Iterate! */
    float lumaEnd1;
    float lumaEnd2;

    bool reached1 = false;
    bool reached2 = false;

    for (size_t i=0; i<MAX_ITERATIONS; i++) {
        /* Sample lumas in both directions along the edge. */
        if (!reached1) {
            lumaEnd1 = luma_from_rgb(
                subsample(
                    samplesIn, width, height,
                    iteration1S, iteration1T
                )
            );
            lumaEnd1 -= lumaLocalAvg;
        }

        if (!reached2) {
            lumaEnd2 = luma_from_rgb(
                subsample(
                    samplesIn, width, height,
                    iteration2S, iteration2T
                )
            );
            lumaEnd2 -= lumaLocalAvg;
        }

        /* Did we reach the end of the edge? */
        reached1 = (fabsf(lumaEnd1) >= gradientNorm);
        reached2 = (fabsf(lumaEnd2) >= gradientNorm);

        /* If we've reached the end, stop iteration. */
        if (reached1 && reached2) {
            break;
        }

        /* But if we HAVEN'T reached the end, continue... */
        if (!reached1) {
            iteration1S -= offsetS;
            iteration1T -= offsetT;
        }

        if (!reached2) {
            iteration2S += offsetS;
            iteration2T += offsetT;
        }
    }

    /* Determine how far we've traveled along the edge. */
    float dist1 = isHoriz ? (s - iteration1S) : (t - iteration1T);
    float dist2 = isHoriz ? (iteration2S - s) : (iteration2T - t);

    /* Which way is closer? */
    bool dir1Closer = dist1 < dist2;
    float closerDist = fminf(dist1, dist2);

    /* Total length of the edge. */
    float edgeLen = dist1 + dist2;

    /*
     * The pixel offset where we should subsample, in the direction of the 
     * closer edge endpoint.
     */
    float pixelOffset;

    if ((lumaCenter < lumaLocalAvg)
            != ((dir1Closer ? lumaEnd1 : lumaEnd2) < 0.0)) {
        pixelOffset = 0.0;
    }
    else {
        pixelOffset = -closerDist / edgeLen + 0.5;
    }

    //printf(
    //    "(%u, %u) %s distance: %f / %f (%f) Offset: %f\n",
    //    static_cast<unsigned int>(s), static_cast<unsigned int>(t),
    //    isHoriz ? "horizontal" : "vertical",
    //    closerDist, edgeLen, closerDist / edgeLen,
    //    pixelOffset
    //);

    /*
     * Subpixel antialiasing
     */

    /* Weighted average of all the lumas in our local 3x3 grid. */
    float lumaAvg = (
        (1.0 / 12.0) * (2.0 * (lumaUD + lumaLR) + lumaULDL + lumaURDR)
    );

    float subpixelOffset1 = clamp(
        fabsf(lumaAvg - lumaCenter) / lumaRange,
        0.0, 1.0
    );
    float subpixelOffset2 = (
        (-2.0 * subpixelOffset1 + 3.0) * subpixelOffset1 * subpixelOffset1
    );

    float subpixelOffset = (
        subpixelOffset2 * subpixelOffset2 * SUBPIXEL_QUALITY
    );

    float finalOffset = fmaxf(subpixelOffset, pixelOffset);

    if (grad1Steeper) {
        finalOffset = -finalOffset;
    }

    /* Determine the final subsample coordinates. */
    float finalS = static_cast<float>(s);
    float finalT = static_cast<float>(t);

    if (isHoriz) {
        finalT += finalOffset;
    }
    else {
        finalS += finalOffset;
    }

    /* Final subsample... */
    float3 color = subsample(samplesIn, width, height, finalS, finalT);

    //{
    //    int s0 = static_cast<int>(floorf(s));
    //    int t0 = static_cast<int>(floorf(t));
    //    int s1 = s0 + 1;
    //    int t1 = t0 + 1;

    //    if (s0 < 0) {
    //        s0 = 0;
    //    }

    //    if (t0 < 0) {
    //        t0 = 0;
    //    }

    //    if (s1 >= width) {
    //        s1 = width - 1;
    //    }

    //    if (t1 >= height) {
    //        t1 = height - 1;
    //    }

    //    float3 sampleUL = samplesIn[t0 * width + s0];
    //    float3 sampleUR = samplesIn[t0 * width + s1];
    //    float3 sampleDL = samplesIn[t1 * width + s0];
    //    float3 sampleDR = samplesIn[t1 * width + s1];

    //    printf(
    //        "(%u, %u) sampled at (%f, %f)\n"
    //        "\tUL(%f, %f, %f) UR(%f, %f, %f)\n"
    //        "\tDL(%f, %f, %f) DR(%f, %f, %f)\n"
    //        "\tyields (%f, %f, %f)\n",
    //        static_cast<unsigned int>(s), static_cast<unsigned int>(t),
    //        finalS, finalT,
    //        sampleUL.x, sampleUL.y, sampleUL.z,
    //        sampleUR.x, sampleUR.y, sampleUR.z,
    //        sampleDL.x, sampleDL.y, sampleDL.z,
    //        sampleDR.x, sampleDR.y, sampleDR.z,
    //        color.x, color.y, color.z
    //    );
    //}

    //color = isHoriz ?
    //    make_float3(color.x * 10.0, color.y, color.z) :
    //    make_float3(color.x, color.y * 10.0, color.z);

    /* ... and we're done! */
    samplesOut[t * width + s] = color;
}


__global__ void map_samples_edgeblur(
        CUDABSP::CUDABSP* pCudaBSP,
        float3* samplesIn,
        /* output */ float3* samplesOut,
        size_t width, size_t height
        ) {

    size_t s = blockIdx.x * blockDim.x + threadIdx.x;
    size_t t = blockIdx.y * blockDim.y + threadIdx.y;

    if (s >= width || t >= height) {
        return;
    }

    float3 sample = samplesIn[t * width + s];

    float lumaCenter = luma_from_rgb(sample);

    /* Grab the lumas of our four direct neighbors. */
    float lumaUp = luma_from_rgb(
        samplesIn[((t > 0) ? (t - 1) : t) * width + s]
    );
    float lumaDown = luma_from_rgb(
        samplesIn[((t < height - 1) ? (t + 1) : t) * width + s]
    );
    float lumaLeft = luma_from_rgb(
        samplesIn[t * width + ((s > 0) ? (s - 1) : s)]
    );
    float lumaRight = luma_from_rgb(
        samplesIn[t * width + ((s < width - 1) ? (s + 1) : s)]
    );

    /* Determine the color contrast between ourselves and our neighbors. */
    float lumaMin = fminf(
        lumaCenter,
        fminf(
            fminf(lumaUp, lumaDown),
            fminf(lumaLeft, lumaRight)
        )
    );

    float lumaMax = fmaxf(
        lumaCenter,
        fmaxf(
            fmaxf(lumaUp, lumaDown),
            fmaxf(lumaLeft, lumaRight)
        )
    );

    float lumaRange = lumaMax - lumaMin;

    /*
    * Luma contrast too low (or this is a really dark spot).
    * Don't perform AA.
    */
    if (lumaRange < fmaxf(EDGE_THRESHOLD_MIN, lumaMax * EDGE_THRESHOLD)) {
        samplesOut[t * width + s] = sample;
        return;
    }

}


__global__ void map_faces(CUDABSP::CUDABSP* pCudaBSP) {
    size_t faceIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (faceIndex >= pCudaBSP->numFaces) {
        return;
    }

    BSP::DFace& face = pCudaBSP->faces[faceIndex];

    size_t width = face.lightmapTextureSizeInLuxels[0] + 1;
    size_t height = face.lightmapTextureSizeInLuxels[1] + 1;

    size_t startIndex = face.lightOffset / sizeof(BSP::RGBExp32);

    float3* lightSamples = pCudaBSP->lightSamples + startIndex;
    float3* results = new float3[width * height];

    const size_t BLOCK_WIDTH = 16;
    const size_t BLOCK_HEIGHT = 16;

    dim3 gridDim(
        div_ceil(width, BLOCK_WIDTH),
        div_ceil(height, BLOCK_HEIGHT)
    );

    dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);

    KERNEL_LAUNCH_DEVICE(
        map_samples_fxaa,
        gridDim, blockDim,
        lightSamples, results,
        width, height
    );

    CUDA_CHECK_ERROR_DEVICE(cudaDeviceSynchronize());

    /* Transfer the AA'd results back into the light sample buffer. */
    memcpy(lightSamples, results, sizeof(float3) * width * height);

    delete[] results;
}


namespace CUDAFXAA {
    void antialias_lightsamples(BSP::BSP& bsp, CUDABSP::CUDABSP* pCudaBSP) {
        CUDABSP::CUDABSP cudaBSP;

        CUDA_CHECK_ERROR(
            cudaMemcpy(
                &cudaBSP, pCudaBSP, sizeof(CUDABSP::CUDABSP),
                cudaMemcpyDeviceToHost
            )
        );

        for (const BSP::Face& face : bsp.get_faces()) {
            size_t width = face.get_lightmap_width();
            size_t height = face.get_lightmap_height();
            size_t numSamples = width * height;

            size_t startIndex
                = face.get_data().lightOffset / sizeof(BSP::RGBExp32);

            float3* samples = cudaBSP.lightSamples + startIndex;
            float3* results;

            CUDA_CHECK_ERROR(
                cudaMalloc(&results, sizeof(float3) * numSamples)
            );

            const size_t BLOCK_WIDTH = 16;
            const size_t BLOCK_HEIGHT = 16;

            dim3 gridDim(
                div_ceil(width, BLOCK_WIDTH),
                div_ceil(height, BLOCK_HEIGHT)
            );

            dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);

            KERNEL_LAUNCH(
                map_samples_fxaa,
                gridDim, blockDim,
                samples, results,
                width, height
            );

            CUDA_CHECK_ERROR(cudaDeviceSynchronize());

            CUDA_CHECK_ERROR(
                cudaMemcpy(
                    samples, results, sizeof(float3) * numSamples,
                    cudaMemcpyDeviceToDevice
                )
            );

            CUDA_CHECK_ERROR(cudaFree(results));
        }

        //size_t numFaces;

        //CUDA_CHECK_ERROR(
        //    cudaMemcpy(
        //        &numFaces, &pCudaBSP->numFaces, sizeof(size_t),
        //        cudaMemcpyDeviceToHost
        //    )
        //);

        //const size_t BLOCK_WIDTH = 32;
        //size_t numBlocks = div_ceil(numFaces, BLOCK_WIDTH);

        //KERNEL_LAUNCH(
        //    map_faces,
        //    numBlocks, BLOCK_WIDTH,
        //    pCudaBSP
        //);

        //CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }
}
