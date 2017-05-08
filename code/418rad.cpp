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

static std::random_device g_random;

static std::unique_ptr<BSP::BSP> g_pBSP;

/* Arbitrary "small" value. */
static const double EPSILON = 1e-3;


struct RGB {
    uint32_t r;
    uint32_t g;
    uint32_t b;
};


static inline uint8_t random_color(void) {
    return static_cast<uint8_t>(g_random() % 255);
}


static BSP::RGBExp32 lightsample_from_rgb(const RGB& color) {
    uint32_t r = color.r;
    uint32_t g = color.g;
    uint32_t b = color.b;
    int8_t exp = 0;

    // BSP format stores colors as biased exponential values.
    // To get actual RGB format, evaluate (color * 2^exp).
    while (r > 255 || g > 255 || b > 255) {
        r >>= 1;
        g >>= 1;
        b >>= 1;
        exp++;
    }

    assert(r <= 255);
    assert(g <= 255);
    assert(b <= 255);

    return BSP::RGBExp32 {
        static_cast<uint8_t>(r),
        static_cast<uint8_t>(g),
        static_cast<uint8_t>(b),
        exp,
    };
}


static RGB sample_at(const BSP::Face& face, float s, float t) {

    const BSP::Vec3<float>& faceNorm = face.get_planedata().normal;

    /*
     * Determine the 3D position of this light sample based on its
     * luxel coordinates on this face.
     */
    BSP::Vec3<float> samplePos = face.xyz_from_lightmap_st(s, t);

    /*
     * Shift the sample point just a *tiny* bit away from the face, to
     * prevent this face (and its direct neighbors) from intersecting
     * with it when we do our line-of-sight tests.
     */
    //samplePos.x += static_cast<float>(EPSILON * faceNorm.x);
    //samplePos.y += static_cast<float>(EPSILON * faceNorm.y);
    //samplePos.z += static_cast<float>(EPSILON * faceNorm.z);

    RGB result {0, 0, 0};

    /*
     * Go through all the lights in the map and add their color
     * contributions to this sample.
     */
    for (const BSP::Light& light : g_pBSP->get_lights()) {
        const BSP::Vec3<float>& lightPos = light.get_coords();

        BSP::Vec3<float> diff {
            lightPos.x - samplePos.x,
            lightPos.y - samplePos.y,
            lightPos.z - samplePos.z,
        };

        float diffDotNormal = (
            diff.x * faceNorm.x +
            diff.y * faceNorm.y +
            diff.z * faceNorm.z
        );

        /*
         * This light is on the wrong side of the current face.
         * There's no way it could possibly light this sample.
         */
        if (diffDotNormal < 0.0) {
            continue;
        }

        bool lightBlocked = false;

        /*
         * Iterate through all other faces to determine if any of them
         * block line-of-sight between the light and this sample.
         *
         * Stupidly inefficient!!!
         */
        for (BSP::Face& otherFace : g_pBSP->get_faces()) {
            if (otherFace.id == face.id) {
                // Don't test intersection with myself...
                continue;
            }

            // std::cout << "    Check Face "
                // << otherFace.id << "..." << std::endl;

            std::vector<BSP::Vec3<float>> points;
            for (const BSP::Edge& edge : otherFace.get_edges()) {
                points.push_back(edge.vertex1);
            }

            assert(points.size() >= 3);

            std::vector<BSP::Vec3<float>>::reverse_iterator pVertex
                = points.rbegin();

            BSP::Vec3<float> vertex1 = *pVertex++;

            BSP::Vec3<float> vertex2;
            BSP::Vec3<float> vertex3 = *pVertex++;

            // Line segment from the light to the sample position.
            gmtl::LineSeg<float> diffLineSeg(
                gmtl::Point<float, 3>(
                    lightPos.x, lightPos.y, lightPos.z
                ),
                gmtl::Point<float, 3>(
                    samplePos.x, samplePos.y, samplePos.z
                )
            );

            const BSP::DPlane& otherFacePlaneData = otherFace.get_planedata();
            const BSP::Vec3<float>& otherFaceNorm = otherFacePlaneData.normal;

            // The plane containing the face we're checking against.
            gmtl::Plane<float> otherFacePlane(
                gmtl::Vec<float, 3>(
                    otherFaceNorm.x,
                    otherFaceNorm.y,
                    otherFaceNorm.z
                ),
                gmtl::Point<float, 3>(
                    vertex1.x,
                    vertex1.y,
                    vertex1.z
                )
            );

            float t;

            BSP::Vec3<float> otherDiff {
                vertex1.x - samplePos.x,
                vertex1.y - samplePos.y,
                vertex1.z - samplePos.z,
            };

            float otherNormalDotOtherDiff = (
                otherFaceNorm.x * otherDiff.x +
                otherFaceNorm.y * otherDiff.y +
                otherFaceNorm.z * otherDiff.z
            );

            // If the sample point itself directly intersects with this 
            // face's plane, then ignore it.
            if (otherNormalDotOtherDiff < EPSILON) {
                continue;
            }

            // If the line segment between the sample point and the 
            // light does not intersect this face's plane, there's no 
            // way this face could be blocking the light.
            // Move on and check the next face.
            if (!gmtl::intersect(otherFacePlane, diffLineSeg, t)) {
                continue;
            }

            // If we got past the plane check, that means this face 
            // MIGHT be blocking the light. Do a finer-grained check 
            // on the individual triangles of this face to see if this 
            // face blocks the light.

            /*
             * Iterate through all the triangles in this face to
             * determine if this face actually blocks the light.
             */
            do {
                vertex2 = vertex3;
                vertex3 = *pVertex;

                // std::cout << "       Check triangle (<"
                    // << vertex1.x << ", "
                    // << vertex1.y << ", "
                    // << vertex1.z << ">, <"
                    // << vertex2.x << ", "
                    // << vertex2.y << ", "
                    // << vertex2.z << ">, <"
                    // << vertex3.x << ", "
                    // << vertex3.y << ", "
                    // << vertex3.z << ">)..."
                    // << std::endl;
                    
                gmtl::Tri<float> tri(
                    gmtl::Point<float, 3>(
                        vertex1.x, vertex1.y, vertex1.z
                    ),
                    gmtl::Point<float, 3>(
                        vertex2.x, vertex2.y, vertex2.z
                    ),
                    gmtl::Point<float, 3>(
                        vertex3.x, vertex3.y, vertex3.z
                    )
                );

                float u, v, t;

                if (gmtl::intersect(tri, diffLineSeg, u, v, t)) {
                    // It turns out that this face actually DOES block 
                    // the light...
                    lightBlocked = true;
                    break;
                }

                pVertex++;

            } while (pVertex != points.rend());

            if (lightBlocked) {
                break;
            }
        }

        if (lightBlocked) {
            // This light can't be seen from the position of the sample.
            // Ignore it.
            continue;
        }

        /* I CAN SEE THE LIGHT */

        double dist = sqrt(
            diff.x * diff.x +
            diff.y * diff.y +
            diff.z * diff.z
        );

        double attenuation = light.attenuate(dist);

        result.r += static_cast<uint32_t>(light.r / attenuation);
        result.g += static_cast<uint32_t>(light.g / attenuation);
        result.b += static_cast<uint32_t>(light.b / attenuation);
    }

    return result;
}


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
    
    try {
        g_pBSP = std::unique_ptr<BSP::BSP>(new BSP::BSP(filename));
    }
    catch (BSP::InvalidBSP e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    g_pBSP->build_worldlights();

    print_cudainfo();

    CUDA_CHECK_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));

    std::cout << "Copy BSP to device memory..." << std::endl;
    CUDABSP::CUDABSP* pCudaBSP = CUDABSP::make_cudabsp(*g_pBSP);
    
    std::cout << "Initialize radiosity subsystem..." << std::endl;
    CUDARAD::init(*g_pBSP);

    std::cout << "Start RAD!" << std::endl;
    
    std::cout << "Compute direct lighting..." << std::endl;
    std::vector<BSP::RGBExp32> directLighting;
    CUDARAD::compute_direct_lighting(*g_pBSP, pCudaBSP, directLighting);

    std::cout << "Compute light bounces..." << std::endl;
    CUDARAD::bounce_lighting(*g_pBSP, pCudaBSP);

    ///* Process the lighting for each face in the map. */
    //for (BSP::Face& face : g_pBSP->get_faces()) {
    //    std::cout << "Processing Face " << face.id << "..." << std::endl;
    //    
    //    RGB sum {0, 0, 0};

    //    size_t lightmapWidth = face.get_lightmap_width();
    //    size_t lightmapHeight = face.get_lightmap_height();
    //    size_t numSamples = lightmapWidth * lightmapHeight;

    //    BSP::FaceLightSampleProxy samples = face.get_lightsamples();
    //    
    //    /* Process each lighting sample on this face. */
    //    for (size_t i=0; i<numSamples; i++) {
    //        // `s` is the luxel coordinate in the face's local x-direction.
    //        // `t` is the luxel coordinate in the face's local y-direction.
    //        float s = static_cast<float>(i % lightmapWidth);
    //        float t = static_cast<float>(i / lightmapWidth);

    //        /*
    //         * Supersample 100-ish points "around" this luxel, to counteract 
    //         * lightmap aliasing.
    //         *
    //         * Easy to implement, but SUPER SLOW AND INEFFICIENT!!!
    //         * TBH, I would like to change this to FXAA at some point.
    //         */
    //        const int NUM_SUPERSAMPLES = 100;
    //        const float SUPERSAMPLE_RADIUS = 1.0;

    //        RGB color {0, 0, 0};

    //        int sampleCount = 0;
    //        for (int i=0; i<NUM_SUPERSAMPLES; i++) {
    //            /*
    //             * This is random radial supersampling.
    //             * It sucks. Don't use it.
    //             */
    //            //float sOffset = std::numeric_limits<float>::infinity();
    //            //float tOffset = std::numeric_limits<float>::infinity();

    //            //while (dist(sOffset, tOffset, 0.0, 0.0) >= 1.0) {
    //            //    const float min = static_cast<float>(std::random_device::min());
    //            //    const float max = static_cast<float>(std::random_device::max());
    //            //    const float diff = max - min;

    //            //    sOffset = static_cast<float>((g_random() - min) / diff);
    //            //    tOffset = static_cast<float>((g_random() - min) / diff);
    //            //}

    //            //assert(0.0 <= sOffset && sOffset <= 1.0);
    //            //assert(0.0 <= tOffset && tOffset <= 1.0);

    //            //sOffset *= SUPERSAMPLE_RADIUS * 2.0;
    //            //tOffset *= SUPERSAMPLE_RADIUS * 2.0;

    //            //sOffset -= SUPERSAMPLE_RADIUS;
    //            //tOffset -= SUPERSAMPLE_RADIUS;

    //            /*
    //             * This is box supersampling. It works much better.
    //             * Also, it's a lot simpler.
    //             */
    //            const int SUPERSAMPLE_WIDTH
    //                = static_cast<int>(sqrt(NUM_SUPERSAMPLES));

    //            const float SUPERSAMPLE_DIFF
    //                = SUPERSAMPLE_RADIUS * 2.0f / SUPERSAMPLE_WIDTH;
    //            
    //            int supersampleS = i % SUPERSAMPLE_WIDTH;
    //            int supersampleT = i / SUPERSAMPLE_WIDTH;

    //            float sOffset
    //                = -SUPERSAMPLE_RADIUS + SUPERSAMPLE_DIFF * supersampleS;

    //            float tOffset
    //                = -SUPERSAMPLE_RADIUS + SUPERSAMPLE_DIFF * supersampleT;

    //            //// Don't supersample anywhere that isn't within the bounds of 
    //            //// this face.
    //            //if (s == 0 && sOffset < 0.0
    //            //        || s == lightmapWidth - 1 && sOffset > 0.0
    //            //        || t == 0 && tOffset < 0.0
    //            //        || t == lightmapWidth - 1 && tOffset > 0.0
    //            //        ) {
    //            //    continue;
    //            //}

    //            assert(
    //                -SUPERSAMPLE_RADIUS <= sOffset
    //                    && sOffset <= SUPERSAMPLE_RADIUS
    //            );
    //            assert(
    //                -SUPERSAMPLE_RADIUS <= tOffset
    //                    && tOffset <= SUPERSAMPLE_RADIUS
    //            );

    //            RGB offsetColor = sample_at(face, s + sOffset, t + tOffset);

    //            color.r += offsetColor.r;
    //            color.g += offsetColor.g;
    //            color.b += offsetColor.b;

    //            sampleCount++;
    //        }

    //        // The final color is the average of the supersamples.
    //        color.r /= sampleCount;
    //        color.g /= sampleCount;
    //        color.b /= sampleCount;

    //        samples[i] = lightsample_from_rgb(color);

    //        sum.r += color.r;
    //        sum.g += color.g;
    //        sum.b += color.b;
    //    }

    //    RGB average {
    //        static_cast<uint32_t>(sum.r / numSamples),
    //        static_cast<uint32_t>(sum.g / numSamples),
    //        static_cast<uint32_t>(sum.b / numSamples),
    //    };

    //    // The BSP format requires us to also store the average lighting 
    //    // value of each face, for some reason.
    //    face.set_average_lighting(lightsample_from_rgb(average));
    //    
    //    // Don't ask. I have no idea.
    //    face.set_styles(std::vector<uint8_t> {0, 0xFF, 0xFF, 0xFF});
    //}
    
    CUDABSP::update_bsp(*g_pBSP, pCudaBSP);

    CUDABSP::destroy_cudabsp(pCudaBSP);

    /*
     * Mark the BSP as non-fullbright.
     *
     * This tells the engine that there is actually lighting information 
     * embedded in the map.
     */
    g_pBSP->set_fullbright(false);
    
    g_pBSP->write("out.bsp");
    
    std::cout << "Wrote to file \"out.bsp\"." << std::endl;

    /* Tear down the radiosity subsystem. */
    CUDARAD::cleanup();
    
    return 0;
}
