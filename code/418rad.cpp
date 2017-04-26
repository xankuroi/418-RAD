#include <iostream>
#include <fstream>

#include <string>
#include <vector>
#include <random>

#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <cmath>

#include <gmtl/LineSeg.h>
#include <gmtl/Plane.h>
#include <gmtl/Intersection.h>

#include "bsp.h"

static std::random_device g_random;

static BSP::BSP g_bsp;

/* Arbitrary "small" value. */
static const double EPSILON = 1e-3;


static inline uint8_t random_color(void) {
    return static_cast<uint8_t>(g_random() % 255);
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Invalid arguments." << std::endl;
        return 1;
    }
    
    const std::string filename(argv[1]);
    std::ifstream f(filename, std::ios::binary);
    
    try {
        g_bsp = BSP::BSP(filename);
    }
    catch (BSP::InvalidBSP e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Start RAD!" << std::endl;
    
    /* Process the lighting for each face in the map. */
    for (BSP::Face& face : g_bsp.get_faces()) {
        std::cout << "Processing Face " << face.id << "..." << std::endl;
        
        uint32_t sumR = 0;
        uint32_t sumG = 0;
        uint32_t sumB = 0;
        
        size_t width = face.get_lightmap_width();
        size_t height = face.get_lightmap_height();
        size_t numSamples = width * height;
        const BSP::Vec3<float>& faceNorm = face.get_planedata().normal;
        
        std::vector<BSP::LightSample>& samples = face.get_lightsamples();
        
        /* Process each lighting sample on this face. */
        for (size_t i=0; i<numSamples; i++) {
            // `s` is the luxel coordinate in the face's local x-direction.
            // `t` is the luxel coordinate in the face's local y-direction.
            float s = static_cast<float>(i % width);
            float t = static_cast<float>(i / width);
            
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
            samplePos.x += EPSILON * faceNorm.x;
            samplePos.y += EPSILON * faceNorm.y;
            samplePos.z += EPSILON * faceNorm.z;
            
            BSP::LightSample& lightSample = samples.at(i);
            
            uint32_t r = 0;
            uint32_t g = 0;
            uint32_t b = 0;
            int32_t exp = 0;
            
            /*
             * Go through all the lights in the map and add their color 
             * contributions to this sample.
             */
            for (const BSP::Light& light : g_bsp.get_lights()) {
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
                for (BSP::Face& otherFace : g_bsp.get_faces()) {
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
                        
                    BSP::Vec3<float> vertex1 = *pVertex;
                    
                    BSP::Vec3<float> vertex2;
                    BSP::Vec3<float> vertex3 = *pVertex;
                    
                    // Line segment from the light to the sample position.
                    gmtl::LineSeg<float> diffLineSeg1(
                        gmtl::Point<float, 3>(
                            lightPos.x, lightPos.y, lightPos.z
                        ),
                        gmtl::Point<float, 3>(
                            samplePos.x, samplePos.y, samplePos.z
                        )
                    );
                    
                    // Line segment from the sample position to the light.
                    // gmtl::LineSeg<float> diffLineSeg2(
                        // gmtl::Point<float, 3>(
                            // samplePos.x, samplePos.y, samplePos.z
                        // ),
                        // gmtl::Point<float, 3>(
                            // lightPos.x, lightPos.y, lightPos.z
                        // )
                    // );
                    
                    const BSP::DPlane& otherFacePlaneData
                        = otherFace.get_planedata();
                        
                    const BSP::Vec3<float>& otherFaceNorm
                        = otherFacePlaneData.normal;
                        
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
                    
                    // If the line segment between the sample point and the 
                    // light does not intersect this face's plane, there's no 
                    // way this face could be blocking the light.
                    // Move on and check the next face.
                    if (!gmtl::intersect(otherFacePlane, diffLineSeg1, t)
                            // && !gmtl::intersect(
                                    // otherFacePlane, diffLineSeg2, t
                                // )
                            ) {
                            
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
                        
                        if (gmtl::intersect(tri, diffLineSeg1, u, v, t)
                                // || gmtl::intersect(tri, diffLineSeg2, u, v, t)
                                ) {
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
                    continue;
                }
                
                /* I CAN SEE THE LIGHT */
                
                double dist = sqrt(
                    diff.x * diff.x +
                    diff.y * diff.y +
                    diff.z * diff.z
                );
                
                double attenuation = 1.0 / light.attenuate(dist);
                
                r += static_cast<uint32_t>(light.r * attenuation);
                g += static_cast<uint32_t>(light.g * attenuation);
                b += static_cast<uint32_t>(light.b * attenuation);
            }
            
            sumR += r;
            sumG += g;
            sumB += b;
            
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
            assert(exp <= 255);
            
            lightSample.r = static_cast<uint8_t>(r);
            lightSample.g = static_cast<uint8_t>(g);
            lightSample.b = static_cast<uint8_t>(b);
            lightSample.exp = static_cast<int8_t>(exp);
            
            // std::cout << "    Sample " << i << " at <"
                // << samplePos.x << ", "
                // << samplePos.y << ", "
                // << samplePos.z << "> has color ("
                // << static_cast<int>(lightSample.r) << ", "
                // << static_cast<int>(lightSample.g) << ", "
                // << static_cast<int>(lightSample.b) << ") * 2^"
                // << static_cast<int>(lightSample.exp) << std::endl;
        }
        
        uint32_t avgR = sumR / numSamples;
        uint32_t avgG = sumG / numSamples;
        uint32_t avgB = sumB / numSamples;
        uint32_t avgExp = 0;
        
        while (avgR > 255 || avgG > 255 || avgB > 255) {
            avgR >>= 1;
            avgG >>= 1;
            avgB >>= 1;
            avgExp++;
        }
        
        assert(avgR <= 255);
        assert(avgG <= 255);
        assert(avgB <= 255);
        assert(avgExp <= 255);
        
        // The BSP format requires us to also store the average lighting value 
        // of each face, for some reason.
        face.set_average_lighting(
            BSP::LightSample {
                static_cast<uint8_t>(avgR),
                static_cast<uint8_t>(avgG),
                static_cast<uint8_t>(avgB),
                static_cast<int8_t>(avgExp),
            }
        );
        
        // Don't ask. I have no idea.
        face.set_styles(std::vector<uint8_t> {0, 0xFF, 0xFF, 0xFF});
    }
    
    /*
     * Mark the BSP as non-fullbright.
     *
     * This tells the engine that there is actually lighting information 
     * embedded in the map.
     */
    g_bsp.set_fullbright(false);
    
    g_bsp.write("out.bsp");
    
    std::cout << "Wrote to file \"out.bsp\"." << std::endl;
    
    return 0;
}
