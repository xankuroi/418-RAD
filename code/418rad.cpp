#include <iostream>
#include <fstream>

#include <string>
#include <vector>
#include <random>

#include <cstdlib>
#include <cstdint>

#include "bsp.h"

static std::random_device r;

static BSP::BSP g_bsp;


static inline uint8_t random_color(void) {
    return static_cast<uint8_t>(r() % 255);
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
    
    for (BSP::Face& face : g_bsp.get_faces()) {
        uint32_t sumR = 0;
        uint32_t sumG = 0;
        uint32_t sumB = 0;
        
        size_t width = face.get_lightmap_width();
        size_t height = face.get_lightmap_height();
        size_t numSamples = width * height;
        
        int i = 0;
        for (BSP::LightSample& lightSample : face.get_lightsamples()) {
            uint8_t r = random_color();
            uint8_t g = random_color();
            uint8_t b = random_color();
            
            // uint8_t r = 255 * (i % 2);
            // uint8_t g = 255 * (i % 2);
            // uint8_t b = 255 * (i % 2);
            
            sumR += r;
            sumG += g;
            sumB += b;
            
            lightSample.r = r;
            lightSample.g = g;
            lightSample.b = b;
            lightSample.exp = 0;
            
            i++;
        }
        
        uint8_t avgR = sumR / numSamples;
        uint8_t avgG = sumG / numSamples;
        uint8_t avgB = sumB / numSamples;
        
        face.set_average_lighting(BSP::LightSample {avgR, avgG, avgB, 0});
        
        std::vector<uint8_t> styles = {0, 0xFF, 0xFF, 0xFF};
        
        face.set_styles(styles);
    }
    
    g_bsp.set_fullbright(false);
    
    g_bsp.write("out.bsp");
    
    std::cout << "Wrote to file \"out.bsp\"." << std::endl;
    
    return 0;
}
