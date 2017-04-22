#include <iostream>
#include <fstream>

#include <string>
#include <vector>
#include <bitset>

#include <cstdlib>

#include "bsp.h"

static BSP::BSP g_bsp;


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
    
    for (BSP::Face& face : g_bsp.get_faces()) {
        face.set_average_lighting(BSP::LightSample {255, 0, 0, 0});
        
        for (BSP::LightSample& lightSample : face.get_lightsamples()) {
            lightSample.r = 255;
            lightSample.g = 0;
            lightSample.b = 0;
            lightSample.exp = 0;
        }
    }
    
    g_bsp.set_fullbright(false);
    
    g_bsp.write("out.bsp");
    
    std::cout << "Wrote to file \"out.bsp\"." << std::endl;
    
    return 0;
}
