#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>

#include "bsp.h"

static BSP::BSP g_bsp;


int main(int argc, char** argv) {
    if (argc != 2) {
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
    
    int i = 0;
    for (const BSP::LightSample& sample : g_bsp.get_lightsamples()) {
        std::cout << "Sample " << i << ": ("
            << static_cast<int>(sample.r) << ", "
            << static_cast<int>(sample.g) << ", "
            << static_cast<int>(sample.b) << ") * 2^"
            << static_cast<int>(sample.exp) << std::endl;
            
        i++;
    }
    
    i = 0;
    for (const BSP::Face& face : g_bsp.get_faces()) {
        std::cout << "Face " << i << ":" << std::endl;
        
        for (const BSP::Edge& edge : face.get_edges()) {
            const BSP::Vec3& vertex1 = edge.vertex1;
            const BSP::Vec3& vertex2 = edge.vertex2;
            
            std::cout << "    ("
                << vertex1.x << ", "
                << vertex1.y << ", "
                << vertex1.z << ") -> ("
                << vertex2.x << ", "
                << vertex2.y << ", "
                << vertex2.z << ")"
                << std::endl;
        }
        
        i++;
    }
    
    // g_bsp.print_lump_offsets();
    
    return 0;
}
