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
    
    std::cout << "BSP Version " << g_bsp.get_format_version() << std::endl;
    std::cout << "Fullbright: " << g_bsp.is_fullbright() << std::endl;
    
    int i = 0;
    for (BSP::Face& face : g_bsp.get_faces()) {
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
        
        std::cout <<  "    Styles:" << std::endl;
        
        int j = 0;
        for (uint8_t style : face.get_styles()) {
            std::cout << "        " << j << " "
                << std::bitset<sizeof(style) * 8>(style)
                << std::endl;
            j++;
        }
        
        const BSP::DFace& faceData = face.get_data();
        
        std::cout << "    Light Offset: "
            << faceData.lightOffset / sizeof(BSP::LightSample) << std::endl;
        
        std::cout << "    Luxels X: "
            << face.get_lightmap_width()
            << std::endl
            << "    Luxels Y: "
            << face.get_lightmap_height()
            << std::endl;
            
        std::vector<BSP::LightSample>& lightSamples = face.get_lightsamples();
        
        std::cout << "    "
            << lightSamples.size() << " Light Samples" << std::endl;
            
        const BSP::LightSample& avgLighting = face.get_average_lighting();
        
        std::cout << "    Average Lighting: ("
            << static_cast<int>(avgLighting.r) << ", "
            << static_cast<int>(avgLighting.g) << ", "
            << static_cast<int>(avgLighting.b) << ") * 2^"
            << static_cast<int>(avgLighting.exp) << std::endl;
            
        std::cout << "    Light Samples:" << std::endl;
        
        j = 0;
        for (BSP::LightSample& lightSample : face.get_lightsamples()) {
            std::cout << "        Sample " << j << ": ("
                << static_cast<int>(lightSample.r) << ", "
                << static_cast<int>(lightSample.g) << ", "
                << static_cast<int>(lightSample.b) << ") * 2^"
                << static_cast<int>(lightSample.exp) << std::endl;
            j++;
        }
        
        i++;
    }
    
    i = 0;
    for (const BSP::Light& light : g_bsp.get_lights()) {
        std::cout << "Light " << i << ":" << std::endl;
        
        const BSP::Vec3& pos = light.get_coords();
        std::cout << "    pos: ("
            << pos.x << ", "
            << pos.y << ", "
            << pos.z << ")" << std::endl;
            
        std::cout << "    r: " << light.r << std::endl;
        std::cout << "    g: " << light.g << std::endl;
        std::cout << "    b: " << light.b << std::endl;
        std::cout << "    brightness: " << light.brightness << std::endl;
        
        i++;
    }
    
    return 0;
}
