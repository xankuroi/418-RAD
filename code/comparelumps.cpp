#include <iostream>
#include <fstream>

#include <string>
#include <vector>
#include <bitset>

#include <cstdlib>
#include <cassert>

#include "bsp.h"


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Invalid arguments." << std::endl;
        return 1;
    }
    
    const std::string filename1(argv[1]);
    const std::string filename2(argv[2]);
    
    BSP::BSP bsp1;
    BSP::BSP bsp2;
    
    try {
        bsp1 = BSP::BSP(filename1);
        bsp2 = BSP::BSP(filename2);
    }
    catch (BSP::InvalidBSP e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    
    const std::unordered_map<int, std::vector<uint8_t>>& extras1
        = bsp1.get_extras();
        
    const std::unordered_map<int, std::vector<uint8_t>>& extras2
        = bsp2.get_extras();
    
    for (int i=0; i<BSP::HEADER_LUMPS; i++) {
        std::cout << "Checking lump " << i << "..." << std::endl;
        
        if (extras1.find(i) == extras1.end()) {
            assert(extras2.find(i) == extras2.end());
            continue;
        }
        
        const std::vector<uint8_t>& data1 = extras1.at(i);
        const std::vector<uint8_t>& data2 = extras2.at(i);
        
        assert(data1.size() == data2.size());
        
        for (int j=0; j<data1.size(); j++) {
            assert(data1[j] == data2[j]);
        }
    }
    
    std::cout << "Extra lumps match!" << std::endl;
    
    return 0;
}
