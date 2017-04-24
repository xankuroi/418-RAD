#include <iostream>
#include <fstream>
#include <sstream>

#include <string>
#include <vector>
#include <unordered_set>

#include <cstdlib>
#include <cassert>

#include "bsp.h"


int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Invalid arguments." << std::endl;
        return 1;
    }
    
    std::unordered_set<int> lumpIDs;
    
    int numLumps = argc - 3;
    while (numLumps > 0) {
        std::stringstream s;
        
        int lumpID;
        
        s << argv[lumpIDs.size() + 3];
        s >> lumpID;
        
        lumpIDs.insert(lumpID);
        
        numLumps--;
    }
    
    const std::string filename1(argv[1]);
    const std::string filename2(argv[2]);
    
    std::ifstream file1(filename1);
    std::ifstream file2(filename2);
    
    BSP::Header header1;
    BSP::Header header2;
    
    file1.seekg(0);
    file1.read(reinterpret_cast<char*>(&header1), sizeof(BSP::Header));
    
    file2.seekg(0);
    file2.read(reinterpret_cast<char*>(&header2), sizeof(BSP::Header));
    
    std::vector<unsigned int> mismatches;
    
    for (unsigned int i=0; i<BSP::HEADER_LUMPS; i++) {
        if (lumpIDs.size() != 0 && lumpIDs.find(i) == lumpIDs.end()) {
            continue;
        }
        
        std::cout << "Checking lump " << i << "..." << std::endl;
        
        const BSP::Lump& lump1 = header1.lumps[i];
        const BSP::Lump& lump2 = header2.lumps[i];
        
        std::vector<uint8_t> data1(lump1.fileLen);
        std::vector<uint8_t> data2(lump2.fileLen);
        
        std::ifstream::off_type offset1 = lump1.fileOffset;
        std::ifstream::off_type offset2 = lump2.fileOffset;
        
        file1.seekg(offset1);
        file1.read(reinterpret_cast<char*>(data1.data()), lump1.fileLen);
        
        file2.seekg(offset2);
        file2.read(reinterpret_cast<char*>(data2.data()), lump2.fileLen);
        
        std::cout << "    " << filename1 << " size: "
            << data1.size() << std::endl;
        
        std::cout << "    " << filename2 << " size: "
            << data2.size() << std::endl;
        
        if (data1.size() != data2.size()) {
            std::cout << "    Size mismatch!" << std::endl;
            mismatches.push_back(i);
            continue;
        }
        
        for (unsigned int j=0; j<data1.size(); j++) {
            if (data1[j] != data2[j]) {
                std::cout << "    Byte " << j << " mismatch!" << std::endl;
                mismatches.push_back(i);
                break;
            }
        }
    }
    
    if (mismatches.size() == 0) {
        std::cout << "Lumps match!" << std::endl;
    }
    else {
        std::cout << "Mismatched lumps:" << std::endl;
        for (int i : mismatches) {
            std::cout << i << std::endl;
        }
    }
    
    return 0;
}
