#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include "bsp.h"


namespace BSP {
    
    /*************
     * BSP Class *
     *************/
     
    BSP::BSP() {}
    
    BSP::BSP(const std::string& filename) : BSP() {
        std::ifstream file(filename, std::ios::binary);
        
        if (file.fail()) {
            throw InvalidBSP("Could not open " + filename + "!");
        }
        
        init(file);
    }
    
    BSP::BSP(std::ifstream& file) : BSP() {
        init(file);
    }
    
    void BSP::init(std::ifstream& file) {
        file.read(reinterpret_cast<char*>(&m_header), sizeof(Header));
        
        if (m_header.ident != IDBSPHEADER) {
            throw InvalidBSP("Bad BSP! :(");
        }
        
        load_lump(file, LUMP_PLANES, m_planes);
        load_lump(file, LUMP_VERTICES, m_vertices);
        load_lump(file, LUMP_EDGES, m_edges);
        load_lump(file, LUMP_SURFEDGES, m_surfEdges);
        load_lump(file, LUMP_LIGHTING, m_lightSamples);
        load_lump(file, LUMP_LIGHTING_HDR, m_hdrLightSamples);
        
        std::vector<DFace> dfaces;
        load_lump(file, LUMP_FACES, dfaces);
        
        for (DFace& faceData : dfaces) {
            m_faces.push_back(
                Face(*this, faceData)
            );
        }
        
        load_extras(file);
    }
    
    template<typename T>
    void BSP::load_lump(
            std::ifstream& file,
            const LumpType lumpID,
            std::vector<T>& dest
            ) {
            
        Lump& lump = m_header.lumps[lumpID];
        
        std::ifstream::off_type offset = lump.fileOffset;
        size_t lumpSize = lump.fileLen;
        size_t numElems = lumpSize / sizeof(T);
        
        dest.resize(numElems);
        
        file.seekg(offset);
        file.read(reinterpret_cast<char*>(dest.data()), lumpSize);
        
        m_loadedLumps.insert(lumpID);
    }
    
    void BSP::load_extras(std::ifstream& file) {
        for (int lumpID=0; lumpID<HEADER_LUMPS; lumpID++) {
            if (m_loadedLumps.find(lumpID) != m_loadedLumps.end()) {
                // Lump was previously loaded; ignore
                continue;
            }
            
            Lump& lump = m_header.lumps[lumpID];
            
            if (lump.fileOffset == 0 && lump.fileLen == 0) {
                // Unused lump; ignore
                continue;
            }
            
            std::ifstream::off_type offset = lump.fileOffset;
            size_t lumpSize = lump.fileLen;
            
            m_extra[lumpID] = std::vector<uint8_t>(lumpSize);
            std::vector<uint8_t>& extraBuffer = m_extra[lumpID];
            
            file.seekg(offset);
            file.read(
                reinterpret_cast<char*>(extraBuffer.data()),
                lumpSize
            );
        }
    }
    
    const std::vector<Face>& BSP::get_faces(void) const {
        return m_faces;
    }
    
    std::vector<LightSample>& BSP::get_lightsamples(void) {
        return m_lightSamples;
    }
    
    std::vector<LightSample>& BSP::get_hdr_lightsamples(void) {
        return m_hdrLightSamples;
    }
    
    void BSP::print_lump_offsets(void) {
        for (Lump& lump : m_header.lumps) {
            std::cout << lump.fileOffset << std::endl;
        }
    }
    
    /**************
     * Face Class *
     **************/
     
    Face::Face(const BSP& bsp, const DFace& faceData) :
            m_faceData(faceData),
            m_planeData(bsp.m_planes[faceData.planeNum]) {
            
        const std::vector<Vec3>& vertices = bsp.m_vertices;
        const std::vector<DEdge>& dEdges = bsp.m_edges;
        const std::vector<int32_t>& surfEdges = bsp.m_surfEdges;
        
        int firstEdge = faceData.firstEdge;
        int lastEdge = faceData.firstEdge + faceData.numEdges;
        
        for (int i=firstEdge; i<lastEdge; i++) {
            int32_t surfEdge = surfEdges[i];
            
            bool firstToSecond = (surfEdge >= 0);
            
            if (!firstToSecond) {
                surfEdge *= -1;
            }
            
            const DEdge& dEdge = dEdges[surfEdge];
            
            Edge edge;
            
            if (firstToSecond) {
                edge.vertex1 = vertices[dEdge.vertex1];
                edge.vertex2 = vertices[dEdge.vertex2];
            }
            else {
                edge.vertex1 = vertices[dEdge.vertex2];
                edge.vertex2 = vertices[dEdge.vertex1];
            }
            
            m_edges.push_back(edge);
        }
    }
    
    const std::vector<Edge>& Face::get_edges(void) const {
        return m_edges;
    }
}
