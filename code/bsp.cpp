#include <iostream>
#include <sstream>
#include <fstream>

#include <string>
#include <cstring>
#include <cassert>

#include <vector>
#include <unordered_set>
#include <unordered_map>

#include "bsp.h"


namespace BSP {
    
    /*************
     * BSP Class *
     *************/
     
    BSP::BSP() : m_fullbright(true) {}
    
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
        
        int version = get_format_version();
        
        if (version != 20) {
            std::stringstream s;
            
            s << "Unsupported BSP version: " << version;
            
            throw InvalidBSP(s.str());
        }
        
        set_fullbright(m_header.lumps[LUMP_LIGHTING_HDR].fileLen == 0);
        
        load_lump(file, LUMP_PLANES, m_planes);
        load_lump(file, LUMP_VERTICES, m_vertices);
        load_lump(file, LUMP_EDGES, m_edges);
        load_lump(file, LUMP_SURFEDGES, m_surfEdges);
        
        std::vector<DFace> dfaces;
        load_lump(file, LUMP_FACES, dfaces);
        
        std::vector<LightSample> lightSamples;
        
        if (!is_fullbright()) {
            load_lump(file, LUMP_LIGHTING_HDR, lightSamples);
        }
        
        for (const DFace& faceData : dfaces) {
            m_faces.push_back(
                Face(*this, faceData, lightSamples)
            );
        }
        
        std::vector<char> entData;
        load_lump(file, LUMP_ENTITIES, entData);
        
        m_entData.assign(entData.begin(), entData.end());
        
        load_lights(m_entData);
        
        load_extras(file);
    }
    
    template<typename Container>
    void BSP::load_lump(
            std::ifstream& file,
            const LumpType lumpID,
            Container& dest
            ) {
            
        Lump& lump = m_header.lumps[lumpID];
        
        std::ifstream::off_type offset = lump.fileOffset;
        size_t lumpSize = lump.fileLen;
        size_t numElems = lumpSize / sizeof(typename Container::value_type);
        
        dest.resize(numElems);
        
        file.seekg(offset);
        file.read(reinterpret_cast<char*>(dest.data()), lumpSize);
        
        m_loadedLumps.insert(lumpID);
    }
    
    void BSP::load_lights(const std::string& entData) {
        EntityParser entParser(entData);
        
        std::unordered_map<std::string, std::string> entity;
        
        while ((entity = entParser.next_ent()).size() > 0) {
            assert(entity.find("classname") != entity.end());
            
            const std::string& classname = entity["classname"];
            
            if (classname == "light"
                    || classname == "light_spot") {
                m_lights.push_back(Light(entity));
            }
        }
    }
    
    void BSP::load_extras(std::ifstream& file) {
        for (int lumpID=0; lumpID<HEADER_LUMPS; lumpID++) {
            if (m_loadedLumps.find(lumpID) != m_loadedLumps.end()) {
                // Lump was previously loaded; ignore
                continue;
            }
            
            Lump& lump = m_header.lumps[lumpID];
            
            if (lump.fileOffset == 0 || lump.fileLen == 0) {
                // Unused lump; ignore
                continue;
            }
            
            std::ifstream::off_type offset = lump.fileOffset;
            size_t lumpSize = lump.fileLen;
            
            m_extras[lumpID] = std::vector<uint8_t>(lumpSize);
            std::vector<uint8_t>& extraBuffer = m_extras[lumpID];
            
            file.seekg(offset);
            file.read(
                reinterpret_cast<char*>(extraBuffer.data()),
                lumpSize
            );
        }
    }
    
    int BSP::get_format_version(void) const {
        return m_header.version;
    }
    
    std::vector<Face>& BSP::get_faces(void) {
        return m_faces;
    }
    
    const std::vector<Light>& BSP::get_lights(void) {
        return m_lights;
    }
    
    bool BSP::is_fullbright(void) const {
        return m_fullbright;
    }
    
    void BSP::set_fullbright(bool fullbright) {
        m_fullbright = fullbright;
    }
    
    void BSP::write(const std::string& filename) {
        std::ofstream f(filename, std::ios::binary);
        write(f);
    }
    
    void BSP::write(std::ofstream& file) {
        // Write out our current header.
        // We'll fix the sizes and offsets later.
        file.seekp(0);
        file.write(reinterpret_cast<char*>(&m_header), sizeof(m_header));
        
        // Keep track of which lumps go where.
        std::unordered_map<int, std::ofstream::off_type> offsets;
        
        // Keep track of lump sizes.
        std::unordered_map<int, size_t> sizes;
        
        save_lump(
            file, LUMP_VERTICES, m_vertices,
            offsets, sizes
        );
        
        save_lump(
            file, LUMP_EDGES, m_edges,
            offsets, sizes
        );
        
        save_lump(
            file, LUMP_SURFEDGES, m_surfEdges,
            offsets, sizes
        );
        
        save_lump(
            file, LUMP_PLANES, m_planes,
            offsets, sizes
        );
        
        save_faces(file, offsets, sizes);
        
        std::vector<char> entData;
        entData.assign(m_entData.begin(), m_entData.end());
        
        save_lump(
            file, LUMP_ENTITIES, entData,
            offsets, sizes
        );
        
        save_extras(file, offsets, sizes);
        
        /* Lump offset and size fixup */
        for (int i=0; i<HEADER_LUMPS; i++) {
            if (offsets.find(i) == offsets.end()) {
                // Unused lump; ignore
                assert(sizes.find(i) == sizes.end());
                continue;
            }
            
            Lump& lump = m_header.lumps[i];
            
            lump.fileOffset = offsets[i];
            lump.fileLen = sizes[i];
        }
        
        // Write the fixed lump offsets and sizes.
        file.seekp(0);
        file.write(reinterpret_cast<char*>(&m_header), sizeof(m_header));
    }
    
    template<typename Container>
    void BSP::save_lump(
            std::ofstream& file,
            const LumpType lumpID,
            Container& src,
            std::unordered_map<int, std::ofstream::off_type>& offsets,
            std::unordered_map<int, size_t>& sizes
            ) {
            
        size_t size = src.size();
        size *= sizeof(typename Container::value_type);
        
        offsets[lumpID] = file.tellp();
        sizes[lumpID] = size;
        
        file.write(reinterpret_cast<char*>(src.data()), size);
    }
    
    void BSP::save_faces(
            std::ofstream& file,
            std::unordered_map<int, std::ofstream::off_type>& offsets,
            std::unordered_map<int, size_t>& sizes
            ) {
            
        std::vector<LightSample> lightSamples;
        std::vector<DFace> dfaces;
        
        for (Face& face : m_faces) {
            if (!is_fullbright()) {
                lightSamples.push_back(face.get_average_lighting());
                
                face.set_lightlump_offset(lightSamples.size());
                
                for (LightSample& lightSample : face.get_lightsamples()) {
                    lightSamples.push_back(lightSample);
                }
            }
            
            dfaces.push_back(face.get_data());
        }
        
        if (!is_fullbright()) {
            save_lump(
                file, LUMP_LIGHTING_HDR, lightSamples,
                offsets, sizes
            );
        }
        else {
            offsets[LUMP_LIGHTING_HDR] = 0;
            sizes[LUMP_LIGHTING_HDR] = 0;
        }
        
        save_lump(
            file, LUMP_FACES, dfaces,
            offsets, sizes
        );
    }
    
    void BSP::save_extras(
            std::ofstream& file, 
            std::unordered_map<int, std::ofstream::off_type>& offsets,
            std::unordered_map<int, size_t>& sizes
            ) {
            
        for (std::pair<int, std::vector<uint8_t>> pair : m_extras) {
            save_lump(
                file, static_cast<LumpType>(pair.first), pair.second,
                offsets, sizes
            );
        }
    }
    
    const std::unordered_map<int, std::vector<uint8_t>>&
    BSP::get_extras(void) const{
        return m_extras;
    }
    
    
    /**********************
     * EntityParser Class *
     **********************/
     
    EntityParser::EntityParser(const std::string& entData) :
        m_index(0),
        m_entData(entData) {}
        
    std::unordered_map<std::string, std::string>
    EntityParser::next_ent(void) {
        using Ent = std::unordered_map<std::string, std::string>;
        
        int entStart = -1;
        std::string entStr = "";
        
        while (m_index < m_entData.size()) {
            char c = m_entData[m_index];
            
            switch (c) {
                case '{':
                    assert(entStart == -1);
                    entStart = m_index;
                    break;
                    
                case '}':
                    assert(entStart != -1);
                    int count = m_index - entStart - 1;
                    entStr = m_entData.substr(entStart + 1, count);
                    break;
            }
            
            if (entStr.size() > 0) {
                m_index++;
                break;
            }
            
            m_index++;
        }
        
        assert(entStr != "" || m_index >= m_entData.size() && entStart == -1);
        
        Ent nextEnt;
        
        if (entStr == "") {
            return nextEnt;
        }
        
        std::string key = "";
        int fieldStart = -1;
        
        for (int i=0; i<entStr.size(); i++) {
            char c = entStr[i];
            
            if (c == '"') {
                if (fieldStart == -1) {
                    fieldStart = i;
                }
                else {
                    int count = i - fieldStart - 1;
                    std::string field = entStr.substr(
                        fieldStart + 1,
                        count
                    );
                    
                    fieldStart = -1;
                    
                    if (key == "") {
                        key = field;
                    }
                    else {
                        assert(nextEnt.find(key) == nextEnt.end());
                        nextEnt[key] = field;
                        key = "";
                    }
                }   
            }
        }
        
        assert(key == "");
        
        return nextEnt;
    }
    
    
    /**************
     * Face Class *
     **************/
     
    Face::Face(
            const BSP& bsp,
            const DFace& faceData,
            const std::vector<LightSample>& lightSamples
            ) :
            m_faceData(faceData),
            m_planeData(bsp.m_planes[faceData.planeNum]),
            m_lightSamples(get_lightmap_width() * get_lightmap_height()),
            m_avgLightSample{0, 0, 0, 0} {
            
        const std::vector<Vec3>& vertices = bsp.m_vertices;
        const std::vector<DEdge>& dEdges = bsp.m_edges;
        const std::vector<int32_t>& surfEdges = bsp.m_surfEdges;
        
        load_edges(faceData, vertices, dEdges, surfEdges);
        
        if (!bsp.is_fullbright()) {
            load_lightsamples(lightSamples);
        }
    }
    
    void Face::load_edges(
            const DFace& faceData,
            const std::vector<Vec3>& vertices,
            const std::vector<DEdge>& dEdges,
            const std::vector<int32_t>& surfEdges
            ) {
            
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
    
    void Face::load_lightsamples(const std::vector<LightSample>& samples) {
        int32_t width = get_lightmap_width();
        int32_t height = get_lightmap_height();
        
        int numSamples = width * height;
        int sampleOffset = m_faceData.lightOffset / sizeof(LightSample);
        
        assert(samples.size() >= numSamples + 1);
        assert(samples.size() - sampleOffset >= numSamples);
        
        std::vector<LightSample>::const_iterator lightSampleStart
            = samples.begin() + sampleOffset;
            
        assert(samples.size() > 1);
        set_average_lighting(lightSampleStart[-1]);
        
        m_lightSamples.reserve(numSamples);
        m_lightSamples.assign(lightSampleStart, lightSampleStart + numSamples);
        
        assert(m_lightSamples.size() == numSamples);
    }
    
    const DFace& Face::get_data(void) const {
        return m_faceData;
    }
    
    const std::vector<Edge>& Face::get_edges(void) const {
        return m_edges;
    }
    
    const std::vector<uint8_t> Face::get_styles(void) const {
        std::vector<uint8_t> styles(4);
        
        for (int i=0; i<4; i++) {
            styles[i] = m_faceData.styles[i];
        }
        
        return styles;
    }
    
    int32_t Face::get_lightmap_width(void) const {
        return m_faceData.lightmapTextureSizeInLuxels[0] + 1;
    }
    
    int32_t Face::get_lightmap_height(void) const {
        return m_faceData.lightmapTextureSizeInLuxels[1] + 1;
    }
    
    std::vector<LightSample>& Face::get_lightsamples(void) {
        return m_lightSamples;
    }
    
    LightSample Face::get_average_lighting(void) const {
        return m_avgLightSample;
    }
    
    void Face::set_average_lighting(const LightSample& sample) {
        m_avgLightSample = sample;
    }
    
    void Face::set_lightlump_offset(int32_t offset) {
        m_faceData.lightOffset = offset * sizeof(LightSample);
    }
    
    
    /***************
     * Light Class *
     ***************/
     
    template<typename T>
    static inline float convert_str(const std::string& str) {
        T result;
        
        std::stringstream converter;
        converter << str;
        converter >> result;
        
        return result;
    }
    
    static Vec3 vec3_from_str(const std::string& str) {
        std::stringstream stream(str);
        
        float x;
        float y;
        float z;
        
        std::string s;
        
        std::getline(stream, s, ' ');
        x = convert_str<float>(s);
        
        std::getline(stream, s, ' ');
        y = convert_str<float>(s);
        
        std::getline(stream, s, ' ');
        z = convert_str<float>(s);
        
        return Vec3 {x, y, z};
    }
    
    Light::Light(const std::unordered_map<std::string, std::string>& entity) :
            m_coords(vec3_from_str(entity.at("origin"))) {
            
        assert(entity.find("_light") != entity.end());
        
        std::stringstream stream(entity.at("_light"));
        std::stringstream converter;
        
        std::string s;
        
        std::getline(stream, s, ' ');
        r = convert_str<double>(s);
        
        std::getline(stream, s, ' ');
        g = convert_str<double>(s);
        
        std::getline(stream, s, ' ');
        b = convert_str<double>(s);
        
        std::getline(stream, s, ' ');
        brightness = convert_str<double>(s);
    }
    
    const Vec3& Light::get_coords(void) const {
        return m_coords;
    }
}
