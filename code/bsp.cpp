#include <iostream>
#include <sstream>
#include <fstream>

#include <string>

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>

#include <cstring>
#include <cassert>
#include <cmath>

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
        load_lump(file, LUMP_FACES_HDR, dfaces);
        
        // The Faces HDR lump can be empty...
        if (dfaces.size() == 0) {
            load_lump(file, LUMP_FACES, dfaces);
        }
        
        std::vector<LightSample> lightSamples;
        
        if (!is_fullbright()) {
            load_lump(file, LUMP_LIGHTING_HDR, lightSamples);
        }
        
        for (const DFace& faceData : dfaces) {
            m_faces.push_back(
                Face(*this, faceData, lightSamples)
            );
        }
        
        load_lump(file, LUMP_LEAVES, m_leaves);
        
        std::vector<char> entData;
        load_lump(file, LUMP_ENTITIES, entData);
        
        m_entData.assign(entData.begin(), entData.end());
        
        load_lights(m_entData);
        
        if (!is_fullbright()) {
            load_lump(file, LUMP_WORLDLIGHTS_HDR, m_worldLights);
        }
        
        load_gamelumps(file);
        
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
        for (unsigned int lumpID=0; lumpID<HEADER_LUMPS; lumpID++) {
            if (m_loadedLumps.find(lumpID) != m_loadedLumps.end()) {
                // Lump was previously loaded; ignore
                continue;
            }
            
            Lump& lump = m_header.lumps[lumpID];
            
            if (lump.fileOffset == 0 || lump.fileLen == 0) {
                // Unused lump; ignore
                continue;
            }
            
            m_extraLumps[lumpID] = std::unique_ptr<std::vector<uint8_t>>(
                new std::vector<uint8_t>
            );
            
            load_lump(
                file,
                static_cast<LumpType>(lumpID),
                *m_extraLumps[lumpID]
            );
        }
    }
    
    template<typename Container>
    void BSP::load_single_gamelump(
            std::ifstream& file,
            const GameLump& gameLump,
            Container& dest
        ) {
        
        std::ifstream::off_type offset = gameLump.fileOffset;
        size_t lumpSize = gameLump.fileLen;
        size_t numElems = lumpSize / sizeof(typename Container::value_type);
        
        dest.resize(numElems);
        
        file.seekg(offset);
        file.read(reinterpret_cast<char*>(dest.data()), lumpSize);
    }
    
    void BSP::load_gamelumps(std::ifstream& file) {
        std::vector<uint8_t> gameLumpData;
        load_lump(file, LUMP_GAME_LUMP, gameLumpData);
        
        GameLumpHeader& gameLumpHeader = *reinterpret_cast<GameLumpHeader*>(
            gameLumpData.data()
        );
        
        int32_t lumpCount = gameLumpHeader.lumpCount;
        
        GameLump* pGameLumps = reinterpret_cast<GameLump*>(
            &gameLumpHeader.firstGameLump
        );
        
        for (int i=0; i<lumpCount; i++) {
            GameLump& gameLump = pGameLumps[i];
            
            switch (gameLump.id) {
                // case GAMELUMP_STATIC_PROPS: {
                    // std::vector<>
                    
                    // break;
                // }
                default: {
                    m_extraGameLumps[gameLump.id]
                        = std::unique_ptr<std::vector<uint8_t>>(
                            new std::vector<uint8_t>
                        );
                        
                    load_single_gamelump(
                        file,
                        gameLump,
                        *m_extraGameLumps[gameLump.id]
                    );
                }
            }
            
            m_gameLumps[gameLump.id] = gameLump;
        }
    }
    
    int BSP::get_format_version(void) const {
        return m_header.version;
    }
    
    const Header& BSP::get_header(void) const {
        return m_header;
    }
    
    std::vector<Face>& BSP::get_faces(void) {
        return m_faces;
    }
    
    const std::vector<Light>& BSP::get_lights(void) const {
        return m_lights;
    }
    
    const std::vector<DWorldLight>& BSP::get_worldlights(void) const {
        return m_worldLights;
    }
    
    const std::string& BSP::get_entdata(void) {
        return m_entData;
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
        
        save_lump(
            file, LUMP_LEAVES, m_leaves,
            offsets, sizes
        );
        
        std::vector<char> entData(m_entData.size());
        entData.assign(m_entData.begin(), m_entData.end());
        
        save_lump(
            file, LUMP_ENTITIES, entData,
            offsets, sizes
        );
        
        if (!is_fullbright()) {
            save_lights(file, offsets, sizes);
            
            // Temp hack for ambient lighting
            size_t numLeaves = m_leaves.size();
            
            std::vector<DLeafAmbientLighting> ambientLighting(numLeaves);
            
            for (DLeafAmbientLighting& ambient : ambientLighting) {
                for (LightSample& sample : ambient.cube.color) {
                    sample.r = 0;
                    sample.g = 255;
                    sample.b = 0;
                    sample.exp = 0;
                }
                
                ambient.x = 128;
                ambient.y = 128;
                ambient.z = 128;
            }
            
            save_lump(
                file, LUMP_LEAF_AMBIENT_LIGHTING_HDR, ambientLighting,
                offsets, sizes
            );
            
            std::vector<DLeafAmbientIndex> ambientIndices(numLeaves);
            
            int i = 0;
            for (DLeafAmbientIndex& ambientIndex : ambientIndices) {
                ambientIndex.ambientSampleCount = 1;
                ambientIndex.firstAmbientSample = i;
                i++;
            }
            
            save_lump(
                file, LUMP_LEAF_AMBIENT_INDEX_HDR, ambientIndices,
                offsets, sizes
            );
        }
        
        save_gamelumps(file, offsets, sizes);
        
        save_extras(file, offsets, sizes);
        
        /* Lump offset and size fixup */
        for (unsigned int i=0; i<HEADER_LUMPS; i++) {
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
            const Container& src,
            std::unordered_map<int, std::ofstream::off_type>& offsets,
            std::unordered_map<int, size_t>& sizes,
            bool isExtraLump
            ) {
            
        size_t size = src.size();
        size *= sizeof(typename Container::value_type);
        
        offsets[lumpID] = file.tellp();
        sizes[lumpID] = size;
        
        file.write(reinterpret_cast<const char*>(src.data()), size);
        
        // TODO: Remove this. (Or do we need to remove it...?)
        if (!isExtraLump) {
            m_extraLumps.erase(lumpID);
        }
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
        
        save_lump(
            file, LUMP_FACES_HDR, dfaces,
            offsets, sizes
        );
    }
    
    void BSP::save_lights(
            std::ofstream& file,
            std::unordered_map<int, std::ofstream::off_type>& offsets,
            std::unordered_map<int, size_t>& sizes
            ) {
        
        std::vector<DWorldLight> worldLights;
        
        for (const Light& light : get_lights()) {
            worldLights.push_back(light.to_worldlight());
        }
        
        save_lump(
            file, LUMP_WORLDLIGHTS_HDR, worldLights,
            offsets, sizes
        );
    }
    
    void BSP::save_extras(
            std::ofstream& file, 
            std::unordered_map<int, std::ofstream::off_type>& offsets,
            std::unordered_map<int, size_t>& sizes
            ) {
            
        using Pair = std::pair<
            const int,
            std::unique_ptr<std::vector<uint8_t>>
        >;
        
        for (const Pair& pair : m_extraLumps) {
            LumpType lumpID = static_cast<LumpType>(pair.first);
            const std::unique_ptr<std::vector<uint8_t>>& pData = pair.second;
            
            assert(m_extraLumps.find(lumpID) != m_extraLumps.end());
            assert(pData);
            
            save_lump(
                file, lumpID, *pData,
                offsets, sizes,
                true    // This is an extra lump.
            );
        }
    }
    
    void BSP::save_gamelumps(
            std::ofstream& file,
            std::unordered_map<int, std::ofstream::off_type>& offsets,
            std::unordered_map<int, size_t>& sizes
            ) {
            
        using Pair = std::pair<
            const int32_t,
            std::unique_ptr<std::vector<uint8_t>>
        >;
        
        for (const Pair& pair : m_extraGameLumps) {
            int32_t gameLumpID = pair.first;
            const std::vector<uint8_t>& gameLumpData = *pair.second;
            
            save_single_gamelump(file, gameLumpID, gameLumpData);
        }
        
        int32_t gameLumpCount = m_gameLumps.size();
        
        size_t size = sizeof(int32_t) + gameLumpCount * sizeof(GameLump);
        
        offsets[LUMP_GAME_LUMP] = file.tellp();
        sizes[LUMP_GAME_LUMP] = size;
        
        file.write(reinterpret_cast<char*>(&gameLumpCount), sizeof(int32_t));
        
        for (const std::pair<int32_t, GameLump>& pair : m_gameLumps) {
            const GameLump& gameLump = pair.second;
            file.write(
                reinterpret_cast<const char*>(&gameLump),
                sizeof(GameLump)
            );
        }
    }
    
    template<typename Container>
    void BSP::save_single_gamelump(
            std::ofstream& file,
            int32_t gameLumpID,
            const Container& src
            ) {
            
        std::ofstream::off_type offset = file.tellp();
        
        size_t size = src.size();
        size *= sizeof(typename Container::value_type);
        
        GameLump& gameLump = m_gameLumps[gameLumpID];
        gameLump.fileOffset = offset;
        gameLump.fileLen = size;
        
        file.write(reinterpret_cast<const char*>(src.data()), size);
    }
    
    const std::unordered_map<int, std::unique_ptr<std::vector<uint8_t>>>&
    BSP::get_extras(void) const{
        return m_extraLumps;
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
        
        assert(
            entStr != ""
                || (m_index >= m_entData.size() && entStart == -1)
        );
        
        Ent nextEnt;
        
        if (entStr == "") {
            return nextEnt;
        }
        
        std::string key = "";
        int fieldStart = -1;
        
        for (size_t i=0; i<entStr.size(); i++) {
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
            m_avgLightSample {0, 0, 0, 0} {
            
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
        
        size_t numSamples = width * height;
        size_t sampleOffset = m_faceData.lightOffset / sizeof(LightSample);
        
        // This face has no lighting.
        if (sampleOffset == 0) {
            return;
        }
        
        assert(samples.size() > 1);
        assert(samples.size() >= numSamples + 1);
        assert(samples.size() - sampleOffset >= numSamples);
        
        std::vector<LightSample>::const_iterator lightSampleStart
            = samples.begin() + sampleOffset;
            
        set_average_lighting(lightSampleStart[-1]);
        
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
    
    void Face::set_styles(const std::vector<uint8_t>& styles) {
        assert(styles.size() == 4);
        
        for (int i=0; i<4; i++) {
            m_faceData.styles[i] = styles[i];
        }
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
    
    static inline double linear_from_encoded(double encoded) {
        return pow(encoded / 255.0, GAMMA) * 255.0;
    }
    
    static inline double encoded_from_linear(double linear) {
        return pow(linear / 255.0, INV_GAMMA) * 255.0;
    }
    
    static inline double attenuate(double x, double c, double l, double q) {
        return c + l * x + q * x * x;
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
        double brightness = convert_str<double>(s) / 255.0;
        
        r = linear_from_encoded(r) * brightness;
        g = linear_from_encoded(g) * brightness;
        b = linear_from_encoded(b) * brightness;
        
        double attenuation = attenuate(100.0, 0.0, 0.0, 1.0);
        
        r *= attenuation;
        g *= attenuation;
        b *= attenuation;
    }
    
    const Vec3& Light::get_coords(void) const {
        return m_coords;
    }
    
    DWorldLight Light::to_worldlight(void) const {
        return DWorldLight {
            get_coords(),
            Vec3 {
                static_cast<float>(r / 255.0),
                static_cast<float>(g / 255.0),
                static_cast<float>(b / 255.0),
            },
            Vec3 {1.0, 0.0, 0.0},
            0,
            EMIT_POINT,
            0x0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0x0,
            0,
            0,
        };
    }
}
