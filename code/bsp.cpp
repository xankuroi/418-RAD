#include <iostream>
#include <sstream>
#include <fstream>

#include <string>

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <limits>
#include <algorithm>

#include <cstring>
#include <cassert>
#include <cmath>

#include <gmtl/Matrix.h>
#include <gmtl/MatrixOps.h>

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
        
        load_lump(file, LUMP_MODELS, m_models);
        load_lump(file, LUMP_PLANES, m_planes);
        load_lump(file, LUMP_VERTICES, m_vertices);
        load_lump(file, LUMP_EDGES, m_edges);
        load_lump(file, LUMP_SURFEDGES, m_surfEdges);
        
        load_lump(file, LUMP_FACES_HDR, m_dFaces);
        
        // The Faces HDR lump can be empty...
        if (m_dFaces.size() == 0) {
            load_lump(file, LUMP_FACES, m_dFaces);
        }
        
        
        if (!is_fullbright()) {
            load_lump(file, LUMP_LIGHTING_HDR, m_lightSamples);
        }

        load_lump(file, LUMP_TEXINFO, m_texInfos);
        load_lump(file, LUMP_TEXDATA, m_texDatas);
        
        for (DFace& faceData : m_dFaces) {
            //std::cout << faceData.texInfo << std::endl;

            m_faces.push_back(Face(*this, faceData));

            Face& face = m_faces.back();

            if (is_fullbright()) {
                // Average lighting entry
                m_lightSamples.push_back(RGBExp32 {0, 0, 0, 0});

                // The lightmap offset always points to the entry *after* the
                // average lighting entry.
                face.set_lightmap_offset(m_lightSamples.size());

                // The rest of the lightmap
                for (size_t i=0; i<face.get_lightmap_size(); i++) {
                    m_lightSamples.push_back(RGBExp32 {0, 0, 0, 0});
                }
            }
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
        
        Entity entity;
        
        while ((entity = entParser.next_ent()).size() > 0) {
            assert(entity.has_key("classname"));
            
            const std::string& classname = entity.get("classname");
            
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
            
            std::vector<uint8_t> extraLumpData;
            
            load_lump(file, static_cast<LumpType>(lumpID), extraLumpData);
            
            m_extraLumps[lumpID] = std::move(extraLumpData);
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
        
        GameLump* pGameLumps = &gameLumpHeader.firstGameLump;
        
        for (int i=0; i<lumpCount; i++) {
            GameLump& gameLump = pGameLumps[i];
            
            switch (gameLump.id) {
                case GAMELUMP_STATIC_PROPS:
                // case GAMELUMP_STATIC_PROPS: {
                    // std::vector<>
                    
                    // break;
                // }
                default: {
                    std::vector<uint8_t> extraGameLumpData;
                    
                    load_single_gamelump(
                        file,
                        gameLump,
                        extraGameLumpData
                    );
                    
                    m_extraGameLumps[gameLump.id]
                        = std::move(extraGameLumpData);
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

    const std::vector<DModel>& BSP::get_models(void) const {
        return m_models;
    }

    const std::vector<DPlane>& BSP::get_planes(void) const {
        return m_planes;
    }

    const std::vector<Vec3<float>>& BSP::get_vertices(void) const {
        return m_vertices;
    }

    const std::vector<DEdge>& BSP::get_edges(void) const {
        return m_edges;
    }

    const std::vector<int32_t>& BSP::get_surfedges(void) const {
        return m_surfEdges;
    }

    std::vector<DFace>& BSP::get_dfaces(void) {
        return m_dFaces;
    }

    const std::vector<DFace>& BSP::get_dfaces(void) const {
        return m_dFaces;
    }

    std::vector<RGBExp32>& BSP::get_lightsamples(void) {
        return m_lightSamples;
    }

    const std::vector<RGBExp32>& BSP::get_lightsamples(void) const {
        return m_lightSamples;
    }

    const std::vector<TexInfo>& BSP::get_texinfos(void) const {
        return m_texInfos;
    }

    const std::vector<DTexData>& BSP::get_texdatas(void) const {
        return m_texDatas;
    }

    std::vector<Face>& BSP::get_faces(void) {
        return m_faces;
    }

    const std::vector<Face>& BSP::get_faces(void) const {
        return m_faces;
    }

    const std::vector<DLeaf>& BSP::get_leaves(void) const {
        return m_leaves;
    }

    const std::vector<DWorldLight>& BSP::get_worldlights(void) const {
        return m_worldLights;
    }

    const std::vector<Light>& BSP::get_lights(void) const {
        return m_lights;
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
            file, LUMP_MODELS, m_models,
            offsets, sizes
        );
        
        save_lump(
            file, LUMP_PLANES, m_planes,
            offsets, sizes
        );
        
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

        if (is_fullbright()) {
            std::vector<DFace> dFaces(m_dFaces);

            // Make sure the light offset is -1 for all DFace structures if 
            // we are in fullbright mode.
            for (DFace& dFace : dFaces) {
                dFace.lightOffset = -1;
            }

            save_lump(
                file, LUMP_FACES, dFaces,
                offsets, sizes
            );

            save_lump(
                file, LUMP_FACES_HDR, dFaces,
                offsets, sizes
            );
        }
        else {
            //for (DFace& dFace : m_dFaces) {
            //    std::cout << dFace.texInfo << std::endl;
            //}

            save_lump(
                file, LUMP_FACES, m_dFaces,
                offsets, sizes
            );

            save_lump(
                file, LUMP_FACES_HDR, m_dFaces,
                offsets, sizes
            );
        }

        if (!is_fullbright()) {
            save_lump(
                file, LUMP_LIGHTING_HDR, m_lightSamples,
                offsets, sizes
            );
        }

        save_lump(
            file, LUMP_TEXINFO, m_texInfos,
            offsets, sizes
        );

        save_lump(
            file, LUMP_TEXDATA, m_texDatas,
            offsets, sizes
        );

        //save_faces(file, offsets, sizes);
        
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
                for (RGBExp32& sample : ambient.cube.color) {
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
            
            lump.fileOffset = static_cast<int32_t>(offsets[i]);
            lump.fileLen = static_cast<int32_t>(sizes[i]);
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
    
    //void BSP::save_faces(
    //        std::ofstream& file,
    //        std::unordered_map<int, std::ofstream::off_type>& offsets,
    //        std::unordered_map<int, size_t>& sizes
    //        ) {
    //        
    //    std::vector<LightSample> lightSamples;
    //    std::vector<DTexData> dTexDatas;
    //    std::vector<TexInfo> texInfos;
    //    std::vector<DFace> dFaces;
    //    
    //    for (Face& face : m_faces) {
    //        if (!is_fullbright()) {
    //            lightSamples.push_back(face.get_average_lighting());
    //            
    //            face.set_lightlump_offset(
    //                static_cast<int32_t>(lightSamples.size())
    //            );
    //            
    //            for (LightSample& lightSample : face.get_lightsamples()) {
    //                lightSamples.push_back(lightSample);
    //            }
    //        }
    //        
    //        face.set_texdata_index(static_cast<int32_t>(dTexDatas.size()));
    //        dTexDatas.push_back(face.get_texdata());
    //        
    //        face.set_texinfo_index(static_cast<int32_t>(texInfos.size()));
    //        texInfos.push_back(face.get_texinfo());
    //        
    //        dFaces.push_back(face.get_data());
    //    }
    //    
    //    if (!is_fullbright()) {
    //        save_lump(
    //            file, LUMP_LIGHTING_HDR, m_lightSamples,
    //            offsets, sizes
    //        );
    //    }
    //    else {
    //        offsets[LUMP_LIGHTING_HDR] = 0;
    //        sizes[LUMP_LIGHTING_HDR] = 0;
    //    }
    //    
    //    save_lump(
    //        file, LUMP_TEXINFO, m_texInfos,
    //        offsets, sizes
    //    );
    //    
    //    save_lump(
    //        file, LUMP_TEXDATA, m_texDatas,
    //        offsets, sizes
    //    );
    //    
    //    save_lump(
    //        file, LUMP_FACES, m_dFaces,
    //        offsets, sizes
    //    );
    //    
    //    save_lump(
    //        file, LUMP_FACES_HDR, m_dFaces,
    //        offsets, sizes
    //    );
    //}
    
    void BSP::save_lights(
            std::ofstream& file,
            std::unordered_map<int, std::ofstream::off_type>& offsets,
            std::unordered_map<int, size_t>& sizes
            ) {
        
        if (m_worldLights.size() == 0) {
            build_worldlights();
        }

        save_lump(
            file, LUMP_WORLDLIGHTS_HDR, m_worldLights,
            offsets, sizes
        );
    }
    
    void BSP::save_extras(
            std::ofstream& file, 
            std::unordered_map<int, std::ofstream::off_type>& offsets,
            std::unordered_map<int, size_t>& sizes
            ) {
            
        using Pair = std::pair<int, std::vector<uint8_t>>;
        
        for (const Pair& pair : m_extraLumps) {
            LumpType lumpID = static_cast<LumpType>(pair.first);
            const std::vector<uint8_t>& lumpData = pair.second;
            
            assert(m_extraLumps.find(lumpID) != m_extraLumps.end());
            
            save_lump(
                file, lumpID, lumpData,
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
            
        using Pair = std::pair<int32_t, std::vector<uint8_t>>;
        
        for (const Pair& pair : m_extraGameLumps) {
            int32_t gameLumpID = pair.first;
            const std::vector<uint8_t>& gameLumpData = pair.second;
            
            save_single_gamelump(file, gameLumpID, gameLumpData);
        }
        
        int32_t gameLumpCount = static_cast<int32_t>(m_gameLumps.size());
        
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
        gameLump.fileOffset = static_cast<int32_t>(offset);
        gameLump.fileLen = static_cast<int32_t>(size);
        
        file.write(reinterpret_cast<const char*>(src.data()), size);
    }
    
    const std::unordered_map<int, std::vector<uint8_t>>&
    BSP::get_extras(void) const{
        return m_extraLumps;
    }

    void BSP::build_worldlights(void) {
        m_worldLights.clear();

        for (const Light& light : get_lights()) {
            m_worldLights.push_back(light.to_worldlight());
        }
    }
    
    
    /**********************
     * EntityParser Class *
     **********************/
     
    EntityParser::EntityParser(const std::string& entData) :
        m_index(0),
        m_entData(entData) {}
        
    Entity EntityParser::next_ent(void) {
        int entStart = -1;
        std::string entStr = "";
        
        while (m_index < static_cast<int>(m_entData.size())) {
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
                || (m_index >= static_cast<int>(m_entData.size())
                    && entStart == -1)
        );
        
        Entity nextEnt;
        
        if (entStr == "") {
            return nextEnt;
        }
        
        std::string key = "";
        int fieldStart = -1;
        
        for (size_t i=0; i<entStr.size(); i++) {
            char c = entStr[i];
            
            if (c == '"') {
                if (fieldStart == -1) {
                    fieldStart = static_cast<int>(i);
                }
                else {
                    size_t count = i - fieldStart - 1;
                    std::string field = entStr.substr(
                        fieldStart + 1,
                        count
                    );
                    
                    fieldStart = -1;
                    
                    if (key == "") {
                        key = field;
                    }
                    else {
                        assert(!nextEnt.has_key(key));
                        nextEnt.set(key, field);
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
     
    size_t Face::s_faceCount = 0;
    
    Face::Face(BSP& bsp, DFace& faceData) :
            m_bsp(bsp),
            m_faceData(faceData),
            m_planeData(bsp.m_planes[faceData.planeNum]),
            m_texInfo(bsp.m_texInfos[faceData.texInfo]),
            m_texData(bsp.m_texDatas[m_texInfo.texData]),
            id(s_faceCount++) {
            
        const std::vector<Vec3<float>>& vertices = bsp.get_vertices();
        const std::vector<DEdge>& dEdges = bsp.get_edges();
        const std::vector<int32_t>& surfEdges = bsp.get_surfedges();

        //std::cout << "Face " << id << " has lightmap size "
        //    << get_lightmap_width() << " x "
        //    << get_lightmap_height() << std::endl;
        
        load_edges(faceData, vertices, dEdges, surfEdges);
        //load_lightsamples(bsp.get_lightsamples());
        
        /* For coordinate transformation from s/t to x/y/z */
        //precalculate_st_xyz_matrix();
        make_st_xyz_matrix(m_Ainv);
    }
    
    void Face::load_edges(
            const DFace& faceData,
            const std::vector<Vec3<float>>& vertices,
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
    
    //void Face::load_lightsamples(
    //        std::vector<LightSample>& lightSamples
    //        ) {
    //        
    //    if (m_bsp.is_fullbright()) {
    //        return;
    //    }

    //    assert(m_bsp.get_lightsamples().size() > 0);
    //    assert(lightSamples.size() > 0);

    //    m_lightSamplesBegin = lightSamples.begin()
    //        + m_faceData.lightOffset / sizeof(LightSample);

    //    m_lightSamplesEnd = m_lightSamplesBegin
    //        + get_lightmap_width() * get_lightmap_height();
    //}

    void Face::make_st_xyz_matrix(gmtl::Matrix<double, 3, 3>& Ainv) const {
        double sx = m_texInfo.lightmapVecs[0][0];
        double sy = m_texInfo.lightmapVecs[0][1];
        double sz = m_texInfo.lightmapVecs[0][2];
        
        double tx = m_texInfo.lightmapVecs[1][0];
        double ty = m_texInfo.lightmapVecs[1][1];
        double tz = m_texInfo.lightmapVecs[1][2];
        
        double nx = m_planeData.normal.x;
        double ny = m_planeData.normal.y;
        double nz = m_planeData.normal.z;
        
        gmtl::Matrix<double, 3, 3> A;

        A.set(
            sx, sy, sz,
            tx, ty, tz,
            nx, ny, nz
        );
        
        gmtl::invert(Ainv, A);
    }
    
    const DFace& Face::get_data(void) const {
        return m_faceData;
    }
    
    const DPlane& Face::get_planedata(void) const {
        return m_planeData;
    }

    const TexInfo& Face::get_texinfo(void) const {
        return m_texInfo;
    }

    const DTexData& Face::get_texdata(void) const {
        return m_texData;
    }

    const gmtl::Matrix<double, 3, 3>& Face::get_st_xyz_matrix(void) const {
        return m_Ainv;
    }

    void Face::set_texinfo_index(int16_t index) {
        m_faceData.texInfo = index;
    }
    
    void Face::set_texdata_index(int32_t index) {
        m_texInfo.texData = index;
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
    
    size_t Face::get_lightmap_size(void) const {
        return get_lightmap_width() * get_lightmap_height();
    }

    size_t Face::get_lightmap_offset(void) const {
        return m_faceData.lightOffset / sizeof(RGBExp32);
    }

    void Face::set_lightmap_offset(size_t offset) {
        m_faceData.lightOffset = static_cast<int32_t>(
            offset * sizeof(RGBExp32)
        );
    }

    FaceLightSampleProxy Face::get_lightsamples(void) {
        std::vector<RGBExp32>& lightSamples = m_bsp.get_lightsamples();

        assert(lightSamples.size() > get_lightmap_offset());

        assert(
            lightSamples.size()
                >= get_lightmap_offset() + get_lightmap_size()
        );
        
        FaceLightSampleProxy::Iter begin
            = lightSamples.begin() + get_lightmap_offset();
        
        FaceLightSampleProxy::Iter end
            = begin + get_lightmap_size();

        return FaceLightSampleProxy(begin, end);
    }
    
    RGBExp32 Face::get_average_lighting(void) const {
        return m_bsp.get_lightsamples().at(get_lightmap_offset() - 1);
    }
    
    void Face::set_average_lighting(const RGBExp32& sample) {
        m_bsp.get_lightsamples()[get_lightmap_offset() - 1] = sample;
    }
    
    Vec3<float> Face::xyz_from_lightmap_st(float s, float t) const {
        double sOffset = m_texInfo.lightmapVecs[0][3];
        double tOffset = m_texInfo.lightmapVecs[1][3];
        
        double sMin = static_cast<double>(
            m_faceData.lightmapTextureMinsInLuxels[0]
        );
        double tMin = static_cast<double>(
            m_faceData.lightmapTextureMinsInLuxels[1]
        );
        
        gmtl::Matrix<double, 3, 1> B;
        gmtl::Matrix<double, 3, 1> result;
        
        B[0][0] = s - sOffset + sMin;
        B[1][0] = t - tOffset + tMin;
        B[2][0] = m_planeData.dist;
        B.mState = gmtl::Matrix<double, 3, 1>::FULL;
        
        gmtl::mult(result, m_Ainv, B);
        
        return Vec3<float> {
            static_cast<float>(result[0][0]),
            static_cast<float>(result[1][0]),
            static_cast<float>(result[2][0]),
        };
    }
    

    /******************************
    * FaceLightSampleProxy Class *
    ******************************/

    FaceLightSampleProxy::FaceLightSampleProxy(
            FaceLightSampleProxy::Iter& begin,
            FaceLightSampleProxy::Iter& end
            ) :
            m_begin(begin),
            m_end(end) {}

    RGBExp32& FaceLightSampleProxy::operator[](size_t i) {
        return *(m_begin + i);
    }

    const RGBExp32& FaceLightSampleProxy::operator[](size_t i) const {
        return *(m_begin + i);
    }

    FaceLightSampleProxy::Iter FaceLightSampleProxy::begin(void) {
        return m_begin;
    }

    FaceLightSampleProxy::Iter FaceLightSampleProxy::end(void) {
        return m_end;
    }
    

    /***************
     * Light Class *
     ***************/
     
    template<typename T>
    static inline T convert_str(const std::string& str) {
        T result;
        
        std::stringstream converter;
        converter << str;
        converter >> result;
        
        return result;
    }
    
    static Vec3<float> vec3_from_str(const std::string& str) {
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
        
        return Vec3<float> {x, y, z};
    }
    
    /* Gamma-correction */
    static inline double linear_from_encoded(double encoded) {
        return pow(encoded / 255.0, GAMMA) * 255.0;
    }
    
    static inline double encoded_from_linear(double linear) {
        return pow(linear / 255.0, INV_GAMMA) * 255.0;
    }
    
    Light::Light(const Entity& entity) :
            m_coords(vec3_from_str(entity.get("origin"))) {
            
        /* Parse color */
        std::stringstream stream(entity.get("_light"));
        
        std::string s;
        
        std::getline(stream, s, ' ');
        r = linear_from_encoded(convert_str<double>(s));
        
        std::getline(stream, s, ' ');
        g = linear_from_encoded(convert_str<double>(s));
        
        std::getline(stream, s, ' ');
        b = linear_from_encoded(convert_str<double>(s));
        
        std::getline(stream, s, ' ');
        double brightness = convert_str<double>(s) / 255.0;
        
        // Note that this means we are scaling brightness linearly, rather 
        // than perceptually.
        r *= brightness;
        g *= brightness;
        b *= brightness;
        
        /* Parse attenuation */
        c = convert_str<double>(entity.get("_constant_attn", "0"));
        l = convert_str<double>(entity.get("_linear_attn", "0"));
        q = convert_str<double>(entity.get("_quadratic_attn", "0"));
        
        /* Scale color intensity to 100-unit inverse attenuation */
        // I don't know why we need to do this.
        // Honestly, it doesn't really make all that much sense to me.
        // But if we don't do it, everything looks way too dark.
        double scale = attenuate(100.0);
        
        r *= scale;
        g *= scale;
        b *= scale;
    }
    
    const Vec3<float>& Light::get_coords(void) const {
        return m_coords;
    }
    
    DWorldLight Light::to_worldlight(void) const {
        return DWorldLight {
            get_coords(),
            Vec3<float> {
                static_cast<float>(r / 255.0),
                static_cast<float>(g / 255.0),
                static_cast<float>(b / 255.0),
            },
            Vec3<float> {1.0, 0.0, 0.0},
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
    

    /****************
     * Entity Class *
     ****************/
    
    Entity::Entity() {}
    
    const std::string& Entity::get(const std::string& key) const {
        return m_data.at(key);
    }
    
    const std::string& Entity::get(
            const std::string& key,
            const std::string& defaultVal
            ) const {
            
        std::unordered_map<std::string, std::string>::const_iterator
            pValue = m_data.find(key);
            
        if (pValue != m_data.end()) {
            return pValue->second;
        }
        else {
            return defaultVal;
        }
    }
    
    bool Entity::has_key(const std::string& key) const {
        return m_data.find(key) != m_data.end();
    }
    
    void Entity::set(const std::string& key, const std::string& value) {
        m_data[key] = value;
    }
    
    size_t Entity::size(void) const {
        return m_data.size();
    }
}
