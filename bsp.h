#ifndef __BSP_H_
#define __BSP_H_

#include <cstdint>
#include <cstdlib>

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <fstream>

namespace BSP {
    const uint32_t IDBSPHEADER = 'V' | ('B' << 8) | ('S' << 16) | ('P' << 24);
    const size_t HEADER_LUMPS = 64;
    const size_t MAX_MAP_PLANES = 65536;
    const size_t MAX_MAP_EDGES = 256000;
    const size_t MAX_MAP_SURFEDGES = 512000;
    
    enum LumpType {
        LUMP_ENTITIES,
        LUMP_PLANES,
        LUMP_TEXDATA,
        LUMP_VERTICES,
        LUMP_VISIBILITY,
        LUMP_NODES,
        LUMP_TEXINFO,
        LUMP_FACES,
        LUMP_LIGHTING,
        LUMP_OCCLUSION,
        LUMP_LEAFS,
        LUMP_FACEIDS,
        LUMP_EDGES,
        LUMP_SURFEDGES,
        LUMP_MODELS,
        LUMP_WORLDLIGHTS,
        LUMP_LEAFFACES,
        LUMP_LEAFBRUSHES,
        LUMP_BRUSHES,
        LUMP_BRUSHSIDES,
        LUMP_AREAS,
        LUMP_AREAPORTALS,
        LUMP_PORTALS = 22,
        LUMP_UNUSED0 = 22,
        LUMP_PROPCOLLISION = 22,
        LUMP_CLUSTERS = 23,
        LUMP_UNUSED1 = 23,
        LUMP_PROPHULLS = 23,
        LUMP_PORTALVERTS = 24,
        LUMP_UNUSED2 = 24,
        LUMP_PROPHULLVERTS = 24,
        LUMP_CLUSTERPORTALS = 25,
        LUMP_UNUSED3 = 25,
        LUMP_PROPTRIS = 25,
        LUMP_DISPINFO,
        LUMP_ORIGINALFACES,
        LUMP_PHYSDISP,
        LUMP_PHYSCOLLIDE,
        LUMP_VERTNORMALS,
        LUMP_VERTNORMALINDICES,
        LUMP_DISP_LIGHTMAP_ALPHAS,
        LUMP_DISP_VERTS,
        LUMP_DISP_LIGHTMAP_SAMPLE_POSITIONS,
        LUMP_GAME_LUMP,
        LUMP_LEAFWATERDATA,
        LUMP_PRIMITIVES,
        LUMP_PRIMVERTS,
        LUMP_PRIMINDICES,
        LUMP_PAKFILE,
        LUMP_CLIPPORTALVERTS,
        LUMP_CUBEMAPS,
        LUMP_TEXDATA_STRING_DATA,
        LUMP_TEXDATA_STRING_TABLE,
        LUMP_OVERLAYS,
        LUMP_LEAFMINDISTTOWATER,
        LUMP_FACE_MACRO_TEXTURE_INFO,
        LUMP_DISP_TRIS,
        LUMP_PHYSCOLLIDESURFACE = 49,
        LUMP_PROP_BLOB = 49,
        LUMP_WATEROVERLAYS,
        LUMP_LIGHTMAPPAGES = 51,
        LUMP_LEAF_AMBIENT_INDEX_HDR = 51,
        LUMP_LIGHTMAPPAGEINFOS = 52,
        LUMP_LEAF_AMBIENT_INDEX = 52,
        LUMP_LIGHTING_HDR,
        LUMP_WORLDLIGHTS_HDR,
        LUMP_LEAF_AMBIENT_LIGHTING_HDR,
        LUMP_LEAF_AMBIENT_LIGHTING,
        LUMP_XZIPPAKFILE,
        LUMP_FACES_HDR,
        LUMP_MAP_FLAGS,
        LUMP_OVERLAY_FADES,
        LUMP_OVERLAY_SYSTEM_LEVELS,
        LUMP_PHYSLEVEL,
        LUMP_DISP_MULTIBLEND,
    };
    
    static_assert(LUMP_ENTITIES == 0, "Lump miscount!");
    static_assert(LUMP_DISP_MULTIBLEND == 63, "Lump miscount!");
    
    struct Lump {
        int32_t fileOffset;
        int32_t fileLen;
        int32_t version;
        uint8_t fourCC[4];
    };
    
    struct Header {
        int32_t ident;
        int32_t version;
        Lump lumps[HEADER_LUMPS];
        int32_t mapRevision;
    };
    
    struct Vec3 {
        float x;
        float y;
        float z;
    };
    
    struct DPlane {
        Vec3 normal;
        float dist;
        int32_t type;
    };
    
    struct DEdge {
        uint16_t vertex1;
        uint16_t vertex2;
    };
    
    struct DFace {
        uint16_t planeNum;
        uint8_t side;
        uint8_t onNode;
        int32_t firstEdge;
        int16_t numEdges;
        int16_t texInfo;
        int16_t dispInfo;
        int16_t surfaceFogVolumeID;
        uint8_t styles[4];
        int32_t lightOffset;
        float area;
        int32_t lightmapTextureMinsInLuxels[2];
        int32_t lightmapTexturesSizeInLuxels[2];
        int32_t origFace;
        uint16_t numPrims;
        uint16_t firstPrimID;
        uint32_t smoothingGroups;
    };
    
    struct LightSample {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        int8_t exp;
    };
    
    struct TexInfo {
        float textureVecs[2][4];
        float lightmapVecs[2][4];
        int32_t flags;
        int32_t texdata;
    };
    
    struct Edge {
        Vec3 vertex1;
        Vec3 vertex2;
    };
    
    class BSP;
    
    class Face {
        private:
            DFace m_faceData;
            DPlane m_planeData;
            
            std::vector<Edge> m_edges;
            
        public:
            Face(const BSP& bsp, const DFace& faceData);
            
            const std::vector<Edge>& get_edges(void) const;
    };
    
    class BSP {
        friend Face::Face(const BSP&, const DFace&);
        
        private:
            Header m_header;
            
            std::vector<Vec3> m_vertices;
            std::vector<DEdge> m_edges;
            std::vector<int32_t> m_surfEdges;
            std::vector<DPlane> m_planes;
            std::vector<Face> m_faces;
            std::vector<LightSample> m_lightSamples;
            std::vector<LightSample> m_hdrLightSamples;
            
            std::unordered_map<int, std::vector<uint8_t>> m_extra;
            
            std::unordered_set<int> m_loadedLumps;
            
            void init(std::ifstream& file);
            
            template<typename T>
            void load_lump(
                std::ifstream& file,
                const LumpType lumpID,
                std::vector<T>& dest
            );
            
            void load_extras(std::ifstream& file);
            
        public:
            BSP();
            BSP(const std::string& filename);
            BSP(std::ifstream& file);
            
            const std::vector<Face>& get_faces(void) const;
            
            std::vector<LightSample>& get_lightsamples(void);
            std::vector<LightSample>& get_hdr_lightsamples(void);
            
            void print_lump_offsets(void);
    };
    
    class InvalidBSP : public std::runtime_error {
        public:
            InvalidBSP(const std::string& what) : std::runtime_error(what) {}
    };
}

#endif
