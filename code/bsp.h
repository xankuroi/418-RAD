#ifndef __BSP_H_
#define __BSP_H_

#include <cstdint>
#include <cstdlib>

#include <string>

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>

#include <fstream>

namespace BSP {
    inline constexpr uint32_t make_id(
            uint32_t a, uint32_t b, uint32_t c, uint32_t d
            ) {
        return static_cast<uint32_t>(a | (b << 8) | (c << 16) | (d << 24));
    }
    
    const uint32_t IDBSPHEADER = make_id('V', 'B', 'S', 'P');
    const size_t HEADER_LUMPS = 64;
    const size_t MAX_MAP_PLANES = 65536;
    const size_t MAX_MAP_EDGES = 256000;
    const size_t MAX_MAP_SURFEDGES = 512000;
    
    /* Standard Gamma-correction constants */
    const double GAMMA = 2.2;
    const double INV_GAMMA = 1.0 / GAMMA;
    
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
        LUMP_LEAVES,
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
        int32_t lightmapTextureSizeInLuxels[2];
        int32_t origFace;
        uint16_t numPrims;
        uint16_t firstPrimID;
        uint32_t smoothingGroups;
    };
    
    enum LeafContents {
        CONTENTS_SOLID = 0x1,
        CONTENTS_WINDOW = 0x2,
        CONTENTS_AUX = 0x4,
        CONTENTS_GRATE = 0x8,
        CONTENTS_SLIME = 0x10,
        CONTENTS_WATER = 0x20,
        CONTENTS_BLOCKLOS = 0x40,
        CONTENTS_OPAQUE = 0x80,
        LAST_VISIBLE_CONTENTS = 0x80,
        ALL_VISIBLE_CONTENTS
            = (LAST_VISIBLE_CONTENTS | (LAST_VISIBLE_CONTENTS - 1)),
        CONTENTS_TESTFOGVOLUME = 0x100,
        CONTENTS_UNUSED = 0x200,
        CONTENTS_UNUSED6 = 0x400,
        CONTENTS_TEAM1 = 0x800,
        CONTENTS_TEAM2 = 0x1000,
        CONTENTS_IGNORE_NODRAW_OPAQUE = 0x2000,
        CONTENTS_MOVEABLE = 0x4000,
        CONTENTS_AREAPORTAL = 0x8000,
        CONTENTS_PLAYERCLIP = 0x10000,
        CONTENTS_MONSTERCLIP = 0x20000,
        CONTENTS_CURRENT_0 = 0x40000,
        CONTENTS_CURRENT_90 = 0x80000,
        CONTENTS_CURRENT_180 = 0x100000,
        CONTENTS_CURRENT_270 = 0x200000,
        CONTENTS_CURRENT_UP = 0x400000,
        CONTENTS_CURRENT_DOWN = 0x800000,
        CONTENTS_ORIGIN = 0x1000000,
        CONTENTS_MONSTER = 0x2000000,
        CONTENTS_DEBRIS = 0x4000000,
        CONTENTS_DETAIL = 0x8000000,
        CONTENTS_TRANSLUCENT = 0x10000000,
        CONTENTS_LADDER = 0x20000000,
        CONTENTS_HITBOX = 0x40000000,
    };
    
    struct DLeaf {
        uint32_t contents;
        int16_t cluster;
        int16_t area:9;
        int16_t flags:7;
        int16_t mins[3];
        int16_t maxs[3];
        uint16_t firstLeafFace;
        uint16_t numLeafFaces;
        uint16_t firstLeafBrush;
        uint16_t numLeafBrushes;
        int16_t leafWaterDataID;
    };
    
    enum EmitType {
        EMIT_SURFACE,
        EMIT_POINT,
        EMIT_SPOTLIGHT,
        EMIT_SKYLIGHT,
        EMIT_QUAKELIGHT,
        EMIT_SKYAMBIENT,
    };
    
    static_assert(EMIT_SURFACE == 0, "EmitType miscount!");
    static_assert(EMIT_SKYAMBIENT == 5, "EmitType miscount!");
    
    struct DWorldLight {
        Vec3 origin;
        Vec3 intensity;
        Vec3 normal;
        int32_t cluster;
        EmitType type;
        int32_t style;
        float stopdot;
        float stopdot2;
        float exponent;
        float radius;
        float constantAtten;
        float linearAtten;
        float quadraticAtten;
        int32_t flags;
        int32_t texinfo;
        int32_t owner;
    };
    
    struct LightSample {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        int8_t exp;
    };
    
    struct CompressedLightCube {
        LightSample color[6];
    };
    
    struct DLeafAmbientLighting {
        CompressedLightCube cube;
        uint8_t x;
        uint8_t y;
        uint8_t z;
        uint8_t unused;
    };
    
    struct DLeafAmbientIndex {
        uint16_t ambientSampleCount;
        uint16_t firstAmbientSample;
    };
    
    enum GameLumpID {
        GAMELUMP_STATIC_PROPS = make_id('s', 'p', 'r', 'p'),
    };
    
    struct GameLump {
        int32_t id;
        uint16_t flags;
        uint16_t version;
        int32_t fileOffset;
        int32_t fileLen;
    };
    
    struct GameLumpHeader {
        int32_t lumpCount;
        GameLump firstGameLump;
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
    
    class Light {
        private:
            Vec3 m_coords;
            
        public:
            double r;
            double g;
            double b;
            
            Light(const std::unordered_map<std::string, std::string>& entity);
            
            const Vec3& get_coords(void) const;
            DWorldLight to_worldlight(void) const;
    };
    
    class Face {
        private:
            DFace m_faceData;
            DPlane m_planeData;
            
            std::vector<Edge> m_edges;
            std::vector<LightSample> m_lightSamples;
            
            LightSample m_avgLightSample;
            
            void load_edges(
                const DFace& faceData,
                const std::vector<Vec3>& vertices,
                const std::vector<DEdge>& dEdges,
                const std::vector<int32_t>& surfEdges
            );
            void load_lightsamples(const std::vector<LightSample>& samples);
            
        public:
            Face(
                const BSP& bsp,
                const DFace& faceData,
                const std::vector<LightSample>& lightSamples
            );
            
            const DFace& get_data(void) const;
            const std::vector<Edge>& get_edges(void) const;
            
            const std::vector<uint8_t> get_styles(void) const;
            void set_styles(const std::vector<uint8_t>& styles);
            
            int32_t get_lightmap_width(void) const;
            int32_t get_lightmap_height(void) const;
            
            std::vector<LightSample>& get_lightsamples(void);
            
            LightSample get_average_lighting(void) const;
            void set_average_lighting(const LightSample& sample);
            
            void set_lightlump_offset(int32_t offset);
    };
    
    class BSP {
        friend Face::Face(
            const BSP&,
            const DFace&,
            const std::vector<LightSample>&
        );
        
        private:
            Header m_header;
            
            std::vector<Vec3> m_vertices;
            std::vector<DEdge> m_edges;
            std::vector<int32_t> m_surfEdges;
            std::vector<DPlane> m_planes;
            std::vector<Face> m_faces;
            std::vector<DLeaf> m_leaves;
            std::vector<DWorldLight> m_worldLights;
            
            std::string m_entData;
            std::vector<Light> m_lights;
            
            std::unordered_map<int, std::unique_ptr<std::vector<uint8_t>>> 
                m_extraLumps;
                
            std::unordered_map<int32_t, std::unique_ptr<std::vector<uint8_t>>> 
                m_extraGameLumps;
                
            std::unordered_set<int> m_loadedLumps;
            
            std::unordered_map<int32_t, GameLump> m_gameLumps;
            
            bool m_fullbright;
            
            void init(std::ifstream& file);
            
            template<typename Container>
            void load_lump(
                std::ifstream& file,
                const LumpType lumpID,
                Container& dest
            );
            
            void load_lights(const std::string& entData);
            void load_extras(std::ifstream& file);
            
            void load_gamelumps(std::ifstream& file);
            
            template<typename Container>
            void load_single_gamelump(
                std::ifstream& file,
                const GameLump& gameLump,
                Container& dest
            );
            
            template<typename Container>
            void save_lump(
                std::ofstream& file,
                const LumpType lumpID,
                const Container& src,
                std::unordered_map<int, std::ofstream::off_type>& offsets,
                std::unordered_map<int, size_t>& sizes,
                bool isExtraLump=false
            );
            
            void save_faces(
                std::ofstream& file,
                std::unordered_map<int, std::ofstream::off_type>& offsets,
                std::unordered_map<int, size_t>& sizes
            );
            
            void save_lights(
                std::ofstream& file,
                std::unordered_map<int, std::ofstream::off_type>& offsets,
                std::unordered_map<int, size_t>& sizes
            );
            
            void save_extras(
                std::ofstream& file, 
                std::unordered_map<int, std::ofstream::off_type>& offsets,
                std::unordered_map<int, size_t>& sizes
            );
            
            void save_gamelumps(
                std::ofstream& file,
                std::unordered_map<int, std::ofstream::off_type>& offsets,
                std::unordered_map<int, size_t>& sizes
            );
            
            template<typename Container>
            void save_single_gamelump(
                std::ofstream& file,
                int32_t gameLumpID,
                const Container& src
            );
            
        public:
            BSP();
            BSP(const std::string& filename);
            BSP(std::ifstream& file);
            
            int get_format_version(void) const;
            
            const Header& get_header(void) const;
            
            std::vector<Face>& get_faces(void);
            const std::vector<Light>& get_lights(void) const;
            
            const std::vector<DWorldLight>& get_worldlights(void) const;
            
            const std::string& get_entdata(void);
            
            const std::unordered_map<
                int, std::unique_ptr<std::vector<uint8_t>>
            >& get_extras(void) const;
            
            bool is_fullbright(void) const;
            void set_fullbright(bool fullbright);
            
            void write(const std::string& filename);
            void write(std::ofstream& file);
    };
    
    class InvalidBSP : public std::runtime_error {
        public:
            InvalidBSP(const std::string& what) : std::runtime_error(what) {}
    };
    
    class EntityParser {
        private:
            size_t m_index;
            std::string m_entData;
            
        public:
            EntityParser(const std::string& entData);
            
            std::unordered_map<std::string, std::string> next_ent(void);
    };
}

#endif
