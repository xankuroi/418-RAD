#ifndef __RAYTRACER_H_
#define __RAYTRACER_H_

#include <vector>

#include "cuda_runtime.h"
#include "bsp.h"

namespace RayTracer {
    struct Triangle {
        float3 vertices[3];
    };

    enum KDNodeType {NODETYPE_NODE, NODETYPE_LEAF};
    enum Axis {AXIS_X, AXIS_Y, AXIS_Z};

    struct KDNode {
        KDNodeType type;
        Axis axis;
        float pos;

        float3 tmin;
        float3 tmax;

        size_t* triangleIDs;
        size_t numTris;

        KDNode* children;
    };

    //class KDTree {
    //    private:
    //        KDNode* m_root;
    //        
    //    public:
    //        KDTree();
    //};

    /**
     * Ray-trace acceleration structure whose data resides solely in device 
     * memory.
     */
    class CUDARayTracer {
        private:
            static const size_t MAX_LEAVES;

            Triangle* m_triangles;
            size_t* m_triangleIDs;
            size_t m_numTriangles;

            KDNode* m_pTreeRoot;

            float3 m_tmin;
            float3 m_tmax;

            __host__ void build_tree(void);
            __host__ void destroy_tree(void);
            
        public:
            CUDARayTracer();
            CUDARayTracer(const CUDARayTracer& other) = delete;
            ~CUDARayTracer();

            CUDARayTracer& operator=(const CUDARayTracer& other) = delete;

            __host__ void add_triangles(const std::vector<Triangle>& tris);

            //__device__ Triangle* get_triangles(void);
            //__device__ size_t* get_tri_ids(void);
            //__device__ Triangle& get_tri_indirect(size_t i);

            __device__ bool LOS_blocked(
                const float3& start, const float3& end
            );
    };
}

#endif
