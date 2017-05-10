#include <iostream>
#include <algorithm>
#include <cassert>

#include "raytracer.h"
#include "cudautils.h"


namespace RayTracer {
    /* Implements the M-T ray-triangle intersection algorithm. */
    static __device__ bool intersects(
            const float3& vertex1,
            const float3& vertex2,
            const float3& vertex3,
            const float3& startPos, const float3& endPos
            ) {

        const float EPSILON = 1e-6;

        float3 diff = endPos - startPos;
        float dist = len(diff);
        float3 dir = diff / dist;

        float3 edge1 = vertex2 - vertex1;
        float3 edge2 = vertex3 - vertex1;

        float3 pVec = cross(dir, edge2);

        float det = dot(edge1, pVec);

        if (det < EPSILON) {
            return false;
        }

        float3 tVec = startPos - vertex1;

        float u = dot(tVec, pVec);
        if (u < 0.0 || u > det) {
            return false;
        }

        float3 qVec = cross(tVec, edge1);

        float v = dot(dir, qVec);

        if (v < 0.0 || u + v > det) {
            return false;
        }

        float t = dot(edge2, qVec) / det;

        return (0.0 < t && t < dist);
    }

    static const size_t MAX_DEPTH = 10;

    __global__ void split_nodes(
            Triangle* triangles,
            KDNode* nodes,
            size_t depth
            ) {
            
        //printf("Split nodes (depth %u)...\n", static_cast<unsigned int>(depth));

        KDNode& node = nodes[threadIdx.x];

        if (depth <= 1 || node.numTris <= 3) {
            //printf(
            //    "Found a leaf! (%u tris)\n",
            //    static_cast<unsigned int>(node.numTris)
            //);
            node.type = NODETYPE_LEAF;
            return;
        }

        float3 nodeSize = node.tmax - node.tmin;
        float3 rightTMin;

        //printf(
        //    "(%u) Node size: (%f, %f, %f)\n",
        //    static_cast<unsigned int>(depth),
        //    nodeSize.x, nodeSize.y, nodeSize.z
        //);

        if (nodeSize.x > nodeSize.y && nodeSize.x > nodeSize.z
                || nodeSize.x == nodeSize.y && nodeSize.x > nodeSize.z
                || nodeSize.x > nodeSize.y && nodeSize.x == nodeSize.z
                ) {

            /* Split along the x-axis. */

            nodeSize.x *= 0.5;
            node.axis = AXIS_X;
            node.pos = node.tmin.x + nodeSize.x;
            rightTMin = node.tmin + make_float3(nodeSize.x, 0.0, 0.0);

            //printf(
            //    "(%u) Split at x = %f\n",
            //    static_cast<unsigned int>(depth),
            //    node.pos
            //);
        }
        else if (nodeSize.y > nodeSize.x && nodeSize.y > nodeSize.z
                || nodeSize.y == nodeSize.x && nodeSize.y > nodeSize.z
                || nodeSize.y > nodeSize.x && nodeSize.y == nodeSize.z
                ) {

            /* Split along the y-axis. */

            nodeSize.y *= 0.5;
            node.axis = AXIS_Y;
            node.pos = node.tmin.y + nodeSize.y;
            rightTMin = node.tmin + make_float3(0.0, nodeSize.y, 0.0);

            //printf(
            //    "(%u) Split at y = %f\n",
            //    static_cast<unsigned int>(depth),
            //    node.pos
            //);
        }
        else {
            /* Split along the z-axis. */
            nodeSize.z *= 0.5;
            node.axis = AXIS_Z;
            node.pos = node.tmin.z + nodeSize.z;
            rightTMin = node.tmin + make_float3(0.0, 0.0, nodeSize.z);
            //printf(
            //    "(%u) Split at z = %f\n",
            //    static_cast<unsigned int>(depth),
            //    node.pos
            //);
        }

        size_t* leftTriIDs;
        CUDA_CHECK_ERROR_DEVICE(
            cudaMalloc(&leftTriIDs, sizeof(size_t) * node.numTris)
        );
        
        size_t* rightTriIDs;
        CUDA_CHECK_ERROR_DEVICE(
            cudaMalloc(&rightTriIDs, sizeof(size_t) * node.numTris)
        );

        size_t numLeft = 0;
        size_t numRight = 0;

        for (size_t i=0; i<node.numTris; i++) {
            size_t triangleID = node.triangleIDs[i];
            Triangle& tri = triangles[triangleID];

            bool onLeft = false;
            bool onRight = false;

            for (int vertex=0; vertex<3; vertex++) {
                switch (node.axis) {
                    case AXIS_X:
                        if (tri.vertices[vertex].x <= node.pos) {
                            onLeft = true;
                        }
                        if (tri.vertices[vertex].x >= node.pos) {
                            onRight = true;
                        }
                        
                        break;

                    case AXIS_Y:
                        if (tri.vertices[vertex].y <= node.pos) {
                            onLeft = true;
                        }
                        if (tri.vertices[vertex].y >= node.pos) {
                            onRight = true;
                        }

                        break;

                    case AXIS_Z:
                        if (tri.vertices[vertex].z <= node.pos) {
                            onLeft = true;
                        }
                        if (tri.vertices[vertex].z >= node.pos) {
                            onRight = true;
                        }

                        break;
                }

                if (onLeft && onRight) {
                    break;
                }
            }

            if (onLeft) {
                leftTriIDs[numLeft++] = triangleID;
            }

            if (onRight) {
                rightTriIDs[numRight++] = triangleID;
            }
        }

        //printf("cudaFree %p\n", node.triangleIDs);

        //cudaFree(node.triangleIDs);

        CUDA_CHECK_ERROR_DEVICE(
            cudaMalloc(&node.children, sizeof(KDNode) * 2)
        );

        node.children[0].type = NODETYPE_NODE;
        node.children[0].tmin = node.tmin;
        node.children[0].tmax = node.tmin + nodeSize;
        node.children[0].triangleIDs = leftTriIDs;
        node.children[0].numTris = numLeft;

        node.children[1].type = NODETYPE_NODE;
        node.children[1].tmin = rightTMin;
        node.children[1].tmax = node.tmax;
        node.children[1].triangleIDs = rightTriIDs;
        node.children[1].numTris = numRight;

        KERNEL_LAUNCH_DEVICE(
            split_nodes, 1, 2,
            triangles, node.children, depth - 1
        );
    }

    __global__ void cleanup_nodes(KDNode* nodes) {
        KDNode& node = nodes[threadIdx.x];

        CUDA_CHECK_ERROR_DEVICE(cudaFree(node.triangleIDs));

        if (node.type == NODETYPE_NODE) {
            KDNode* children = node.children;

            if (threadIdx.x == 0) {
                CUDA_CHECK_ERROR_DEVICE(cudaFree(nodes));
            }
            
            KERNEL_LAUNCH_DEVICE(
                cleanup_nodes, 1, 2,
                children
            );
        }
    }


    /***********************
     * CUDARayTracer Class *
     ***********************/

    const size_t CUDARayTracer::MAX_LEAVES = 1024;

    CUDARayTracer::CUDARayTracer() :
            m_triangles(nullptr),
            m_triangleIDs(nullptr),
            m_numTriangles(0),
            m_pTreeRoot(nullptr),
            m_tmin(make_float3(0.0, 0.0, 0.0)),
            m_tmax(make_float3(0.0, 0.0, 0.0)) {}

    CUDARayTracer::~CUDARayTracer() {
        if (m_triangles != nullptr) {
            cudaFree(m_triangles);
        }
        
        if (m_pTreeRoot != nullptr) {
            destroy_tree();
        }
    }

    __host__ void CUDARayTracer::build_tree(void) {
        KDNode root;
        
        root.type = NODETYPE_NODE;

        root.tmin = m_tmin;
        root.tmax = m_tmax;

        root.triangleIDs = m_triangleIDs;
        root.numTris = m_numTriangles;

        CUDA_CHECK_ERROR(cudaMalloc(&m_pTreeRoot, sizeof(KDNode)));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                m_pTreeRoot, &root, sizeof(KDNode),
                cudaMemcpyHostToDevice
            )
        );

        Triangle* deviceTriangles;
        size_t deviceTrianglesSize = sizeof(Triangle) * m_numTriangles;

        CUDA_CHECK_ERROR(cudaMalloc(&deviceTriangles, deviceTrianglesSize));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                deviceTriangles, m_triangles, deviceTrianglesSize,
                cudaMemcpyHostToDevice
            )
        );

        KERNEL_LAUNCH(
            split_nodes, 1, 1,
            deviceTriangles, m_pTreeRoot, MAX_DEPTH
        );

        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        CUDA_CHECK_ERROR(cudaFree(deviceTriangles));
    }

    __host__ void CUDARayTracer::destroy_tree(void) {
        KDNode root;

        CUDA_CHECK_ERROR(
            cudaMemcpy(
                &root, m_pTreeRoot, sizeof(KDNode),
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK_ERROR(cudaFree(root.triangleIDs));

        if (root.type == NODETYPE_NODE) {
            KDNode* children = root.children;

            CUDA_CHECK_ERROR(cudaFree(m_pTreeRoot));

            KERNEL_LAUNCH(
                cleanup_nodes, 1, 2,
                children
            );
            CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        }
    }

    __host__ void CUDARayTracer::add_triangles(
            const std::vector<Triangle>& tris
            ) {
            
        // This method should only ever be called exactly once.
        assert(m_triangles == nullptr);
        assert(m_numTriangles == 0);

        m_numTriangles = tris.size();

        size_t trianglesSize = sizeof(Triangle) * m_numTriangles;

        CUDA_CHECK_ERROR(cudaMalloc(&m_triangles, trianglesSize));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                m_triangles, tris.data(), trianglesSize,
                cudaMemcpyHostToDevice
            )
        );

        std::vector<size_t> triangleIDs;
        for (size_t i=0; i<m_numTriangles; i++) {
            triangleIDs.push_back(i);
        }

        size_t triangleIDsSize = sizeof(size_t) * m_numTriangles;

        CUDA_CHECK_ERROR(cudaMalloc(&m_triangleIDs, triangleIDsSize));
        CUDA_CHECK_ERROR(
            cudaMemcpy(
                m_triangleIDs, triangleIDs.data(),
                triangleIDsSize,
                cudaMemcpyHostToDevice
            )
        );

        //std::cout << "cudaMalloc: " << m_triangleIDs << std::endl;

        /* Figure out the bounding box of the KD Tree. */
        for (const Triangle& tri : tris) {
            for (int vertex=0; vertex<3; vertex++) {
                m_tmin.x = std::min(m_tmin.x, tri.vertices[vertex].x);
                m_tmin.y = std::min(m_tmin.y, tri.vertices[vertex].y);
                m_tmin.z = std::min(m_tmin.z, tri.vertices[vertex].z);

                m_tmax.x = std::max(m_tmax.x, tri.vertices[vertex].x);
                m_tmax.y = std::max(m_tmax.y, tri.vertices[vertex].y);
                m_tmax.z = std::max(m_tmax.z, tri.vertices[vertex].z);
            }
        }

        build_tree();
    }

    //__device__ Triangle* CUDARayTracer::get_triangles(void) {
    //    return m_triangles;
    //}

    //__device__ size_t* CUDARayTracer::get_tri_ids(void) {
    //    return m_triangleIDs;
    //}

    //__device__ Triangle& CUDARayTracer::get_tri_indirect(size_t i) {
    //    return m_triangles[m_triangleIDs[i]];
    //}

    __device__ bool CUDARayTracer::LOS_blocked(
            const float3& startPos, const float3& endPos
            ) {

        const float EPSILON = 1e-6;

        float3 dir = normalized(endPos - startPos);
        float3 invDir = make_float3(
            1.0 / (dir.x + ((dir.x < 0) ? -EPSILON : EPSILON)),
            1.0 / (dir.y + ((dir.y < 0) ? -EPSILON : EPSILON)),
            1.0 / (dir.z + ((dir.z < 0) ? -EPSILON : EPSILON))
        );
        
        struct StackEntry {
            KDNode* pNode;
            float3 start;
            float3 end;
        };

        StackEntry stack[1024];   // empty ascending stack
        size_t stackSize = 0;

        stack[stackSize++] = {
            m_pTreeRoot,
            startPos,
            endPos,
        };

        while (stackSize > 0) {
            if (stackSize >= 1024) {
                printf("ALERT: Stack size too big!!!\n");
                return false;
            }

            StackEntry& entry = stack[--stackSize];

            KDNode* pNode = entry.pNode;
            float3 start = entry.start;
            float3 end = entry.end;

            float len = dist(start, end);
            
            KDNode* children = pNode->children;

            float t;

            switch (pNode->type) {
                case NODETYPE_LEAF:
                    for (size_t ti=0; ti<pNode->numTris; ti++) {
                        Triangle& tri = m_triangles[pNode->triangleIDs[ti]];

                        // The M-T intersection algorithm uses CCW vertex 
                        // winding, but Source uses CW winding. So, we need to 
                        // pass the vertices in reverse order to get backface 
                        // culling to work correctly.
                        bool isLOSBlocked = intersects(
                            tri.vertices[2], tri.vertices[1], tri.vertices[0],
                            startPos, endPos
                        );

                        if (isLOSBlocked) {
                            return true;
                        }
                    }

                    break;

                case NODETYPE_NODE:
                    bool dirPositive;

                    switch (pNode->axis) {
                        case AXIS_X:
                            t = (pNode->pos - start.x) * invDir.x;
                            dirPositive = dir.x >= 0.0;
                            break;

                        case AXIS_Y:
                            t = (pNode->pos - start.y) * invDir.y;
                            dirPositive = dir.y >= 0.0;
                            break;

                        case AXIS_Z:
                            t = (pNode->pos - start.z) * invDir.z;
                            dirPositive = dir.z >= 0.0;
                            break;
                    }

                    if (t < 0.0) {
                        // Plane is "behind" the line start.
                        // Recurse on the right side if dir is positive.
                        // Recurse on the left side if dir is negative.

                        stack[stackSize++] = {
                            &children[dirPositive ? 1 : 0],
                            start,
                            end,
                        };
                    }
                    else if (t >= len) {
                        // Plane is "ahead" of the line end.
                        // Recurse on the left side if dir is positive.
                        // Recurse on the right side if dir is negative.

                        stack[stackSize++] = {
                            &children[dirPositive ? 0 : 1],
                            start,
                            end
                        };
                    }
                    else {
                        // The line segment straddles the plane.
                        // Clip the line and recurse on both sides.

                        float3 clipPoint = start + t * dir;

                        stack[stackSize++] = {
                            &children[dirPositive ? 0 : 1],
                            start,
                            clipPoint,
                        };

                        stack[stackSize++] = {
                            &children[dirPositive ? 1 : 0],
                            clipPoint,
                            end,
                        };
                    }

                    break;
            }
        }

        return false;
    }
}
