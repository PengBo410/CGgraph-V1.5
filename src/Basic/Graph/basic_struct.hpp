#pragma once

#include "Basic/Type/data_type.hpp"
#include <atomic>
#include <cstddef>
#include <string>
#include <vector>

struct CSR_Result_type
{
    count_type vertexNum = 0;
    countl_type edgeNum = 0;
    countl_type* csr_offset = nullptr;
    vertex_id_type* csr_dest = nullptr;
    edge_data_type* csr_weight = nullptr;

    degree_type* outDegree = nullptr;
    degree_type* inDegree = nullptr;

    count_type noZeroOutDegreeNum = 0;

    void clearCSR()
    {
        if (csr_offset != nullptr)
        {
            delete[] csr_offset;
            csr_offset = nullptr;
        }

        if (csr_dest != nullptr)
        {
            delete[] csr_dest;
            csr_dest = nullptr;
        }

        if (csr_weight != nullptr)
        {
            delete[] csr_weight;
            csr_weight = nullptr;
        }

        if (outDegree != nullptr)
        {
            delete[] outDegree;
            outDegree = nullptr;
        }

        if (inDegree != nullptr)
        {
            delete[] inDegree;
            inDegree = nullptr;
        }
    }
};

struct CSC_Result_type
{
    count_type vertexNum = 0;
    countl_type edgeNum = 0;
    countl_type* csc_offset = nullptr;
    vertex_id_type* csc_src = nullptr;
    edge_data_type* csc_weight = nullptr;

    degree_type* outDegree = nullptr;
    degree_type* inDegree = nullptr;

    count_type noZeroOutDegreeNum = 0;

    void clearCSC()
    {
        if (csc_offset != nullptr)
        {
            delete[] csc_offset;
            csc_offset = nullptr;
        }

        if (csc_src != nullptr)
        {
            delete[] csc_src;
            csc_src = nullptr;
        }

        if (csc_weight != nullptr)
        {
            delete[] csc_weight;
            csc_weight = nullptr;
        }

        if (outDegree != nullptr)
        {
            delete[] outDegree;
            outDegree = nullptr;
        }

        if (inDegree != nullptr)
        {
            delete[] inDegree;
            inDegree = nullptr;
        }
    }
};

struct Degree_type
{
    count_type vertexNum;
    countl_type edgeNum;
    degree_type* outDegree;
    degree_type* inDegree;
};

/****************************************************************************
 *                              [Degree ÅÅÐò]
 ****************************************************************************/
enum SortDegree { OUTDEGREE, INDEGREES, DEGREE };

/****************************************************************************
 *                            [Graph Representation]
 ****************************************************************************/
enum GraphRepresentation { CSR, CSC };

/****************************************************************************
 *                              [GraphFile Struct]
 ****************************************************************************/
struct GraphFile_type
{
    std::string graphFile = "";
    std::string graphFile_noWeight = "";
    size_t vertices = 0;

    size_t common_root = 0;
    size_t edges = 0;
    std::string old2newFile = "";
    std::string addtitionFile = "";
    std::string rankFile = ""; // new2oldFile

    std::string csrOffsetFile = "";
    std::string csrDestFile = "";
    std::string csrWeightFile = "";
};

struct GraphFileV2_type
{
    GraphRepresentation graphRepresentation;

    std::string graphFile = "";
    size_t vertices = 0;
    size_t edges = 0;

    size_t common_root = 0;
    std::string old2newFile = "";
    std::string addtitionFile = "";

    std::string csrOffsetFile = "";
    std::string csrDestFile = "";
    std::string csrWeightFile = "";

    std::string cscOffsetFile = "";
    std::string cscSrcFile = "";
    std::string cscWeightFile = "";

    std::string inDegreeFile = "";
    std::string outDegreeFile = "";
};
struct Empty_type
{
};

struct EdgeUnit_hasWeight
{
    vertex_id_type src;
    vertex_id_type dest;
    edge_data_type weight;
} __attribute__((packed)); // Cancel align

struct EdgeUnit_noWeight
{
    vertex_id_type src;
    vertex_id_type dest;
} __attribute__((packed)); // Cancel align

template <typename T>
using Edge_unit_type = typename std::conditional<std::is_same_v<T, Empty_type>, EdgeUnit_noWeight, EdgeUnit_hasWeight>::type;

struct NbrUnit_hasWeight
{
    vertex_id_type dest;
    edge_data_type weight;
} __attribute__((packed));

struct NbrUnit_noWeight
{
    vertex_id_type dest;
} __attribute__((packed));

template <typename T>
using Nbr_unit_type = typename std::conditional<std::is_same_v<T, Empty_type>, NbrUnit_noWeight, NbrUnit_hasWeight>::type;

/****************************************************************************
 *                              [Adaptive]
 ****************************************************************************/
struct Adaptive_type
{
    double rate_cpu = 0.0;
    double rate_gpu = 0.0;
    double rate_pcie = 0.0;
    std::string adaptiveFile = "";
};

enum RATE_Type { CPU, GPU, PCIe, Reduce };

struct Adaptive_info_type
{
    RATE_Type rate_type;
    // std::string graphName = "";
    // std::string algorithm = "";
    size_t ite = 0;
    size_t vertexNum_workload = 0;
    size_t edgeNum_workload = 0;
    double score = 0.0;
    double time = 0.0;
    double rate = 0.0;
};

typedef std::vector<std::pair<vertex_id_type, vertex_id_type>> EdgeList_noWeight_type;

//>-----------------------------------------------------------------------------------------------------------
//>-------------------------------------[We Update A New Version]---------------------------------------------
//>-----------------------------------------------------------------------------------------------------------

enum class Algorithm_type {
    BFS,
    SSSP,

    MAX_VALUE // Use to get max value
};
const char* Algorithm_type_name[static_cast<uint32_t>(Algorithm_type::MAX_VALUE)] = {"BFS", "SSSP"};
const char* Algorithm_type_help = "The Algorithm To Be Run: [0]:BFS, [1]:SSSP";

enum class GPU_memory_type {
    GPU_MEM,   // cudaMalloc
    UVM,       // cudaMemAdviseSetReadMostly
    ZERO_COPY, // cudaMemAdviseSetAccessedBy

    MAX_VALUE // Use to get max value
};
const char* GPU_memory_type_name[static_cast<uint32_t>(GPU_memory_type::MAX_VALUE)] = {"GPU_MEM", "UVM", "ZERO_COPY"};
const char* GPU_memory_type_help = "The GPU Memory Type: [0]:GPU_MEM, [1]:UVM, [2]:ZERO_COPY";

enum class OrderMethod_type {
    NATIVE,
    CGgraphRV1_5,

    MAX_VALUE
};
const char* OrderMethod_type_name[static_cast<uint32_t>(OrderMethod_type::MAX_VALUE)] = {"native", "CGgraphRV1_5"};
const char* OrderMethod_type_help = "The OrderMethod Can Be: [0]:Native, [1]:CGgraphRV1_5";

struct CheckInfo_type
{
    std::string graphName = "";
    Algorithm_type algorithm;
    OrderMethod_type orderMethod;
    size_t root = 0;
};

enum class AUTO_GPUMEM_type {
    FULL_DEVICE_MEM,
    PARTIAL_DEVICE_MEM,
    DISABLE_DEVICE_MEM,
    DISABLE_GPU,

    MAX_VALUE
};
const char* AUTO_GPUMEM_type_name[static_cast<uint32_t>(AUTO_GPUMEM_type::MAX_VALUE)] = {"FullDeviceMem", "PartialDeviceMem", "DisableDeviceMem",
                                                                                         "DisableGPU"};

struct Host_dataPointer_type
{
    /* DATA */
    count_type vertexNum_host_{0};
    countl_type edgeNum_host_{0};
    countl_type* csr_offset_host_{nullptr};
    vertex_id_type* csr_dest_host_{nullptr};
    edge_data_type* csr_weight_host_{nullptr};

    vertex_id_type* csr_destWeight_host_{nullptr};
    bool SSSP_dest_weight{false};
};

struct Device_dataPointer_type
{
    /* DATA */
    count_type vertexNum_device_{0};
    countl_type edgeNum_device_{0};
    count_type cutVertexId_device_{0};
    countl_type* csr_offset_device_{nullptr};
    vertex_id_type* csr_dest_device_{nullptr};
    edge_data_type* csr_weight_device_{nullptr};
    bool isEntireGraph{true};
    bool disableGPU{false};

    vertex_id_type* csr_destWeight_device_{nullptr};
    bool SSSP_dest_weight{false};
};

struct SpeedRecord_type
{
    int iteId{0};
    count_type activeVertexNum{0};
    count_type activeEdgeNum{0};
    double time_ms{0.0};
    double total_time_ms{0.0};
};