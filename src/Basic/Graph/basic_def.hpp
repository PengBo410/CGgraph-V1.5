#pragma once

#include "Basic/Type/data_type.hpp"
#include <cstdint>
#include <string>

#define WORDSIZE 64
#define RATE_UTIL 100000000
#define PAGESIZE 4096

#define BLOCKSIZE 256
#define WARPSIZE 32
#define HALFWARP 16
#define WARP_SHIFT 5
const constexpr uint32_t WARP_NUM_PER_BLOCK = BLOCKSIZE / WARPSIZE;
#define DONATE_POOL_SIZE 32

#define SUB_VERTEXSET 64
#define VERTEXSTEAL_CHUNK 64
#define VERTEXWORK_CHUNK 64
#define EDGESTEAL_THRESHOLD 12800
#define EDGESTEAL_CHUNK 6400
#define VertexValue_MAX 999999999

/* **********************************************************************
 *                              【ALIGN】
 * **********************************************************************/
#define CACHE_ALIGNED __attribute__((aligned(64)))

/* **********************************************************************
 *                              【WORD】
 * **********************************************************************/
#define WORD_OFFSET(word) ((word) >> 6) // word / 64
#define WORD_MOD(word) ((word)&0x3f)    // word % 64

// 不检查类型
#define cpj_max(a, b) ((a > b) ? a : b)
#define cpj_min(a, b) ((a > b) ? b : a)

#define NUMA_AWARE

// 存储
#define KB(x) (static_cast<size_t>(x) << 10)
#define MB(x) (static_cast<size_t>(x) << 20)
#define GB(x) (static_cast<size_t>(x) << 30)

#define BYTES_TO_GB(bytes) ((bytes) / (1024.0 * 1024 * 1024))
#define BYTES_TO_MB(bytes) ((bytes) / (1024.0 * 1024))
#define BYTES_TO_KB(bytes) ((bytes) / 1024.0)

// static_cast
#define SC(x) (static_cast<uint64_t>(x))
#define SCast(x) (static_cast<uint64_t>(x))
#define SCU64(x) (static_cast<uint64_t>(x))
#define SCI64(x) (static_cast<int64_t>(x))
#define SCD(x) (static_cast<double>(x))
#define SCU32(x) (static_cast<uint32_t>(x))
#define SCI32(x) (static_cast<int32_t>(x))

// PageRank
#define Alpha 0.85
#define Tolerance 0.0000001 // 0.0001
#define MaxIte_PR 5000
#define CHECK_Tolerance 0.1

// Count
#define MILLION(x) (x * 1000000.0)
#define BILLION(x) (x * 1000000000.0)
#define TO_MILLION(x) (x / 1000000.0)
#define TO_BILLION(x) (x / 1000000000.0)

// 文件路劲符号
#define SEPARATOR (std::filesystem::path::preferred_separator)

inline std::string getCGgraph_reorder_basePath() { return "./CGgraphV1-5/"; }
inline std::string getCGgraph_reorder_csrOffset(std::string graphName)
{
    return getCGgraph_reorder_basePath() + graphName + "_csrOffset_u" + std::string((sizeof(countl_type) == 4) ? "32" : "64") + ".bin";
}
inline std::string getCGgraph_reorder_csrDest(std::string graphName)
{
    return getCGgraph_reorder_basePath() + graphName + "_csrDest_u" + std::string((sizeof(vertex_id_type) == 4) ? "32" : "64") + ".bin";
}
inline std::string getCGgraph_reorder_csrWeight(std::string graphName)
{
    return getCGgraph_reorder_basePath() + graphName + "_csrWeight_u" + std::string((sizeof(edge_data_type) == 4) ? "32" : "64") + ".bin";
}
inline std::string getCGgraph_reorder_rankFile(std::string graphName)
{
    return getCGgraph_reorder_basePath() + graphName + "_rank_u" + std::string((sizeof(count_type) == 4) ? "32" : "64") + ".bin";
}
inline std::string getCGgraph_reorder_old2newFile(std::string graphName)
{
    return getCGgraph_reorder_basePath() + graphName + "_old2new_u" + std::string((sizeof(edge_data_type) == 4) ? "32" : "64") + ".bin";
}