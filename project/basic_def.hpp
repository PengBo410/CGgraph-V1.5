#pragma once

#include "Basic/Type/data_type.hpp"
#include <limits>

namespace CPJ {

#define WARPSIZE 32
#define CHUNK_SHIFT 3
#define CHUNK_SIZE (1 << CHUNK_SHIFT)
#define BLOCKSIZE_2 128
#define WARP_SHIFT 5

#define MEM_ALIGN_64 (~(0xfULL))
#define MEM_ALIGN_32 (~(0x1fULL))

typedef vertex_id_type emogi_type;

template <typename T>
struct MemAlignTrait
{
    static const emogi_type value = static_cast<emogi_type>(MEM_ALIGN_32); // 默认值为MEM_ALIGN_32，表示不支持的类型
    constexpr static const vertex_data_type vertexValue = std::numeric_limits<emogi_type>::max();
};

template <>
struct MemAlignTrait<uint32_t>
{
    static const emogi_type value = static_cast<emogi_type>(MEM_ALIGN_32); // 32位时的对齐值
    constexpr static const vertex_data_type vertexValue = std::numeric_limits<emogi_type>::max();
};

template <>
struct MemAlignTrait<uint64_t>
{
    static const emogi_type value = static_cast<emogi_type>(MEM_ALIGN_64); // 64位时的对齐值
    constexpr static const vertex_data_type vertexValue = std::numeric_limits<emogi_type>::max();
};

constexpr emogi_type MEM_ALIGN_CPJ = MemAlignTrait<emogi_type>::value;
constexpr vertex_data_type MYINFINITY_cpj = MemAlignTrait<vertex_data_type>::vertexValue;

#define TASK_PER_THREAD 4
#define TASK_PER_BLOCK (TASK_PER_THREAD * BLOCKSIZE_2)
#define MAX_BLOCKS 65535
#define INT_SIZE 32

// #define BFS_OPT_DEBUG
// #define BFS_OPT_ASSERT_DEBUG

//> ----------------------------
#define HOST_VERTEXSTEAL_VERTEXNUM 64

#define HOST_PER_THREAD_MIN_EDGES_TASKS 6400
#define HOST_PER_THREAD_MIN_VERTEX_TASKS 64
#define HOST_THREAD_FLUSH_VALUE 640

// #define HOST_BFS_OPT_DEBUG
// #define TEMP_THREAD_TIME_DEBUG

#define HOST_PER_THREAD_WORKING_VERTEX 64
#define HOST_PER_THREAD_STEALING_VERTEX 16

// #define getVisitedIncAndSortedFrontierOut_debug

//>------------------------------------
#define ALLOC_ALIGNMENT 64
#define CPU_ONLY_MODEL_VERTEX_TH 256000
#define CPU_ONLY_MODEL_EDGE_TH 50000000 // 80000000

#define GPU_ONLY_MODEL_VERTEX_TH (CPU_ONLY_MODEL_VERTEX_TH * 2)
#define GPU_ONLY_MODEL_EDGE_TH (CPU_ONLY_MODEL_EDGE_TH * 2)

// #define NO_USE_GPU_ONLY_DEBUG

} // namespace CPJ