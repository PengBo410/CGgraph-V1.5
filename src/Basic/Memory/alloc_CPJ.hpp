#pragma once

#include "Basic/CUDA/cuda_check.cuh"
#include "Basic/CUDA/gpu_util.cuh"
#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Graph/basic_def.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numa.h>
#include <omp.h>

#ifdef USE_JEMALLOC
#include <jemalloc/jemalloc.h>
#endif

namespace CPJ {

namespace AllocMem {

enum AllocMem_type { NEW, MALLOC, ALIGNMNET, PIN, NUMA, JEMALLOC_MALLOC, JEMALLOC_ALIGNMENT };
constexpr AllocMem_type ALLOC_MEM_TYPE = AllocMem_type::NEW;
constexpr size_t ALLOC_ALIGNMENT_SIZE = 64;

template <typename T>
inline T* allocMem(const size_t eleNum, const int socketId = 0)
{
    T* res{nullptr};
    if constexpr (ALLOC_MEM_TYPE == AllocMem_type::NEW)
    {
        res = new T[eleNum];
        assert_msg(res != nullptr, "new fail");
    }
    else if constexpr (ALLOC_MEM_TYPE == AllocMem_type::MALLOC)
    {
#ifdef USE_JEMALLOC
        assert_msg(false, "[MALLOC] need close JEMALLOC lib");
#endif
        res = (T*)malloc(eleNum * sizeof(T));
        assert_msg(res != nullptr, "malloc fail");
    }
    else if constexpr (ALLOC_MEM_TYPE == AllocMem_type::ALIGNMNET)
    {
        res = (T*)std::aligned_alloc(ALLOC_ALIGNMENT_SIZE, eleNum * sizeof(T));
        assert_msg(res != nullptr, "std::aligned_alloc fail");
    }
    else if constexpr (ALLOC_MEM_TYPE == AllocMem_type::PIN)
    {
        CUDA_CHECK(MALLOC_HOST(&res, eleNum));
    }
    else if constexpr (ALLOC_MEM_TYPE == AllocMem_type::NUMA)
    {
        res = (T*)numa_alloc_onnode(eleNum * sizeof(T), socketId);
        assert_msg(res != nullptr, "numa_alloc_onnode on socket(%d) fail", socketId);
    }
    else if constexpr (ALLOC_MEM_TYPE == AllocMem_type::JEMALLOC_MALLOC)
    {
#ifndef USE_JEMALLOC
        assert_msg(false, "[JEMALLOC_MALLOC] need open JEMALLOC lib");
#endif
        res = (T*)malloc(eleNum * sizeof(T));
        assert_msg(res != nullptr, "malloc fail");
    }
    else if constexpr (ALLOC_MEM_TYPE == AllocMem_type::JEMALLOC_ALIGNMENT)
    {
#ifndef USE_JEMALLOC
        assert_msg(false, "[JEMALLOC_ALIGNMENT] need open JEMALLOC lib");
#endif
#ifdef USE_JEMALLOC
        res = (T*)memalign(ALLOC_ALIGNMENT_SIZE, eleNum * sizeof(T));
#endif
        assert_msg(res != nullptr, "std::aligned_alloc fail");
    }
    else
    {
        assert_msg(false, "Unknown ALLOC Type");
    }
    return res;
}

template <typename T>
inline T* allocMem_memset(const size_t eleNum, const int socketId = 0)
{
    T* res{nullptr};
    if constexpr (ALLOC_MEM_TYPE == AllocMem_type::NEW)
    {
        res = new T[eleNum];
        assert_msg(res != nullptr, "new fail");
    }
    else if constexpr (ALLOC_MEM_TYPE == AllocMem_type::MALLOC)
    {
#ifdef USE_JEMALLOC
        assert_msg(false, "[MALLOC] need close JEMALLOC lib");
#endif
        res = (T*)malloc(eleNum * sizeof(T));
        assert_msg(res != nullptr, "malloc fail");
    }
    else if constexpr (ALLOC_MEM_TYPE == AllocMem_type::ALIGNMNET)
    {
        res = (T*)std::aligned_alloc(ALLOC_ALIGNMENT_SIZE, eleNum * sizeof(T));
        assert_msg(res != nullptr, "std::aligned_alloc fail");
    }
    else if constexpr (ALLOC_MEM_TYPE == AllocMem_type::PIN)
    {
        CUDA_CHECK(MALLOC_HOST(&res, eleNum));
    }
    else if constexpr (ALLOC_MEM_TYPE == AllocMem_type::NUMA)
    {
        res = (T*)numa_alloc_onnode(eleNum * sizeof(T), socketId);
        assert_msg(res != nullptr, "numa_alloc_onnode on socket(%d) fail", socketId);
    }
    else if constexpr (ALLOC_MEM_TYPE == AllocMem_type::JEMALLOC_MALLOC)
    {
#ifndef USE_JEMALLOC
        assert_msg(false, "[JEMALLOC_MALLOC] need open JEMALLOC lib");
#endif
        res = (T*)malloc(eleNum * sizeof(T));
        assert_msg(res != nullptr, "malloc fail");
    }
    else if constexpr (ALLOC_MEM_TYPE == AllocMem_type::JEMALLOC_ALIGNMENT)
    {
#ifndef USE_JEMALLOC
        assert_msg(false, "[JEMALLOC_ALIGNMENT] need open JEMALLOC lib");
#endif
#ifdef USE_JEMALLOC
        res = (T*)memalign(ALLOC_ALIGNMENT_SIZE, eleNum * sizeof(T));
#endif
        assert_msg(res != nullptr, "std::aligned_alloc fail");
    }
    else
    {
        assert_msg(false, "Unknown ALLOC Type");
    }
    std::memset(res, 0, eleNum * sizeof(T));
    return res;
}

template <typename T>
inline void freeMem(T* res, const size_t eleNum)
{
    if (res == nullptr) return;

    if constexpr (ALLOC_MEM_TYPE == AllocMem_type::NEW)
    {
        delete[] res;
    }
    else if constexpr (ALLOC_MEM_TYPE == AllocMem_type::MALLOC)
    {
        free(res);
    }
    else if constexpr (ALLOC_MEM_TYPE == AllocMem_type::ALIGNMNET)
    {
        std::free(res);
    }
    else if constexpr (ALLOC_MEM_TYPE == AllocMem_type::PIN)
    {
        CUDA_CHECK(FREE_HOST(res));
    }
    else if constexpr (ALLOC_MEM_TYPE == AllocMem_type::NUMA)
    {
        numa_free(res, eleNum * sizeof(T));
    }
    else if constexpr (ALLOC_MEM_TYPE == AllocMem_type::JEMALLOC_MALLOC)
    {
        free(res);
    }
    else if constexpr (ALLOC_MEM_TYPE == AllocMem_type::JEMALLOC_ALIGNMENT)
    {
        free(res);
    }
    else
    {
        assert_msg(false, "Unknown ALLOC Type");
    }
    res = nullptr;
}

} // namespace AllocMem

namespace ManagedMem {

/* ******************************************************************************************************************************
 * @description:
 *          我们以twitter2010的边为例子
 *          CPJ::ManagedMem::memcpy_smart_old(csr_dest_device_, csr_dest_host_, edgeNum_); (406.37(ms))
 *          CPJ::ManagedMem::memcpy_smart(csr_dest_device_, csr_dest_host_, edgeNum_); (354.8(ms))
 *          std::memcpy(csr_dest_device_, csr_dest_host_, edgeNum_ * sizeof(vertex_id_type)); (4.51(s))
 *          std::copy(std::execution::par_unseq, csr_dest_host_, csr_dest_host_ + edgeNum_, csr_dest_device_); (435.36(ms))
 * @return [*]
 * ******************************************************************************************************************************/
template <typename T>
void memcpy_smart_old(T* dest_ptr, T* src_ptr, const uint64_t ele_size, const uint64_t chunk_size = MB(1), size_t TH = 1024 * 1024)
{
    const size_t size_char = ele_size * sizeof(T);
    // 如果数据量小于阈值，则直接使用 std::memcpy
    if (size_char < TH)
    {
        std::memcpy(dest_ptr, src_ptr, size_char);
    }
    else
    {
        const size_t num_chunks = size_char / chunk_size; // 确保按字节计算 chunk 数量

        // 使用 OpenMP 并行化内存拷贝
#pragma omp parallel for
        for (size_t i = 0; i < num_chunks; ++i)
        {
            std::memcpy(reinterpret_cast<char*>(dest_ptr) + i * chunk_size, reinterpret_cast<char*>(src_ptr) + i * chunk_size, chunk_size);
        }

        // 处理剩余部分
        if (size_char % chunk_size != 0)
        {
            std::memcpy(reinterpret_cast<char*>(dest_ptr) + num_chunks * chunk_size, reinterpret_cast<char*>(src_ptr) + num_chunks * chunk_size,
                        size_char % chunk_size);
        }
    }
}

template <typename T>
void memcpy_smart(T* dest_ptr, T* src_ptr, const uint64_t ele_size, const uint64_t chunk_size = MB(1), size_t TH = 1024 * 1024)
{
    const size_t size_char = ele_size * sizeof(T);
    // 如果数据量小于阈值，则直接使用 std::memcpy
    if (size_char < TH)
    {
        std::memcpy(dest_ptr, src_ptr, size_char);
    }
    else
    {
        const size_t num_chunks = size_char / chunk_size; // 确保按字节计算 chunk 数量
        const size_t num_chunks_avg = num_chunks / omp_get_max_threads();

#pragma omp parallel
        {
            const int threadId = omp_get_thread_num();
            const size_t chunkId_first = threadId * num_chunks_avg;
            if (threadId == (omp_get_max_threads() - 1))
            {
                const size_t chunkId_last = num_chunks;
                if (size_char % chunk_size != 0)
                {
                    std::memcpy(reinterpret_cast<char*>(dest_ptr) + chunkId_first * chunk_size,
                                reinterpret_cast<char*>(src_ptr) + chunkId_first * chunk_size,
                                (chunkId_last - chunkId_first) * chunk_size + (size_char % chunk_size));
                }
                else
                {
                    std::memcpy(reinterpret_cast<char*>(dest_ptr) + chunkId_first * chunk_size,
                                reinterpret_cast<char*>(src_ptr) + chunkId_first * chunk_size, (chunkId_last - chunkId_first) * chunk_size);
                }
            }
            else
            {
                const size_t chunkId_last = (threadId + 1) * num_chunks_avg;
                std::memcpy(reinterpret_cast<char*>(dest_ptr) + chunkId_first * chunk_size,
                            reinterpret_cast<char*>(src_ptr) + chunkId_first * chunk_size, (chunkId_last - chunkId_first) * chunk_size);
            }
        }
    }
}

template <typename T>
void memset_smart(T* ptr, const uint64_t ele_size, const uint64_t chunk_size = MB(1), size_t TH = 1024 * 1024)
{
    const size_t size_char = ele_size * sizeof(T); // 计算总字节数
    // 如果数据量小于阈值，则直接使用 std::memset
    if (size_char < TH)
    {
        std::memset(ptr, 0, size_char); // 串行清零
    }
    else
    {
        const size_t num_chunks = size_char / chunk_size; // 按字节计算需要的块数
        const size_t num_chunks_avg = num_chunks / omp_get_max_threads();

#pragma omp parallel
        {
            const int threadId = omp_get_thread_num();
            const size_t chunkId_first = threadId * num_chunks_avg;
            if (threadId == (omp_get_max_threads() - 1))
            {
                const size_t chunkId_last = num_chunks;
                if (size_char % chunk_size != 0)
                {
                    std::memset(reinterpret_cast<char*>(ptr) + chunkId_first * chunk_size, 0,
                                (chunkId_last - chunkId_first) * chunk_size + (size_char % chunk_size));
                }
                else
                {
                    std::memset(reinterpret_cast<char*>(ptr) + chunkId_first * chunk_size, 0, (chunkId_last - chunkId_first) * chunk_size);
                }
            }
            else
            {
                const size_t chunkId_last = (threadId + 1) * num_chunks_avg;
                std::memset(reinterpret_cast<char*>(ptr) + chunkId_first * chunk_size, 0, (chunkId_last - chunkId_first) * chunk_size);
            }
        }
    }
}

} // namespace ManagedMem

} // namespace CPJ