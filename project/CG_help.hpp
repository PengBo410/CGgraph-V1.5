#pragma once

#include "Basic/CUDA/cuda_check.cuh"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Thread/atomic_linux.hpp"
#include "Basic/Type/data_type.hpp"
#include "basic_def.hpp"
#include "favor.hpp"

#include <atomic>

#include <cub/block/block_scan.cuh>

namespace CPJ {

namespace Host {
namespace Balance_CG {

enum class ThreadStatus { VERTEX_WORKING, VERTEX_STEALING };

struct ThreadState_type
{
    count_type start{0};
    std::atomic<count_type> cur{0};
    count_type end{0};

    degree_type firstNbrIndex_InFirstVertex_thread{0};
    degree_type lastNbrIndex_InLastVertex_thread{0};

    ThreadStatus status;
};

struct ThreadState_forIncModel_type
{
    count_type start{0};
    std::atomic<count_type> cur{0};
    count_type end{0};

    std::atomic<count_type> startIndex{0};

    ThreadStatus status;
};

} // namespace Balance_CG
} // namespace Host

namespace Device {

struct Set_activeDegree_type
{
    vertex_id_type* frontier_in_device_{nullptr};
    const countl_type* csr_offset_device_{nullptr};
    countl_type* frontier_degExSum_device_{nullptr};

    Set_activeDegree_type(vertex_id_type* frontier_in_device, const countl_type* csr_offset_device, countl_type* frontier_degExSum_device)
        : frontier_in_device_(frontier_in_device), csr_offset_device_(csr_offset_device), frontier_degExSum_device_(frontier_degExSum_device)
    {
    }

    __device__ __forceinline__ void operator()(const count_type& frontier_id)
    {
        vertex_id_type vid = frontier_in_device_[frontier_id];
        countl_type deg = csr_offset_device_[vid + 1] - csr_offset_device_[vid];
        frontier_degExSum_device_[frontier_id] = deg;
    }
};

struct Set_blockTask_type
{
    __device__ __forceinline__ countl_type operator()(const countl_type& blockId) { return (TASK_PER_BLOCK * blockId); }
};

template <typename Value_type, typename Size_type>
__device__ __forceinline__ Size_type binsearch(const Value_type* vec, const Value_type val, Size_type low, Size_type high)
{
    while (true)
    {
        if (low == high) return low; // we know it exists
        if ((low + 1) == high) return (vec[high] <= val) ? high : low;

        Size_type mid = low + (high - low) / 2;

        if (vec[mid] > val) high = mid - 1;
        else low = mid;
    }
}

__global__ void balance_model_device_kernel(vertex_data_type* __restrict__ vertexValue, const countl_type* __restrict__ csr_offset,
                                            const vertex_id_type* __restrict__ csr_dest, const int level, /* 基本数据 */
                                            const vertex_id_type* __restrict__ frontier_in, vertex_id_type* __restrict__ frontier_out, /* frontier */
                                            const count_type frontierNum, const countl_type totalDegree,
                                            count_type* frontierNum_next, /* frontier 常量信息 */
                                            const countl_type* __restrict__ frontier_degExSum_,
                                            const countl_type* __restrict__ frontier_balance_device, const uint32_t* __restrict__ notSinkBitset,
                                            uint32_t* __restrict__ visitedBitset)
{
    countl_type maxBlockNum = (totalDegree + gridDim.x * TASK_PER_BLOCK - 1) / (gridDim.x * TASK_PER_BLOCK);
    for (countl_type maxBlock_id = 0; maxBlock_id < maxBlockNum; maxBlock_id++)
    {
        countl_type logical_block_id = gridDim.x * maxBlock_id + blockIdx.x;
        countl_type firstEdgeIndex_eachBlock = logical_block_id * TASK_PER_BLOCK;
        if (firstEdgeIndex_eachBlock >= totalDegree) break;

        countl_type lastEdgeIndex_eachBlock =
            cuda_min(static_cast<countl_type>(totalDegree - 1), static_cast<countl_type>((logical_block_id + 1) * TASK_PER_BLOCK - 1));

        count_type firstVertexIndex_eachBlock = frontier_balance_device[logical_block_id] - 1;
        count_type lastVertexIndex_eachBlock;

        if (lastEdgeIndex_eachBlock < (totalDegree - 1))
        {
            lastVertexIndex_eachBlock = frontier_balance_device[logical_block_id + 1] - 1;
            if (frontier_degExSum_[lastVertexIndex_eachBlock] == lastEdgeIndex_eachBlock + 1) lastVertexIndex_eachBlock--;
        }
        else
        {
            lastVertexIndex_eachBlock = frontierNum - 1;
        }

        count_type vertexNum_inCurBlock = lastVertexIndex_eachBlock - firstVertexIndex_eachBlock + 1;

        countl_type firstNbrIndex_InFirstVertex_inCurBlock = firstEdgeIndex_eachBlock - frontier_degExSum_[firstVertexIndex_eachBlock];
        countl_type lastNbrIndex_InLastVertex_inCurBlock = lastEdgeIndex_eachBlock - frontier_degExSum_[lastVertexIndex_eachBlock]; // 不对

        /* 共享内存 */
        __shared__ count_type frontierCounter_atomic_shared;
        __shared__ vertex_id_type frontierOut_shared[TASK_PER_BLOCK];
        __shared__ countl_type nbr_offset_start[BLOCKSIZE_2];
        __shared__ countl_type nbrSize_exsum_block[BLOCKSIZE_2];
        __shared__ count_type global_offset;

        typedef cub::BlockScan<countl_type, BLOCKSIZE_2> BlockScan_countl_type;
        __shared__ typename BlockScan_countl_type::TempStorage temp_storage;

        const unsigned iterNumForVertex_curBlock = (vertexNum_inCurBlock + BLOCKSIZE_2 - 1) / BLOCKSIZE_2;
        for (unsigned iter_id_forVertex = 0; iter_id_forVertex < iterNumForVertex_curBlock; iter_id_forVertex++)
        {
            if (iter_id_forVertex) sync_block();

            if ((!threadIdx.x) && (!iter_id_forVertex))
            {
                frontierCounter_atomic_shared = 0;
            }

            unsigned logical_threadId_forVertex = iter_id_forVertex * BLOCKSIZE_2 + threadIdx.x;
            unsigned vertexNum_forCurIterBlock;
            countl_type nbrSize_forCurThread = 0;

            if (iter_id_forVertex != iterNumForVertex_curBlock - 1)
            {
                vertexNum_forCurIterBlock = BLOCKSIZE_2;
            }
            else
            {
                vertexNum_forCurIterBlock = (vertexNum_inCurBlock % BLOCKSIZE_2) ? (vertexNum_inCurBlock % BLOCKSIZE_2) : BLOCKSIZE_2;
            }

            if (logical_threadId_forVertex < vertexNum_inCurBlock)
            {
                auto vid = frontier_in[firstVertexIndex_eachBlock + logical_threadId_forVertex];
                degree_type nbrSize = csr_offset[vid + 1] - csr_offset[vid];
                countl_type nbrStartIndex = csr_offset[vid];

                nbrSize_forCurThread = static_cast<countl_type>(nbrSize);
                nbr_offset_start[threadIdx.x] = nbrStartIndex;

                if (!logical_threadId_forVertex)
                {
                    if (firstNbrIndex_InFirstVertex_inCurBlock)
                    {
                        nbrSize_forCurThread -= firstNbrIndex_InFirstVertex_inCurBlock;
                        nbr_offset_start[threadIdx.x] += firstNbrIndex_InFirstVertex_inCurBlock;
                    }
                }

                if (logical_threadId_forVertex == (vertexNum_inCurBlock - 1))
                {
                    countl_type last_select_val = lastNbrIndex_InLastVertex_inCurBlock + 1;
                    if (!(last_select_val == nbrSize))
                    {
                        nbrSize_forCurThread -= (nbrSize - last_select_val);
                    }
                }
            }

            countl_type totalDegree_forCurIterBlock;
            BlockScan_countl_type(temp_storage).ExclusiveSum(nbrSize_forCurThread, nbrSize_exsum_block[threadIdx.x], totalDegree_forCurIterBlock);
            sync_block();

            countl_type nbrSize_iters_block = (totalDegree_forCurIterBlock + BLOCKSIZE_2 - 1) / BLOCKSIZE_2;
            for (countl_type nbrSize_iter = 0; nbrSize_iter < nbrSize_iters_block; nbrSize_iter++)
            {
                countl_type logical_tid = nbrSize_iter * BLOCKSIZE_2 + threadIdx.x;
                uint16_t bsearch_idx = UINT16_MAX;
                if (logical_tid < totalDegree_forCurIterBlock)
                {
                    bsearch_idx = binsearch(nbrSize_exsum_block, logical_tid, 0u, (vertexNum_forCurIterBlock - 1));
                    countl_type nbrIndex_inCurVertex = logical_tid - nbrSize_exsum_block[bsearch_idx];

                    vertex_id_type dest = csr_dest[nbr_offset_start[bsearch_idx] + nbrIndex_inCurVertex];
                    bool is_visited = (visitedBitset[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                    if (!is_visited)
                    {
                        vertexValue[dest] = level + 1;
                        bool is_notSink = (notSinkBitset[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;

                        if (!is_notSink)
                        {
                            int new_val = 1 << (dest % INT_SIZE);
                            LinuxAtomic::CUDA::atomicOr(&visitedBitset[dest / INT_SIZE], new_val);
                        }
                        else
                        {
                            int new_val = 1 << (dest % INT_SIZE);
                            int old_val = LinuxAtomic::CUDA::atomicOr(&visitedBitset[dest / INT_SIZE], new_val);
                            bool atomic_was_visited = (old_val >> (dest % INT_SIZE)) & 1;
                            if (!atomic_was_visited)
                            {
                                auto old_idx = LinuxAtomic::CUDA::atomicAdd(&frontierCounter_atomic_shared, 1);
                                frontierOut_shared[old_idx] = dest;
                            }
                        }
                    }
                }
            }
        }

        sync_block();
        if (!threadIdx.x)
        {
            global_offset = LinuxAtomic::CUDA::atomicAdd(frontierNum_next, static_cast<count_type>(frontierCounter_atomic_shared));
        }

        sync_block();
        count_type cand_iters = (frontierCounter_atomic_shared + BLOCKSIZE_2 - 1) / BLOCKSIZE_2;
        for (count_type i = 0; i < cand_iters; i++)
        {
            auto idx = i * BLOCKSIZE_2 + threadIdx.x;
            if (idx < frontierCounter_atomic_shared)
            {
                frontier_out[global_offset + idx] = frontierOut_shared[idx];
            }
        }
    }
}

__global__ void balance_model_device_kernel_onlyBitset(const countl_type* __restrict__ csr_offset,
                                                       const vertex_id_type* __restrict__ csr_dest, /* 基本数据 */
                                                       const vertex_id_type* __restrict__ frontier_in, const count_type frontierNum,
                                                       const countl_type totalDegree, const countl_type* __restrict__ frontier_degExSum_,
                                                       const countl_type* __restrict__ frontier_balance_device, /* frontier 常量信息 */
                                                       uint32_t* __restrict__ visitedBitset)
{
    countl_type maxBlockNum = (totalDegree + gridDim.x * TASK_PER_BLOCK - 1) / (gridDim.x * TASK_PER_BLOCK);
    for (countl_type maxBlock_id = 0; maxBlock_id < maxBlockNum; maxBlock_id++)
    {
        countl_type logical_block_id = gridDim.x * maxBlock_id + blockIdx.x;
        countl_type firstEdgeIndex_eachBlock = logical_block_id * TASK_PER_BLOCK;
        if (firstEdgeIndex_eachBlock >= totalDegree) break;

        countl_type lastEdgeIndex_eachBlock =
            cuda_min(static_cast<countl_type>(totalDegree - 1), static_cast<countl_type>((logical_block_id + 1) * TASK_PER_BLOCK - 1));

        count_type firstVertexIndex_eachBlock = frontier_balance_device[logical_block_id] - 1;
        count_type lastVertexIndex_eachBlock;

        if (lastEdgeIndex_eachBlock < (totalDegree - 1))
        {
            lastVertexIndex_eachBlock = frontier_balance_device[logical_block_id + 1] - 1;
            if (frontier_degExSum_[lastVertexIndex_eachBlock] == lastEdgeIndex_eachBlock + 1) lastVertexIndex_eachBlock--;
        }
        else
        {
            lastVertexIndex_eachBlock = frontierNum - 1;
        }

        count_type vertexNum_inCurBlock = lastVertexIndex_eachBlock - firstVertexIndex_eachBlock + 1;

        countl_type firstNbrIndex_InFirstVertex_inCurBlock = firstEdgeIndex_eachBlock - frontier_degExSum_[firstVertexIndex_eachBlock];
        countl_type lastNbrIndex_InLastVertex_inCurBlock = lastEdgeIndex_eachBlock - frontier_degExSum_[lastVertexIndex_eachBlock]; // 不对

        __shared__ countl_type nbr_offset_start[BLOCKSIZE_2];
        __shared__ countl_type nbrSize_exsum_block[BLOCKSIZE_2];

        typedef cub::BlockScan<countl_type, BLOCKSIZE_2> BlockScan_countl_type;
        __shared__ typename BlockScan_countl_type::TempStorage temp_storage;

        const unsigned iterNumForVertex_curBlock = (vertexNum_inCurBlock + BLOCKSIZE_2 - 1) / BLOCKSIZE_2;
        for (unsigned iter_id_forVertex = 0; iter_id_forVertex < iterNumForVertex_curBlock; iter_id_forVertex++)
        {
            if (iter_id_forVertex) sync_block();

            unsigned logical_threadId_forVertex = iter_id_forVertex * BLOCKSIZE_2 + threadIdx.x;
            unsigned vertexNum_forCurIterBlock;
            countl_type nbrSize_forCurThread = 0;

            if (iter_id_forVertex != iterNumForVertex_curBlock - 1)
            {
                vertexNum_forCurIterBlock = BLOCKSIZE_2;
            }
            else
            {
                vertexNum_forCurIterBlock = (vertexNum_inCurBlock % BLOCKSIZE_2) ? (vertexNum_inCurBlock % BLOCKSIZE_2) : BLOCKSIZE_2;
            }

            if (logical_threadId_forVertex < vertexNum_inCurBlock)
            {
                auto vid = frontier_in[firstVertexIndex_eachBlock + logical_threadId_forVertex];
                degree_type nbrSize = csr_offset[vid + 1] - csr_offset[vid];
                countl_type nbrStartIndex = csr_offset[vid];

                nbrSize_forCurThread = static_cast<countl_type>(nbrSize);
                nbr_offset_start[threadIdx.x] = nbrStartIndex;

                if (!logical_threadId_forVertex)
                {
                    if (firstNbrIndex_InFirstVertex_inCurBlock)
                    {
                        nbrSize_forCurThread -= firstNbrIndex_InFirstVertex_inCurBlock;
                        nbr_offset_start[threadIdx.x] += firstNbrIndex_InFirstVertex_inCurBlock;
                    }
                }

                if (logical_threadId_forVertex == (vertexNum_inCurBlock - 1))
                {
                    countl_type last_select_val = lastNbrIndex_InLastVertex_inCurBlock + 1;
                    if (!(last_select_val == nbrSize))
                    {
                        nbrSize_forCurThread -= (nbrSize - last_select_val);
                    }
                }
            }

            countl_type totalDegree_forCurIterBlock;
            BlockScan_countl_type(temp_storage).ExclusiveSum(nbrSize_forCurThread, nbrSize_exsum_block[threadIdx.x], totalDegree_forCurIterBlock);
            sync_block();

            countl_type nbrSize_iters_block = (totalDegree_forCurIterBlock + BLOCKSIZE_2 - 1) / BLOCKSIZE_2;
            for (countl_type nbrSize_iter = 0; nbrSize_iter < nbrSize_iters_block; nbrSize_iter++)
            {
                countl_type logical_tid = nbrSize_iter * BLOCKSIZE_2 + threadIdx.x;
                uint16_t bsearch_idx = UINT16_MAX;
                if (logical_tid < totalDegree_forCurIterBlock)
                {
                    bsearch_idx = binsearch(nbrSize_exsum_block, logical_tid, 0u, (vertexNum_forCurIterBlock - 1));
                    countl_type nbrIndex_inCurVertex = logical_tid - nbrSize_exsum_block[bsearch_idx];

                    vertex_id_type dest = csr_dest[nbr_offset_start[bsearch_idx] + nbrIndex_inCurVertex];
                    bool is_visited = (visitedBitset[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                    if (!is_visited)
                    {
                        int new_val = 1 << (dest % INT_SIZE);
                        LinuxAtomic::CUDA::atomicOr(&visitedBitset[dest / INT_SIZE], new_val);
                    }
                }
            }
        }

        sync_block();
    }

} // end of func [balance_model_device_kernel_onlyBitset]

void balance_model_device(const int& nBlock, vertex_data_type* __restrict__ vertexValue, const countl_type* __restrict__ csr_offset,
                          const vertex_id_type* __restrict__ csr_dest, const int level,                              /* 基本数据 */
                          const vertex_id_type* __restrict__ frontier_in, vertex_id_type* __restrict__ frontier_out, /* frontier */
                          const count_type frontierNum, const countl_type totalDegree, count_type* frontierNum_next, /* frontier 常量信息 */
                          const countl_type* __restrict__ frontier_degExSum_, const countl_type* __restrict__ frontier_balance_device_,
                          const uint32_t* __restrict__ notSinkBitset, uint32_t* __restrict__ visitedBitset, const bool onlyBitset)
{
    if (onlyBitset)
    {
        CUDA_KERNEL_CALL(balance_model_device_kernel_onlyBitset, nBlock, BLOCKSIZE_2, csr_offset, csr_dest, frontier_in, frontierNum, totalDegree,
                         frontier_degExSum_, frontier_balance_device_, visitedBitset);
    }
    else
    {
        CUDA_KERNEL_CALL(balance_model_device_kernel, nBlock, BLOCKSIZE_2, vertexValue, csr_offset, csr_dest, level, frontier_in, frontier_out,
                         frontierNum, totalDegree, frontierNum_next, frontier_degExSum_, frontier_balance_device_, notSinkBitset, visitedBitset);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename Bitset_type, typename T>
struct DeviceInc_sinkPopcount_type
{
    const Bitset_type* visitedBitset_temp_device_{nullptr};
    Bitset_type* visitedBitset_device_{nullptr};
    Bitset_type* visitedBitset_curIte_device_{nullptr};
    const Bitset_type* notSinkBitset_device_{nullptr};
    int level_cur_{0};
    vertex_data_type* vertexValue_device_{nullptr};
    T* frontier_balance_device_{nullptr};

    DeviceInc_sinkPopcount_type(const Bitset_type* visitedBitset_temp_device, Bitset_type* visitedBitset_device,
                                Bitset_type* visitedBitset_curIte_device, const Bitset_type* notSinkBitset_device, int level_cur,
                                vertex_data_type* vertexValue_device, T* frontier_balance_device)
        : visitedBitset_temp_device_(visitedBitset_temp_device), visitedBitset_device_(visitedBitset_device),
          visitedBitset_curIte_device_(visitedBitset_curIte_device), notSinkBitset_device_(notSinkBitset_device), level_cur_(level_cur),
          vertexValue_device_(vertexValue_device), frontier_balance_device_(frontier_balance_device)
    {
    }

    __device__ __forceinline__ void operator()(const count_type& chunk_id)
    {
        constexpr int BITS = (sizeof(Bitset_type) * 8);

        visitedBitset_device_[chunk_id] |= visitedBitset_temp_device_[chunk_id];

        Bitset_type visitedBitset_curIte_device_temp = visitedBitset_curIte_device_[chunk_id];
        Bitset_type visitedBitset_device_temp = visitedBitset_device_[chunk_id];

        if (visitedBitset_curIte_device_temp != visitedBitset_device_temp)
        {
            visitedBitset_curIte_device_temp ^= visitedBitset_device_temp;
            if (visitedBitset_curIte_device_temp != 0)
            {
                Bitset_type notSinkBitset_device_TEMP = notSinkBitset_device_[chunk_id];
                Bitset_type onlyActiveSink = visitedBitset_curIte_device_temp & (~notSinkBitset_device_TEMP);
                vertex_id_type vertex_id_start = chunk_id * BITS;
                while (onlyActiveSink != 0)
                {
                    if (onlyActiveSink & 1)
                    {
                        vertexValue_device_[vertex_id_start] = level_cur_;
                    }
                    vertex_id_start++;
                    onlyActiveSink = onlyActiveSink >> 1;
                }

                /* 计算 popoCount */
                visitedBitset_curIte_device_temp &= (notSinkBitset_device_TEMP);
                if (visitedBitset_curIte_device_temp != 0)
                {
                    uint8_t temp = __popc(visitedBitset_curIte_device_temp);
                    frontier_balance_device_[chunk_id] = temp;
                }
            }
            visitedBitset_curIte_device_[chunk_id] = visitedBitset_curIte_device_temp;
        }
        else
        {
            visitedBitset_curIte_device_[chunk_id] = 0;
        }
    }
};

template <typename Bistset_type>
struct DeviceInc_fillFrontier_type
{
    const Bistset_type* visitedBitset_curIte_device_{nullptr};
    const countl_type* frontier_balance_device_{nullptr};
    int level_cur_{0};
    vertex_data_type* vertexValue_device_{nullptr};
    vertex_id_type* frontier_out_device_{nullptr};

    DeviceInc_fillFrontier_type(const Bistset_type* visitedBitset_curIte_device, const countl_type* frontier_balance_device, int level_cur,
                                vertex_data_type* vertexValue_device, vertex_id_type* frontier_out_device)
        : visitedBitset_curIte_device_(visitedBitset_curIte_device), frontier_balance_device_(frontier_balance_device), level_cur_(level_cur),
          vertexValue_device_(vertexValue_device), frontier_out_device_(frontier_out_device)
    {
    }

    __device__ __forceinline__ void operator()(const count_type& chunk_id)
    {
        constexpr int BITS = (sizeof(Bistset_type) * 8);

        countl_type startIndex_thread = frontier_balance_device_[chunk_id];
        Bistset_type word = visitedBitset_curIte_device_[chunk_id];
        vertex_id_type vertex_id_start = chunk_id * BITS;

        while (word != 0)
        {
            if (word & 1)
            {
                frontier_out_device_[startIndex_thread] = vertex_id_start;
                vertexValue_device_[vertex_id_start] = level_cur_;
                startIndex_thread++;
            }
            vertex_id_start++;
            word = word >> 1;
        }
    }
};

struct Static_u8_to_count_type_type
{
    uint8_t* array_{nullptr};
    Static_u8_to_count_type_type(uint8_t* array) : array_(array) {}
    __device__ __forceinline__ count_type operator()(const count_type& index) { return static_cast<count_type>(array_[index]); }
};

} // namespace Device

namespace SSSP_DEVICE_SPACE {

__global__ void SSSP_balance_model_device_kernel_destWeight(vertex_data_type* __restrict__ vertexValue, const countl_type* __restrict__ csr_offset,
                                                            const vertex_id_type* __restrict__ csr_destWeight, /* 基本数据 */
                                                            const vertex_id_type* __restrict__ frontier_in,
                                                            vertex_id_type* __restrict__ frontier_out, /* frontier */
                                                            const count_type frontierNum, const countl_type totalDegree,
                                                            count_type* frontierNum_next, /* frontier 常量信息 */
                                                            const countl_type* __restrict__ frontier_degExSum_,
                                                            const countl_type* __restrict__ frontier_balance_device,
                                                            const uint32_t* __restrict__ notSinkBitset, int* __restrict__ visitedBitset)
{
    countl_type maxBlockNum = (totalDegree + gridDim.x * TASK_PER_BLOCK - 1) / (gridDim.x * TASK_PER_BLOCK);
    for (countl_type maxBlock_id = 0; maxBlock_id < maxBlockNum; maxBlock_id++)
    {
        countl_type logical_block_id = gridDim.x * maxBlock_id + blockIdx.x;
        countl_type firstEdgeIndex_eachBlock = logical_block_id * TASK_PER_BLOCK;
        if (firstEdgeIndex_eachBlock >= totalDegree) break;

        countl_type lastEdgeIndex_eachBlock =
            cuda_min(static_cast<countl_type>(totalDegree - 1), static_cast<countl_type>((logical_block_id + 1) * TASK_PER_BLOCK - 1));

        count_type firstVertexIndex_eachBlock = frontier_balance_device[logical_block_id] - 1;
        count_type lastVertexIndex_eachBlock;

        if (lastEdgeIndex_eachBlock < (totalDegree - 1))
        {
            lastVertexIndex_eachBlock = frontier_balance_device[logical_block_id + 1] - 1;
            if (frontier_degExSum_[lastVertexIndex_eachBlock] == lastEdgeIndex_eachBlock + 1) lastVertexIndex_eachBlock--;
        }
        else
        {
            lastVertexIndex_eachBlock = frontierNum - 1;
        }

        count_type vertexNum_inCurBlock = lastVertexIndex_eachBlock - firstVertexIndex_eachBlock + 1;
        countl_type firstNbrIndex_InFirstVertex_inCurBlock = firstEdgeIndex_eachBlock - frontier_degExSum_[firstVertexIndex_eachBlock];
        countl_type lastNbrIndex_InLastVertex_inCurBlock = lastEdgeIndex_eachBlock - frontier_degExSum_[lastVertexIndex_eachBlock];

        /* 共享内存 */
        __shared__ count_type frontierCounter_atomic_shared;
        __shared__ vertex_id_type frontierOut_shared[TASK_PER_BLOCK];
        __shared__ countl_type nbr_offset_start[BLOCKSIZE_2];
        __shared__ vertex_id_type process_vertexId[BLOCKSIZE_2];
        __shared__ countl_type nbrSize_exsum_block[BLOCKSIZE_2];
        __shared__ count_type global_offset;

        typedef cub::BlockScan<countl_type, BLOCKSIZE_2> BlockScan_countl_type;
        __shared__ typename BlockScan_countl_type::TempStorage temp_storage;

        const unsigned iterNumForVertex_curBlock = (vertexNum_inCurBlock + BLOCKSIZE_2 - 1) / BLOCKSIZE_2;
        for (unsigned iter_id_forVertex = 0; iter_id_forVertex < iterNumForVertex_curBlock; iter_id_forVertex++)
        {
            if (iter_id_forVertex) sync_block();

            if ((!threadIdx.x) && (!iter_id_forVertex))
            {
                frontierCounter_atomic_shared = 0;
            }

            unsigned logical_threadId_forVertex = iter_id_forVertex * BLOCKSIZE_2 + threadIdx.x;
            unsigned vertexNum_forCurIterBlock;
            countl_type nbrSize_forCurThread = 0;

            if (iter_id_forVertex != iterNumForVertex_curBlock - 1)
            {
                vertexNum_forCurIterBlock = BLOCKSIZE_2;
            }
            else
            {
                vertexNum_forCurIterBlock = (vertexNum_inCurBlock % BLOCKSIZE_2) ? (vertexNum_inCurBlock % BLOCKSIZE_2) : BLOCKSIZE_2;
            }

            if (logical_threadId_forVertex < vertexNum_inCurBlock)
            {
                auto vid = frontier_in[firstVertexIndex_eachBlock + logical_threadId_forVertex];
                degree_type nbrSize = csr_offset[vid + 1] - csr_offset[vid];
                countl_type nbrStartIndex = csr_offset[vid];

                nbrSize_forCurThread = static_cast<countl_type>(nbrSize);
                nbr_offset_start[threadIdx.x] = nbrStartIndex;
                process_vertexId[threadIdx.x] = vid;

                if (!logical_threadId_forVertex)
                {
                    if (firstNbrIndex_InFirstVertex_inCurBlock)
                    {
                        nbrSize_forCurThread -= firstNbrIndex_InFirstVertex_inCurBlock;
                        nbr_offset_start[threadIdx.x] += firstNbrIndex_InFirstVertex_inCurBlock;
                    }
                }

                if (logical_threadId_forVertex == (vertexNum_inCurBlock - 1))
                {
                    countl_type last_select_val = lastNbrIndex_InLastVertex_inCurBlock + 1;
                    if (!(last_select_val == nbrSize))
                    {
                        nbrSize_forCurThread -= (nbrSize - last_select_val);
                    }
                }
            }

            countl_type totalDegree_forCurIterBlock;
            BlockScan_countl_type(temp_storage).ExclusiveSum(nbrSize_forCurThread, nbrSize_exsum_block[threadIdx.x], totalDegree_forCurIterBlock);
            sync_block();

            countl_type nbrSize_iters_block = (totalDegree_forCurIterBlock + BLOCKSIZE_2 - 1) / BLOCKSIZE_2;
            for (countl_type nbrSize_iter = 0; nbrSize_iter < nbrSize_iters_block; nbrSize_iter++)
            {
                countl_type logical_tid = nbrSize_iter * BLOCKSIZE_2 + threadIdx.x;
                uint16_t bsearch_idx = UINT16_MAX;
                if (logical_tid < totalDegree_forCurIterBlock)
                {
                    bsearch_idx = Device::binsearch(nbrSize_exsum_block, logical_tid, 0u, (vertexNum_forCurIterBlock - 1));
                    countl_type nbrIndex_inCurVertex = logical_tid - nbrSize_exsum_block[bsearch_idx];
                    vertex_id_type vid = process_vertexId[bsearch_idx];

                    const countl_type nbr_index = (nbr_offset_start[bsearch_idx] + nbrIndex_inCurVertex) * 2;
                    vertex_id_type dest = csr_destWeight[nbr_index];
                    edge_data_type weight = csr_destWeight[nbr_index + 1];
                    edge_data_type msg = vertexValue[vid] + weight;
                    if (msg < vertexValue[dest])
                    {
                        if (msg < LinuxAtomic::CUDA::atomicMin(&vertexValue[dest], msg))
                        {
                            bool is_visited = (visitedBitset[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                            if (!is_visited)
                            {
                                int new_val = 1 << (dest % INT_SIZE);
                                uint32_t old_val = LinuxAtomic::CUDA::atomicOr(&visitedBitset[dest / INT_SIZE], new_val);
                                bool is_notSink = (notSinkBitset[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                if (is_notSink)
                                {
                                    bool atomic_was_visited = (old_val >> (dest % INT_SIZE)) & 1;
                                    if (!atomic_was_visited)
                                    {
                                        auto old_idx = LinuxAtomic::CUDA::atomicAdd(&frontierCounter_atomic_shared, 1);
                                        frontierOut_shared[old_idx] = dest;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        sync_block();
        if (!threadIdx.x)
        {
            global_offset = LinuxAtomic::CUDA::atomicAdd(frontierNum_next, static_cast<count_type>(frontierCounter_atomic_shared));
        }
        sync_block();
        count_type cand_iters = (frontierCounter_atomic_shared + BLOCKSIZE_2 - 1) / BLOCKSIZE_2;
        for (count_type i = 0; i < cand_iters; i++)
        {
            auto idx = i * BLOCKSIZE_2 + threadIdx.x;
            if (idx < frontierCounter_atomic_shared)
            {
                frontier_out[global_offset + idx] = frontierOut_shared[idx];
            }
        }
    }
}

__global__ void
SSSP_balance_model_device_kernel(vertex_data_type* __restrict__ vertexValue, const countl_type* __restrict__ csr_offset,
                                 const vertex_id_type* __restrict__ csr_dest, const vertex_id_type* __restrict__ csr_weight, /* 基本数据 */
                                 const vertex_id_type* __restrict__ frontier_in, vertex_id_type* __restrict__ frontier_out,  /* frontier */
                                 const count_type frontierNum, const countl_type totalDegree, count_type* frontierNum_next, /* frontier 常量信息 */
                                 const countl_type* __restrict__ frontier_degExSum_, const countl_type* __restrict__ frontier_balance_device,
                                 const uint32_t* __restrict__ notSinkBitset, int* __restrict__ visitedBitset)
{
    countl_type maxBlockNum = (totalDegree + gridDim.x * TASK_PER_BLOCK - 1) / (gridDim.x * TASK_PER_BLOCK);
    for (countl_type maxBlock_id = 0; maxBlock_id < maxBlockNum; maxBlock_id++)
    {
        countl_type logical_block_id = gridDim.x * maxBlock_id + blockIdx.x;
        countl_type firstEdgeIndex_eachBlock = logical_block_id * TASK_PER_BLOCK;
        if (firstEdgeIndex_eachBlock >= totalDegree) break;

        countl_type lastEdgeIndex_eachBlock =
            cuda_min(static_cast<countl_type>(totalDegree - 1), static_cast<countl_type>((logical_block_id + 1) * TASK_PER_BLOCK - 1));

        count_type firstVertexIndex_eachBlock = frontier_balance_device[logical_block_id] - 1;
        count_type lastVertexIndex_eachBlock;

        if (lastEdgeIndex_eachBlock < (totalDegree - 1))
        {
            lastVertexIndex_eachBlock = frontier_balance_device[logical_block_id + 1] - 1;
            if (frontier_degExSum_[lastVertexIndex_eachBlock] == lastEdgeIndex_eachBlock + 1) lastVertexIndex_eachBlock--;
        }
        else
        {
            lastVertexIndex_eachBlock = frontierNum - 1;
        }

        count_type vertexNum_inCurBlock = lastVertexIndex_eachBlock - firstVertexIndex_eachBlock + 1;

        countl_type firstNbrIndex_InFirstVertex_inCurBlock = firstEdgeIndex_eachBlock - frontier_degExSum_[firstVertexIndex_eachBlock];
        countl_type lastNbrIndex_InLastVertex_inCurBlock = lastEdgeIndex_eachBlock - frontier_degExSum_[lastVertexIndex_eachBlock]; // 不对

        __shared__ count_type frontierCounter_atomic_shared;
        __shared__ vertex_id_type frontierOut_shared[TASK_PER_BLOCK];
        __shared__ countl_type nbr_offset_start[BLOCKSIZE_2];
        __shared__ vertex_id_type process_vertexId[BLOCKSIZE_2];
        __shared__ countl_type nbrSize_exsum_block[BLOCKSIZE_2];
        __shared__ count_type global_offset;

        typedef cub::BlockScan<countl_type, BLOCKSIZE_2> BlockScan_countl_type;
        __shared__ typename BlockScan_countl_type::TempStorage temp_storage;

        const unsigned iterNumForVertex_curBlock = (vertexNum_inCurBlock + BLOCKSIZE_2 - 1) / BLOCKSIZE_2;
        for (unsigned iter_id_forVertex = 0; iter_id_forVertex < iterNumForVertex_curBlock; iter_id_forVertex++)
        {
            if (iter_id_forVertex) sync_block();

            if ((!threadIdx.x) && (!iter_id_forVertex))
            {
                frontierCounter_atomic_shared = 0;
            }

            unsigned logical_threadId_forVertex = iter_id_forVertex * BLOCKSIZE_2 + threadIdx.x;
            unsigned vertexNum_forCurIterBlock;
            countl_type nbrSize_forCurThread = 0;

            if (iter_id_forVertex != iterNumForVertex_curBlock - 1)
            {
                vertexNum_forCurIterBlock = BLOCKSIZE_2;
            }
            else
            {
                vertexNum_forCurIterBlock = (vertexNum_inCurBlock % BLOCKSIZE_2) ? (vertexNum_inCurBlock % BLOCKSIZE_2) : BLOCKSIZE_2;
            }

            if (logical_threadId_forVertex < vertexNum_inCurBlock)
            {
                auto vid = frontier_in[firstVertexIndex_eachBlock + logical_threadId_forVertex];
                degree_type nbrSize = csr_offset[vid + 1] - csr_offset[vid];
                countl_type nbrStartIndex = csr_offset[vid];

                nbrSize_forCurThread = static_cast<countl_type>(nbrSize);
                nbr_offset_start[threadIdx.x] = nbrStartIndex;
                process_vertexId[threadIdx.x] = vid;

                if (!logical_threadId_forVertex)
                {
                    if (firstNbrIndex_InFirstVertex_inCurBlock)
                    {
                        nbrSize_forCurThread -= firstNbrIndex_InFirstVertex_inCurBlock;
                        nbr_offset_start[threadIdx.x] += firstNbrIndex_InFirstVertex_inCurBlock;
                    }
                }

                if (logical_threadId_forVertex == (vertexNum_inCurBlock - 1))
                {
                    countl_type last_select_val = lastNbrIndex_InLastVertex_inCurBlock + 1;
                    if (!(last_select_val == nbrSize))
                    {
                        nbrSize_forCurThread -= (nbrSize - last_select_val);
                    }
                }
            }

            countl_type totalDegree_forCurIterBlock;
            BlockScan_countl_type(temp_storage).ExclusiveSum(nbrSize_forCurThread, nbrSize_exsum_block[threadIdx.x], totalDegree_forCurIterBlock);
            sync_block();

            countl_type nbrSize_iters_block = (totalDegree_forCurIterBlock + BLOCKSIZE_2 - 1) / BLOCKSIZE_2;
            for (countl_type nbrSize_iter = 0; nbrSize_iter < nbrSize_iters_block; nbrSize_iter++)
            {
                countl_type logical_tid = nbrSize_iter * BLOCKSIZE_2 + threadIdx.x;
                uint16_t bsearch_idx = UINT16_MAX;
                if (logical_tid < totalDegree_forCurIterBlock)
                {
                    bsearch_idx = Device::binsearch(nbrSize_exsum_block, logical_tid, 0u, (vertexNum_forCurIterBlock - 1));
                    countl_type nbrIndex_inCurVertex = logical_tid - nbrSize_exsum_block[bsearch_idx];
                    vertex_id_type vid = process_vertexId[bsearch_idx];

                    const countl_type nbr_index = (nbr_offset_start[bsearch_idx] + nbrIndex_inCurVertex);
                    vertex_id_type dest = csr_dest[nbr_index];
                    edge_data_type weight = csr_weight[nbr_index];
                    edge_data_type msg = vertexValue[vid] + weight;
                    if (msg < vertexValue[dest])
                    {
                        if (msg < LinuxAtomic::CUDA::atomicMin(&vertexValue[dest], msg))
                        {
                            bool is_visited = (visitedBitset[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                            if (!is_visited)
                            {
                                int new_val = 1 << (dest % INT_SIZE);
                                uint32_t old_val = LinuxAtomic::CUDA::atomicOr(&visitedBitset[dest / INT_SIZE], new_val);
                                bool is_notsink = (notSinkBitset[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                if (is_notsink)
                                {
                                    bool atomic_was_visited = (old_val >> (dest % INT_SIZE)) & 1;
                                    if (!atomic_was_visited)
                                    {
                                        auto old_idx = LinuxAtomic::CUDA::atomicAdd(&frontierCounter_atomic_shared, 1);
                                        frontierOut_shared[old_idx] = dest;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        sync_block();
        if (!threadIdx.x)
        {
            global_offset = LinuxAtomic::CUDA::atomicAdd(frontierNum_next, static_cast<count_type>(frontierCounter_atomic_shared));
        }

        sync_block();
        count_type cand_iters = (frontierCounter_atomic_shared + BLOCKSIZE_2 - 1) / BLOCKSIZE_2;
        for (count_type i = 0; i < cand_iters; i++)
        {
            auto idx = i * BLOCKSIZE_2 + threadIdx.x;
            if (idx < frontierCounter_atomic_shared)
            {
                frontier_out[global_offset + idx] = frontierOut_shared[idx];
            }
        }
    }
}

void balance_model(const int& nBlock, vertex_data_type* __restrict__ vertexValue, const countl_type* __restrict__ csr_offset,
                   const vertex_id_type* __restrict__ csr_dest, const edge_data_type* __restrict__ csr_weight,
                   const vertex_id_type* __restrict__ csr_destWeight,                                         /* 基本数据 */
                   const vertex_id_type* __restrict__ frontier_in, vertex_id_type* __restrict__ frontier_out, /* frontier */
                   const count_type frontierNum, const countl_type totalDegree, count_type* frontierNum_next, /* frontier 常量信息 */
                   const countl_type* __restrict__ frontier_degExSum_, const countl_type* __restrict__ frontier_balance_device_,
                   const uint32_t* __restrict__ notSinkBitset, int* __restrict__ visitedBitset, const bool is_SSSP_destWeight)
{

    if (is_SSSP_destWeight)
    {
        CUDA_KERNEL_CALL(SSSP_balance_model_device_kernel_destWeight, nBlock, BLOCKSIZE_2, vertexValue, csr_offset, csr_destWeight, frontier_in,
                         frontier_out, frontierNum, totalDegree, frontierNum_next, frontier_degExSum_, frontier_balance_device_, notSinkBitset,
                         visitedBitset);
    }
    else
    {
        CUDA_KERNEL_CALL(SSSP_balance_model_device_kernel, nBlock, BLOCKSIZE_2, vertexValue, csr_offset, csr_dest, csr_weight, frontier_in,
                         frontier_out, frontierNum, totalDegree, frontierNum_next, frontier_degExSum_, frontier_balance_device_, notSinkBitset,
                         visitedBitset);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void SSSP_balance_model_device_kernel_destWeight_CPUGPU(vertex_data_type* __restrict__ vertexValue,
                                                                   const countl_type* __restrict__ csr_offset,
                                                                   const vertex_id_type* __restrict__ csr_destWeight, /* 基本数据 */
                                                                   const vertex_id_type* __restrict__ frontier_in, const count_type frontierNum,
                                                                   const countl_type totalDegree, const countl_type* __restrict__ frontier_degExSum_,
                                                                   const countl_type* __restrict__ frontier_balance_device,
                                                                   const uint32_t* __restrict__ notSinkBitset, int* __restrict__ visitedBitset)
{
    countl_type maxBlockNum = (totalDegree + gridDim.x * TASK_PER_BLOCK - 1) / (gridDim.x * TASK_PER_BLOCK);
    for (countl_type maxBlock_id = 0; maxBlock_id < maxBlockNum; maxBlock_id++)
    {
        countl_type logical_block_id = gridDim.x * maxBlock_id + blockIdx.x;
        countl_type firstEdgeIndex_eachBlock = logical_block_id * TASK_PER_BLOCK;
        if (firstEdgeIndex_eachBlock >= totalDegree) break;

        countl_type lastEdgeIndex_eachBlock =
            cuda_min(static_cast<countl_type>(totalDegree - 1), static_cast<countl_type>((logical_block_id + 1) * TASK_PER_BLOCK - 1));

        count_type firstVertexIndex_eachBlock = frontier_balance_device[logical_block_id] - 1;
        count_type lastVertexIndex_eachBlock;

        if (lastEdgeIndex_eachBlock < (totalDegree - 1))
        {
            lastVertexIndex_eachBlock = frontier_balance_device[logical_block_id + 1] - 1;
            if (frontier_degExSum_[lastVertexIndex_eachBlock] == lastEdgeIndex_eachBlock + 1) lastVertexIndex_eachBlock--;
        }
        else
        {
            lastVertexIndex_eachBlock = frontierNum - 1;
        }

        count_type vertexNum_inCurBlock = lastVertexIndex_eachBlock - firstVertexIndex_eachBlock + 1;

        countl_type firstNbrIndex_InFirstVertex_inCurBlock = firstEdgeIndex_eachBlock - frontier_degExSum_[firstVertexIndex_eachBlock];
        countl_type lastNbrIndex_InLastVertex_inCurBlock = lastEdgeIndex_eachBlock - frontier_degExSum_[lastVertexIndex_eachBlock];

        __shared__ countl_type nbr_offset_start[BLOCKSIZE_2];
        __shared__ vertex_id_type process_vertexId[BLOCKSIZE_2];
        __shared__ countl_type nbrSize_exsum_block[BLOCKSIZE_2];

        typedef cub::BlockScan<countl_type, BLOCKSIZE_2> BlockScan_countl_type;
        __shared__ typename BlockScan_countl_type::TempStorage temp_storage;

        const unsigned iterNumForVertex_curBlock = (vertexNum_inCurBlock + BLOCKSIZE_2 - 1) / BLOCKSIZE_2;
        for (unsigned iter_id_forVertex = 0; iter_id_forVertex < iterNumForVertex_curBlock; iter_id_forVertex++)
        {

            if (iter_id_forVertex) sync_block();

            unsigned logical_threadId_forVertex = iter_id_forVertex * BLOCKSIZE_2 + threadIdx.x;
            unsigned vertexNum_forCurIterBlock;
            countl_type nbrSize_forCurThread = 0;

            if (iter_id_forVertex != iterNumForVertex_curBlock - 1)
            {
                vertexNum_forCurIterBlock = BLOCKSIZE_2;
            }
            else
            {
                vertexNum_forCurIterBlock = (vertexNum_inCurBlock % BLOCKSIZE_2) ? (vertexNum_inCurBlock % BLOCKSIZE_2) : BLOCKSIZE_2;
            }

            if (logical_threadId_forVertex < vertexNum_inCurBlock)
            {
                auto vid = frontier_in[firstVertexIndex_eachBlock + logical_threadId_forVertex];
                degree_type nbrSize = csr_offset[vid + 1] - csr_offset[vid];
                countl_type nbrStartIndex = csr_offset[vid];

                nbrSize_forCurThread = static_cast<countl_type>(nbrSize);
                nbr_offset_start[threadIdx.x] = nbrStartIndex;
                process_vertexId[threadIdx.x] = vid;

                if (!logical_threadId_forVertex)
                {

                    if (firstNbrIndex_InFirstVertex_inCurBlock)
                    {
                        nbrSize_forCurThread -= firstNbrIndex_InFirstVertex_inCurBlock;
                        nbr_offset_start[threadIdx.x] += firstNbrIndex_InFirstVertex_inCurBlock;
                    }
                }

                if (logical_threadId_forVertex == (vertexNum_inCurBlock - 1))
                {
                    countl_type last_select_val = lastNbrIndex_InLastVertex_inCurBlock + 1;
                    if (!(last_select_val == nbrSize))
                    {
                        nbrSize_forCurThread -= (nbrSize - last_select_val);
                    }
                }
            }

            countl_type totalDegree_forCurIterBlock;
            BlockScan_countl_type(temp_storage).ExclusiveSum(nbrSize_forCurThread, nbrSize_exsum_block[threadIdx.x], totalDegree_forCurIterBlock);
            sync_block();

            countl_type nbrSize_iters_block = (totalDegree_forCurIterBlock + BLOCKSIZE_2 - 1) / BLOCKSIZE_2;
            for (countl_type nbrSize_iter = 0; nbrSize_iter < nbrSize_iters_block; nbrSize_iter++)
            {
                countl_type logical_tid = nbrSize_iter * BLOCKSIZE_2 + threadIdx.x;
                uint16_t bsearch_idx = UINT16_MAX;

                if (logical_tid < totalDegree_forCurIterBlock)
                {
                    bsearch_idx = Device::binsearch(nbrSize_exsum_block, logical_tid, 0u, (vertexNum_forCurIterBlock - 1));
                    countl_type nbrIndex_inCurVertex = logical_tid - nbrSize_exsum_block[bsearch_idx];
                    vertex_id_type vid = process_vertexId[bsearch_idx];

                    const countl_type nbr_index = (nbr_offset_start[bsearch_idx] + nbrIndex_inCurVertex) * 2;
                    vertex_id_type dest = csr_destWeight[nbr_index];
                    edge_data_type weight = csr_destWeight[nbr_index + 1];
                    edge_data_type msg = vertexValue[vid] + weight;
                    if (msg < vertexValue[dest])
                    {
                        if (msg < LinuxAtomic::CUDA::atomicMin(&vertexValue[dest], msg))
                        {
                            bool is_notSink = (notSinkBitset[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                            if (is_notSink)
                            {
                                bool is_visited = (visitedBitset[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                if (!is_visited)
                                {
                                    int new_val = 1 << (dest % INT_SIZE);
                                    LinuxAtomic::CUDA::atomicOr(&visitedBitset[dest / INT_SIZE], new_val);
                                }
                            }
                        }
                    }
                }
            }
        }

        sync_block();
    }
}

__global__ void SSSP_balance_model_device_kernel_CPUGPU(vertex_data_type* __restrict__ vertexValue, const countl_type* __restrict__ csr_offset,
                                                        const vertex_id_type* __restrict__ csr_dest,
                                                        const vertex_id_type* __restrict__ csr_weight,               /* 基本数据 */
                                                        const vertex_id_type* __restrict__ frontier_in,              /* frontier */
                                                        const count_type frontierNum, const countl_type totalDegree, /* frontier 常量信息 */
                                                        const countl_type* __restrict__ frontier_degExSum_,
                                                        const countl_type* __restrict__ frontier_balance_device,
                                                        const uint32_t* __restrict__ notSinkBitset, int* __restrict__ visitedBitset)
{
    /* 65536个Block 需要多少轮迭代*/
    countl_type maxBlockNum = (totalDegree + gridDim.x * TASK_PER_BLOCK - 1) / (gridDim.x * TASK_PER_BLOCK);
    for (countl_type maxBlock_id = 0; maxBlock_id < maxBlockNum; maxBlock_id++)
    {
        countl_type logical_block_id = gridDim.x * maxBlock_id + blockIdx.x;
        countl_type firstEdgeIndex_eachBlock = logical_block_id * TASK_PER_BLOCK;
        if (firstEdgeIndex_eachBlock >= totalDegree) break;

        countl_type lastEdgeIndex_eachBlock =
            cuda_min(static_cast<countl_type>(totalDegree - 1), static_cast<countl_type>((logical_block_id + 1) * TASK_PER_BLOCK - 1));

        count_type firstVertexIndex_eachBlock = frontier_balance_device[logical_block_id] - 1;
        count_type lastVertexIndex_eachBlock;

        if (lastEdgeIndex_eachBlock < (totalDegree - 1))
        {
            lastVertexIndex_eachBlock = frontier_balance_device[logical_block_id + 1] - 1;
            if (frontier_degExSum_[lastVertexIndex_eachBlock] == lastEdgeIndex_eachBlock + 1) lastVertexIndex_eachBlock--;
        }
        else
        {
            lastVertexIndex_eachBlock = frontierNum - 1;
        }

        count_type vertexNum_inCurBlock = lastVertexIndex_eachBlock - firstVertexIndex_eachBlock + 1;

        countl_type firstNbrIndex_InFirstVertex_inCurBlock = firstEdgeIndex_eachBlock - frontier_degExSum_[firstVertexIndex_eachBlock];
        countl_type lastNbrIndex_InLastVertex_inCurBlock = lastEdgeIndex_eachBlock - frontier_degExSum_[lastVertexIndex_eachBlock]; // 不对

        __shared__ countl_type nbr_offset_start[BLOCKSIZE_2];
        __shared__ vertex_id_type process_vertexId[BLOCKSIZE_2];
        __shared__ countl_type nbrSize_exsum_block[BLOCKSIZE_2];

        typedef cub::BlockScan<countl_type, BLOCKSIZE_2> BlockScan_countl_type;
        __shared__ typename BlockScan_countl_type::TempStorage temp_storage;

        const unsigned iterNumForVertex_curBlock = (vertexNum_inCurBlock + BLOCKSIZE_2 - 1) / BLOCKSIZE_2;
        for (unsigned iter_id_forVertex = 0; iter_id_forVertex < iterNumForVertex_curBlock; iter_id_forVertex++)
        {

            if (iter_id_forVertex) sync_block();

            unsigned logical_threadId_forVertex = iter_id_forVertex * BLOCKSIZE_2 + threadIdx.x;
            unsigned vertexNum_forCurIterBlock;
            countl_type nbrSize_forCurThread = 0;

            if (iter_id_forVertex != iterNumForVertex_curBlock - 1)
            {
                vertexNum_forCurIterBlock = BLOCKSIZE_2;
            }
            else
            {
                vertexNum_forCurIterBlock = (vertexNum_inCurBlock % BLOCKSIZE_2) ? (vertexNum_inCurBlock % BLOCKSIZE_2) : BLOCKSIZE_2;
            }

            if (logical_threadId_forVertex < vertexNum_inCurBlock)
            {
                auto vid = frontier_in[firstVertexIndex_eachBlock + logical_threadId_forVertex];
                degree_type nbrSize = csr_offset[vid + 1] - csr_offset[vid];
                countl_type nbrStartIndex = csr_offset[vid];

                nbrSize_forCurThread = static_cast<countl_type>(nbrSize);
                nbr_offset_start[threadIdx.x] = nbrStartIndex;
                process_vertexId[threadIdx.x] = vid;

                if (!logical_threadId_forVertex)
                {

                    if (firstNbrIndex_InFirstVertex_inCurBlock)
                    {
                        nbrSize_forCurThread -= firstNbrIndex_InFirstVertex_inCurBlock;
                        nbr_offset_start[threadIdx.x] += firstNbrIndex_InFirstVertex_inCurBlock;
                    }
                }

                if (logical_threadId_forVertex == (vertexNum_inCurBlock - 1))
                {
                    countl_type last_select_val = lastNbrIndex_InLastVertex_inCurBlock + 1;
                    if (!(last_select_val == nbrSize))
                    {
                        nbrSize_forCurThread -= (nbrSize - last_select_val);
                    }
                }
            }

            countl_type totalDegree_forCurIterBlock;
            BlockScan_countl_type(temp_storage).ExclusiveSum(nbrSize_forCurThread, nbrSize_exsum_block[threadIdx.x], totalDegree_forCurIterBlock);
            sync_block();

            countl_type nbrSize_iters_block = (totalDegree_forCurIterBlock + BLOCKSIZE_2 - 1) / BLOCKSIZE_2;
            for (countl_type nbrSize_iter = 0; nbrSize_iter < nbrSize_iters_block; nbrSize_iter++)
            {
                countl_type logical_tid = nbrSize_iter * BLOCKSIZE_2 + threadIdx.x;
                uint16_t bsearch_idx = UINT16_MAX;

                if (logical_tid < totalDegree_forCurIterBlock)
                {
                    bsearch_idx = Device::binsearch(nbrSize_exsum_block, logical_tid, 0u, (vertexNum_forCurIterBlock - 1));
                    countl_type nbrIndex_inCurVertex = logical_tid - nbrSize_exsum_block[bsearch_idx];
                    vertex_id_type vid = process_vertexId[bsearch_idx];

                    const countl_type nbr_index = (nbr_offset_start[bsearch_idx] + nbrIndex_inCurVertex);
                    vertex_id_type dest = csr_dest[nbr_index];
                    edge_data_type weight = csr_weight[nbr_index];
                    edge_data_type msg = vertexValue[vid] + weight;
                    if (msg < vertexValue[dest])
                    {
                        if (msg < LinuxAtomic::CUDA::atomicMin(&vertexValue[dest], msg))
                        {
                            bool is_notSink = (notSinkBitset[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                            if (is_notSink)
                            {
                                bool is_visited = (visitedBitset[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                if (!is_visited)
                                {
                                    int new_val = 1 << (dest % INT_SIZE);
                                    LinuxAtomic::CUDA::atomicOr(&visitedBitset[dest / INT_SIZE], new_val);
                                }
                            }
                        }
                    }
                }
            }
        }

        sync_block();
    }
}

void balance_model_CPUGPU(const int& nBlock, vertex_data_type* __restrict__ vertexValue, const countl_type* __restrict__ csr_offset,
                          const vertex_id_type* __restrict__ csr_dest, const edge_data_type* __restrict__ csr_weight,
                          const vertex_id_type* __restrict__ csr_destWeight,                                         /* 基本数据 */
                          const vertex_id_type* __restrict__ frontier_in, vertex_id_type* __restrict__ frontier_out, /* frontier */
                          const count_type frontierNum, const countl_type totalDegree, count_type* frontierNum_next, /* frontier 常量信息 */
                          const countl_type* __restrict__ frontier_degExSum_, const countl_type* __restrict__ frontier_balance_device_,
                          const uint32_t* __restrict__ notSinkBitset, int* __restrict__ visitedBitset, const bool is_SSSP_destWeight)
{

    if (is_SSSP_destWeight)
    {
        CUDA_KERNEL_CALL(SSSP_balance_model_device_kernel_destWeight_CPUGPU, nBlock, BLOCKSIZE_2, vertexValue, csr_offset, csr_destWeight,
                         frontier_in, frontierNum, totalDegree, frontier_degExSum_, frontier_balance_device_, notSinkBitset, visitedBitset);
    }
    else
    {
        CUDA_KERNEL_CALL(SSSP_balance_model_device_kernel_CPUGPU, nBlock, BLOCKSIZE_2, vertexValue, csr_offset, csr_dest, csr_weight, frontier_in,
                         frontierNum, totalDegree, frontier_degExSum_, frontier_balance_device_, notSinkBitset, visitedBitset);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void merge_vertexValue_kernel(vertex_data_type* __restrict__ vertexValue, vertex_data_type* __restrict__ vertexValue2,
                                         const count_type vertexNum, const uint32_t* __restrict__ notSinkBitset)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < vertexNum)
    {
        bool is_notSink = (notSinkBitset[tid / INT_SIZE] >> (tid % INT_SIZE)) & 1;
        if (!is_notSink)
        {
            vertexValue2[tid] = cuda_min(vertexValue[tid], vertexValue2[tid]);
        }
    }
}

struct Merge_vertexValue_type
{

    edge_data_type* vertexValue_device_{nullptr};
    edge_data_type* vertexValue_device2_{nullptr};
    const count_type vertexNum_host_{0};
    const uint32_t* notSinkBitset_device_{nullptr};

    Merge_vertexValue_type(edge_data_type* vertexValue_device, edge_data_type* vertexValue_device2, const count_type vertexNum_host,
                           uint32_t* notSinkBitset_device)
        : vertexValue_device_(vertexValue_device), vertexValue_device2_(vertexValue_device2), vertexNum_host_(vertexNum_host),
          notSinkBitset_device_(notSinkBitset_device)
    {
    }

    __device__ __forceinline__ void operator()(const count_type& thread_id)
    {
        bool is_notSink = (notSinkBitset_device_[thread_id / INT_SIZE] >> (thread_id % INT_SIZE)) & 1;
        if (!is_notSink)
        {
            vertexValue_device2_[thread_id] = cuda_min(vertexValue_device_[thread_id], vertexValue_device2_[thread_id]);
        }
    }
};

} // namespace SSSP_DEVICE_SPACE

namespace SSSP_HOST_SPACE {

vertex_id_type getFirst_bitset(const uint32_t* visitedBitset_storeTempDevice_host_, const count_type& vertexNum_host_)
{
    for (count_type vertex_id = 0; vertex_id < vertexNum_host_; vertex_id++)
    {
        uint32_t word32 = visitedBitset_storeTempDevice_host_[vertex_id];
        vertex_id_type vertex_id_start = vertex_id * INT_SIZE;
        while (word32 != 0)
        {
            if (word32 & 1)
            {
                return vertex_id_start;
            }
            vertex_id_start++;
            word32 = word32 >> 1;
        }
    }

    return vertexNum_host_;
}

vertex_id_type getLast_bitset(const uint32_t* visitedBitset_storeTempDevice_host_, const count_type& vertexNum_host_)
{
    for (int64_t i = vertexNum_host_ - 1; i >= 0; i--)
    {
        if (visitedBitset_storeTempDevice_host_[i] != 0)
        {
            for (int bit = 31; bit >= 0; --bit)
            {
                if ((visitedBitset_storeTempDevice_host_[i] >> bit) & 1)
                {
                    return i * 32 + bit;
                }
            }
        }
    }

    return 0;
}
} // namespace SSSP_HOST_SPACE

} // namespace CPJ