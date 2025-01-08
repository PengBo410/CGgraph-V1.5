#pragma once

#include "Basic/Bitmap/bitsOp_CPJ.hpp"
#include "Basic/CUDA/cuda_check.cuh"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Memory/alloc_CPJ.hpp"
#include "Basic/Memory/memInfo_CPJ.hpp"
#include "Basic/Other/scan_CPJ.hpp"
#include "Basic/Sort/OpenMPMergeSort.hpp"
#include "Basic/Thread/atomic_linux.hpp"
#include "Basic/Thread/omp_def.hpp"
#include "Basic/Timer/timer_CPJ.hpp"
#include "Basic/Type/data_type.hpp"
#include "basic_def.hpp"
#include "favor.hpp"
#include "flag.hpp"

#include <atomic>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <cub/cub.cuh>

namespace CPJ {

namespace Balance_BFS_OPT_V8 {

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

} // namespace Balance_BFS_OPT_V8

class CPU_BFS
{
  private:
    const count_type vertexNum_;
    const countl_type edgeNum_;

    const countl_type* csr_offset_{nullptr};
    const vertex_id_type* csr_dest_{nullptr};

    vertex_id_type* frontier_in_{nullptr};
    vertex_id_type* frontier_out_{nullptr};
    countl_type* frontier_degExSum_{nullptr};
    countl_type* frontier_degExSum_forPar_{nullptr};
    countl_type* frontier_balance_{nullptr};

    vertex_data_type* vertexValue_{nullptr};

    count_type frontierNum_in_{0};
    std::atomic<count_type> frontierNum_out_{0};
    countl_type* thread_edges_{nullptr};
    vertex_id_type** frontierOut_thread{nullptr};

    std::atomic<uint32_t>* visitedBitset_{nullptr};
    uint32_t* visitedBitset_curIte_{nullptr};
    const count_type visitedBitsetNum_{0};

    uint32_t* notSinkBitset_{nullptr};

    int level_{0};
    int ite_{0};

    Balance_BFS_OPT_V8::ThreadState_type* threadState_{nullptr};

    count_type* frontierAppend_thread_{nullptr};
    const count_type frontierAppendNum_thread_{0};
    static constexpr count_type CUSTOM_CHUNK_NUM_FRONTIER_APPEND_EACH_THREAD = 6;
    const count_type chunkNum_frontierAppend_eachThread{0};

    static constexpr bool ISSORT = true;
    static constexpr bool TEMP_TIME_DEBUG = false;
    static constexpr bool USED_STD_PAR = true;
    static constexpr int socketId_ = 0;
    static constexpr std::memory_order ATOMIC_ORDER = std::memory_order_relaxed; // std::memory_order_seq_cst;
    static constexpr int SINK_VERTEXVALUE_MODEL = 2;                             // 为0: 则不输出Result, 为1是一种模式, 为2是一种模式

  public:
    CPU_BFS(CSR_Result_type& csrResult)
        : vertexNum_(csrResult.vertexNum), edgeNum_(csrResult.edgeNum), csr_offset_(csrResult.csr_offset), csr_dest_(csrResult.csr_dest),
          visitedBitsetNum_((vertexNum_ + INT_SIZE - 1) / INT_SIZE),
          frontierAppendNum_thread_((ThreadNum * CUSTOM_CHUNK_NUM_FRONTIER_APPEND_EACH_THREAD) > visitedBitsetNum_
                                        ? ThreadNum
                                        : (ThreadNum * CUSTOM_CHUNK_NUM_FRONTIER_APPEND_EACH_THREAD)),
          chunkNum_frontierAppend_eachThread(frontierAppendNum_thread_ / ThreadNum)
    {
        CPJ::Timer time;
        vertexValue_ = CPJ::AllocMem::allocMem<vertex_data_type>(vertexNum_, socketId_);
        Msg_info("VertexValue alloc finish, Used time: %s", time.get_time_str().c_str());

        time.start();
        frontier_in_ = CPJ::AllocMem::allocMem<vertex_id_type>(vertexNum_, socketId_);
        std::fill(frontier_in_, frontier_in_ + vertexNum_, 0);
        frontier_out_ = CPJ::AllocMem::allocMem<vertex_id_type>(vertexNum_, socketId_);
        std::fill(frontier_out_, frontier_out_ + vertexNum_, 0);
        frontier_degExSum_ = CPJ::AllocMem::allocMem<countl_type>(vertexNum_, socketId_);
        std::fill(frontier_degExSum_, frontier_degExSum_ + vertexNum_, 0); //
        if constexpr (USED_STD_PAR)
        {
            frontier_degExSum_forPar_ = CPJ::AllocMem::allocMem<countl_type>(vertexNum_, socketId_);
            std::fill(frontier_degExSum_forPar_, frontier_degExSum_forPar_ + vertexNum_, 0);
        }
        frontier_balance_ = CPJ::AllocMem::allocMem<countl_type>(ThreadNum, socketId_);
        memset(frontier_balance_, 0, ThreadNum * sizeof(countl_type));
        Msg_info("Frontier alloc finish, Used time: %s", time.get_time_str().c_str());

        time.start();
        visitedBitset_ = CPJ::AllocMem::allocMem<std::atomic<uint32_t>>(visitedBitsetNum_, socketId_);
        memset(visitedBitset_, 0, visitedBitsetNum_ * sizeof(uint32_t));
        visitedBitset_curIte_ = CPJ::AllocMem::allocMem<uint32_t>(visitedBitsetNum_, socketId_);
        memset(visitedBitset_curIte_, 0, visitedBitsetNum_ * sizeof(uint32_t));
        Msg_info("VisitedBitset and VisitedBitset_curIte alloc finish, Used time: %s", time.get_time_str().c_str());

        time.start();
        notSinkBitset_ = CPJ::AllocMem::allocMem<uint32_t>(visitedBitsetNum_, socketId_);
        memset(notSinkBitset_, 0, visitedBitsetNum_ * sizeof(uint32_t));
        setSinkBitset();
        Msg_info("Set-SinkBitset finish, Used time: %s", time.get_time_str().c_str());

        time.start();
        thread_edges_ = CPJ::AllocMem::allocMem<countl_type>(ThreadNum + 1, socketId_);
        memset(thread_edges_, 0, (ThreadNum + 1) * sizeof(countl_type));
        Msg_info("Set-threadEdges finish, Used time: %s", time.get_time_str().c_str());

        time.start();
        frontierOut_thread = new vertex_id_type*[ThreadNum];
        for (int thread_id = 0; thread_id < ThreadNum; thread_id++)
        {
            frontierOut_thread[thread_id] = new vertex_id_type[HOST_THREAD_FLUSH_VALUE];
            memset(frontierOut_thread[thread_id], 0, HOST_THREAD_FLUSH_VALUE * sizeof(vertex_id_type));
        }
        Msg_info("Set-threadFrontierOut finish, Used time: %s", time.get_time_str().c_str());

        time.start();
        threadState_ = CPJ::AllocMem::allocMem<Balance_BFS_OPT_V8::ThreadState_type>(ThreadNum, socketId_);
        memset(threadState_, 0, sizeof(Balance_BFS_OPT_V8::ThreadState_type) * ThreadNum);
        Msg_info("Set-threadState finish, Used time: %s", time.get_time_str().c_str());

        time.start();
        frontierAppend_thread_ = CPJ::AllocMem::allocMem<count_type>(frontierAppendNum_thread_, socketId_);
        memset(frontierAppend_thread_, 0, sizeof(count_type) * frontierAppendNum_thread_);
        Msg_info("Set-frontierAppendNum_thread finish, Used time: %s", time.get_time_str().c_str());
    }

    void freeAssistCPUMemory()
    {
        if (vertexValue_ != nullptr)
        {
            CPJ::AllocMem::freeMem(vertexValue_, vertexNum_);
            vertexValue_ = nullptr;
        }
        if (frontier_in_ != nullptr)
        {
            CPJ::AllocMem::freeMem(frontier_in_, vertexNum_);
            frontier_in_ = nullptr;
        }
        if (frontier_out_ != nullptr)
        {
            CPJ::AllocMem::freeMem(frontier_out_, vertexNum_);
            frontier_out_ = nullptr;
        }
        if (frontier_degExSum_ != nullptr)
        {
            CPJ::AllocMem::freeMem(frontier_degExSum_, vertexNum_);
            frontier_degExSum_ = nullptr;
        }
        if (frontier_balance_ != nullptr)
        {
            CPJ::AllocMem::freeMem(frontier_balance_, ThreadNum);
            frontier_balance_ = nullptr;
        }
        if (notSinkBitset_ != nullptr)
        {
            CPJ::AllocMem::freeMem(notSinkBitset_, visitedBitsetNum_);
            notSinkBitset_ = nullptr;
        }
        if (thread_edges_ != nullptr)
        {
            CPJ::AllocMem::freeMem(thread_edges_, ThreadNum + 1);
            thread_edges_ = nullptr;
        }
        if (frontierOut_thread != nullptr)
        {
            for (int thread_id = 0; thread_id < ThreadNum; thread_id++) delete[] frontierOut_thread[thread_id];
            delete[] frontierOut_thread;
            frontierOut_thread = nullptr;
        }
        if (threadState_ != nullptr)
        {
            CPJ::AllocMem::freeMem(threadState_, ThreadNum);
            threadState_ = nullptr;
        }
        if (frontierAppend_thread_ != nullptr)
        {
            CPJ::AllocMem::freeMem(frontierAppend_thread_, frontierAppendNum_thread_);
            frontierAppend_thread_ = nullptr;
        }
    }
    ~CPU_BFS() { freeAssistCPUMemory(); }

    std::vector<SpeedRecord_type> measureBFS(int64_t root)
    {
        std::vector<SpeedRecord_type> iteRecord_vec;
        iteRecord_vec.reserve(1000);
        iteRecord_vec.clear();

        assert_msg(root >= 0 && root < vertexNum_, "Error root, root = %ld", root);
        if ((csr_offset_[root + 1] - csr_offset_[root]) == 0) return iteRecord_vec;

        std::fill(vertexValue_, vertexValue_ + vertexNum_, std::numeric_limits<vertex_data_type>::max());
        vertexValue_[root] = 0;

        frontier_in_[0] = root;
        frontierNum_in_ = 1;
        memset(visitedBitset_, 0, visitedBitsetNum_ * sizeof(uint32_t));
        memset(visitedBitset_curIte_, 0, visitedBitsetNum_ * sizeof(uint32_t));
        visitedBitset_[root / INT_SIZE] = 1U << (root % INT_SIZE);
        visitedBitset_curIte_[root / INT_SIZE] = 1U << (root % INT_SIZE);
        level_ = 0;
        ite_ = 0;
        CPJ::Timer totalTime;
        CPJ::Timer singleTime;
        CPJ::Timer tempTime;
        bool visitedBitset_increment_model = true;

        do
        {
            singleTime.start();
            ite_++;
            SpeedRecord_type speedRecord;
            speedRecord.iteId = ite_;

            setActiveDegree();
            countl_type degreeTotal = 0;
            if (USED_STD_PAR && (frontierNum_in_ > 10000))
            {
                degreeTotal = CPJ::SCAN::HOST::exclusive_scan_std_par(frontier_degExSum_, frontierNum_in_, frontier_degExSum_forPar_);
                std::swap(frontier_degExSum_forPar_, frontier_degExSum_);
            }
            else
            {
                degreeTotal = CPJ::SCAN::HOST::exclusive_scan_std(frontier_degExSum_, frontierNum_in_, frontier_degExSum_);
            }

            speedRecord.activeVertexNum = frontierNum_in_;
            speedRecord.activeEdgeNum = degreeTotal;

            visitedBitset_increment_model = false;
            if (degreeTotal > 1000000)
            {
                visitedBitset_increment_model = true;
                if constexpr (USED_STD_PAR) std::copy_n(std::execution::par_unseq, visitedBitset_, visitedBitsetNum_, visitedBitset_curIte_);
                else std::copy_n(visitedBitset_, visitedBitsetNum_, visitedBitset_curIte_);
            }

            countl_type threads_req = (degreeTotal + HOST_PER_THREAD_MIN_EDGES_TASKS - 1) / HOST_PER_THREAD_MIN_EDGES_TASKS;
            threads_req = std::min(threads_req, static_cast<countl_type>(ThreadNum));
            countl_type threads_max_avg = (degreeTotal + threads_req - 1) / threads_req;
            thread_edges_[0] = 0;
            for (int thread_id = 1; thread_id < threads_req; thread_id++)
            {
                thread_edges_[thread_id] = thread_edges_[thread_id - 1] + threads_max_avg;
            }
            thread_edges_[threads_req] = degreeTotal;
            thrust::upper_bound(thrust::host, frontier_degExSum_, frontier_degExSum_ + frontierNum_in_, thread_edges_, thread_edges_ + threads_req,
                                frontier_balance_);
            balance_model(threads_req, degreeTotal, visitedBitset_increment_model);

            if (visitedBitset_increment_model)
            {
                frontierNum_out_ = getVisitedIncAndSortedFrontierOut_Opt();
            }

            if (frontierNum_out_ == 0)
            {
                speedRecord.time_ms = singleTime.get_time_ms();
                speedRecord.total_time_ms = totalTime.get_time_ms();
                iteRecord_vec.push_back(speedRecord);
                Msg_node("[Complete]: CPU-BFS -> iteration: %2d, Used time: %5.2lf (ms)", ite_, speedRecord.total_time_ms);
                break;
            }

            if (ISSORT && !visitedBitset_increment_model)
            {
                if constexpr (USED_STD_PAR) //  && (frontierNum_out_ > 10000)
                {
                    std::sort(std::execution::par, frontier_out_, frontier_out_ + frontierNum_out_);
                }
                else
                {
                    count_type frontierNum_out_temp = frontierNum_out_;
                    CPJ::SORT::MergeSortRecursive::mergeSortRecursive_array(frontier_out_, frontierNum_out_temp);
                }
            }

            level_++;
            frontierNum_in_ = frontierNum_out_;
            frontierNum_out_ = 0;
            std::swap(frontier_in_, frontier_out_);

            speedRecord.time_ms = singleTime.get_time_ms();
            iteRecord_vec.push_back(speedRecord);
            // Msg_info("\t[CPU_BFS_OPT V8]: The (%2d) iteration, activeNum = %zu, BitsetInc = %s, Used time: = %7.2lf (ms)", ite_,
            //          SCU64(frontierNum_in_), (visitedBitset_increment_model ? "true" : "false"), singleTime.get_time_ms());
        }
        while (true);
        return iteRecord_vec;
    }

    vertex_data_type* getResult() { return vertexValue_; }

  private:
    void setSinkBitset()
    {
        omp_par_for(count_type bitSet_id = 0; bitSet_id < visitedBitsetNum_; bitSet_id++)
        {
            count_type vertexId = bitSet_id * INT_SIZE;
            uint32_t mask = 0;
#pragma unroll
            for (uint32_t bit_id = 0; bit_id < INT_SIZE; bit_id++)
            {
                uint32_t temp = (vertexId < vertexNum_) ? ((!(csr_offset_[vertexId + 1] - csr_offset_[vertexId])) ? 1 : 0) : 0;
                mask |= (temp << bit_id);
                vertexId++;
            }
            notSinkBitset_[bitSet_id] = (~mask);
        }
    }

    void setActiveDegree()
    {
        omp_par_for(count_type frontier_in_id = 0; frontier_in_id < frontierNum_in_; frontier_in_id++)
        {
            vertex_id_type vertexId = frontier_in_[frontier_in_id];
            countl_type degree = csr_offset_[vertexId + 1] - csr_offset_[vertexId];
            frontier_degExSum_[frontier_in_id] = degree;
        }
    }

    void balance_model(const int threads_req, const countl_type degreeTotal, const bool visitedBitset_increment_model)
    {
        if (visitedBitset_increment_model)
        {
            balance_model_inc(threads_req, degreeTotal);
        }
        else
        {
            balance_model_noInc(threads_req, degreeTotal);
        }
    }

    void balance_model_inc(const int threads_req, const countl_type degreeTotal)
    {

        omp_par_threads(threads_req)
        {
            const int threadId = omp_get_thread_num();
            const countl_type firstEdgeIndex_thread = thread_edges_[threadId];
            const countl_type lastEdgeIndex_thread = thread_edges_[threadId + 1] - 1;

            const count_type firstVertexIndex_thread = frontier_balance_[threadId] - 1;
            count_type lastVertexIndex_thread;
            if (lastEdgeIndex_thread < (degreeTotal - 1))
            {
                lastVertexIndex_thread = frontier_balance_[threadId + 1] - 1;
                if (frontier_degExSum_[lastVertexIndex_thread] == lastEdgeIndex_thread + 1) lastVertexIndex_thread--;
            }
            else
            {
                lastVertexIndex_thread = frontierNum_in_ - 1;
            }

            const count_type vertexNum_thread = lastVertexIndex_thread - firstVertexIndex_thread + 1;
            const degree_type firstNbrIndex_InFirstVertex_thread = firstEdgeIndex_thread - frontier_degExSum_[firstVertexIndex_thread];
            const degree_type lastNbrIndex_InLastVertex_thread = lastEdgeIndex_thread - frontier_degExSum_[lastVertexIndex_thread];

            threadState_[threadId].start = firstVertexIndex_thread;
            threadState_[threadId].cur = firstVertexIndex_thread;
            threadState_[threadId].end = firstVertexIndex_thread + vertexNum_thread;
            threadState_[threadId].firstNbrIndex_InFirstVertex_thread = firstNbrIndex_InFirstVertex_thread;
            threadState_[threadId].lastNbrIndex_InLastVertex_thread = lastNbrIndex_InLastVertex_thread;
            threadState_[threadId].status = Balance_BFS_OPT_V8::ThreadStatus::VERTEX_WORKING;

            while (true)
            {
                const count_type curAtomic_fetch_forVertexIndex = threadState_[threadId].cur.fetch_add(HOST_PER_THREAD_WORKING_VERTEX, ATOMIC_ORDER);
                if (curAtomic_fetch_forVertexIndex >= threadState_[threadId].end) break;
                const count_type curAtomic_fetch_forLastVertexIndex =
                    std::min(threadState_[threadId].end, (curAtomic_fetch_forVertexIndex + HOST_PER_THREAD_WORKING_VERTEX));
                for (count_type iter_id_forVertex = curAtomic_fetch_forVertexIndex; iter_id_forVertex < curAtomic_fetch_forLastVertexIndex;
                     iter_id_forVertex++)
                {
                    const vertex_id_type vid = frontier_in_[iter_id_forVertex];
                    countl_type nbrStartIndex = csr_offset_[vid];
                    const degree_type nbrSize = csr_offset_[vid + 1] - nbrStartIndex;
                    degree_type nbrSize_forCurVertex_thread = nbrSize;

                    if (iter_id_forVertex == threadState_[threadId].start)
                    {
                        nbrSize_forCurVertex_thread -= firstNbrIndex_InFirstVertex_thread;
                        nbrStartIndex += firstNbrIndex_InFirstVertex_thread;
                    }

                    if (iter_id_forVertex == (threadState_[threadId].end - 1))
                    {
                        nbrSize_forCurVertex_thread -= (nbrSize - lastNbrIndex_InLastVertex_thread - 1);
                    }

                    for (degree_type nbr_id = 0; nbr_id < nbrSize_forCurVertex_thread; nbr_id++)
                    {
                        vertex_id_type dest = csr_dest_[nbrStartIndex + nbr_id];
                        bool is_visited = (visitedBitset_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                        if (!is_visited)
                        {

                            if constexpr (SINK_VERTEXVALUE_MODEL == 1)
                            {
                                bool is_notSink = (notSinkBitset_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                if (!is_notSink) vertexValue_[dest] = level_ + 1;
                            }

                            int new_val = 1 << (dest % INT_SIZE);
                            visitedBitset_[dest / INT_SIZE].fetch_or(new_val, ATOMIC_ORDER);
                        }
                    }
                }
            }

            threadState_[threadId].status = Balance_BFS_OPT_V8::ThreadStatus::VERTEX_STEALING;
            for (int steal_offset = 1; steal_offset < threads_req; steal_offset++)
            {
                const int threadId_help = (threadId + steal_offset) % threads_req;
                while (threadState_[threadId_help].status != Balance_BFS_OPT_V8::ThreadStatus::VERTEX_STEALING)
                {
                    const count_type curAtomic_fetch_forVertexIndex =
                        threadState_[threadId_help].cur.fetch_add(HOST_PER_THREAD_STEALING_VERTEX, ATOMIC_ORDER);
                    if (curAtomic_fetch_forVertexIndex >= threadState_[threadId_help].end) break;
                    const count_type curAtomic_fetch_forLastVertexIndex =
                        std::min(threadState_[threadId_help].end, (curAtomic_fetch_forVertexIndex + HOST_PER_THREAD_STEALING_VERTEX));
                    for (count_type iter_id_forVertex = curAtomic_fetch_forVertexIndex; iter_id_forVertex < curAtomic_fetch_forLastVertexIndex;
                         iter_id_forVertex++)
                    {
                        const vertex_id_type vid = frontier_in_[iter_id_forVertex];
                        countl_type nbrStartIndex = csr_offset_[vid];
                        const degree_type nbrSize = csr_offset_[vid + 1] - nbrStartIndex;
                        degree_type nbrSize_forCurVertex_thread = nbrSize;

                        if (iter_id_forVertex == threadState_[threadId_help].start)
                        {
                            nbrSize_forCurVertex_thread -= threadState_[threadId_help].firstNbrIndex_InFirstVertex_thread;
                            nbrStartIndex += threadState_[threadId_help].firstNbrIndex_InFirstVertex_thread;
                        }

                        if (iter_id_forVertex == (threadState_[threadId_help].end - 1))
                        {
                            nbrSize_forCurVertex_thread -= (nbrSize - threadState_[threadId_help].lastNbrIndex_InLastVertex_thread - 1);
                        }

                        for (degree_type nbr_id = 0; nbr_id < nbrSize_forCurVertex_thread; nbr_id++)
                        {
                            vertex_id_type dest = csr_dest_[nbrStartIndex + nbr_id];
                            bool is_visited = (visitedBitset_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                            if (!is_visited)
                            {
                                if constexpr (SINK_VERTEXVALUE_MODEL == 1)
                                {
                                    bool is_notSink = (notSinkBitset_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                    if (!is_notSink) vertexValue_[dest] = level_ + 1;
                                }
                                int new_val = 1 << (dest % INT_SIZE);
                                visitedBitset_[dest / INT_SIZE].fetch_or(new_val, ATOMIC_ORDER);
                            }
                        }
                    }
                }
            }
        }
    }

    void balance_model_noInc(const int threads_req, const countl_type degreeTotal)
    {
        omp_par_threads(threads_req)
        {
            const int threadId = omp_get_thread_num();
            const countl_type firstEdgeIndex_thread = thread_edges_[threadId];
            const countl_type lastEdgeIndex_thread = thread_edges_[threadId + 1] - 1;

            const count_type firstVertexIndex_thread = frontier_balance_[threadId] - 1;
            count_type lastVertexIndex_thread;
            if (lastEdgeIndex_thread < (degreeTotal - 1))
            {
                lastVertexIndex_thread = frontier_balance_[threadId + 1] - 1;
                if (frontier_degExSum_[lastVertexIndex_thread] == lastEdgeIndex_thread + 1) lastVertexIndex_thread--;
            }
            else
            {
                lastVertexIndex_thread = frontierNum_in_ - 1;
            }

            const count_type vertexNum_thread = lastVertexIndex_thread - firstVertexIndex_thread + 1;
            const degree_type firstNbrIndex_InFirstVertex_thread = firstEdgeIndex_thread - frontier_degExSum_[firstVertexIndex_thread];
            const degree_type lastNbrIndex_InLastVertex_thread = lastEdgeIndex_thread - frontier_degExSum_[lastVertexIndex_thread];

            threadState_[threadId].start = firstVertexIndex_thread;
            threadState_[threadId].cur = firstVertexIndex_thread;
            threadState_[threadId].end = firstVertexIndex_thread + vertexNum_thread;
            threadState_[threadId].firstNbrIndex_InFirstVertex_thread = firstNbrIndex_InFirstVertex_thread;
            threadState_[threadId].lastNbrIndex_InLastVertex_thread = lastNbrIndex_InLastVertex_thread;
            threadState_[threadId].status = Balance_BFS_OPT_V8::ThreadStatus::VERTEX_WORKING;

            count_type frontierCounter_thread = 0;
            while (true)
            {
                const count_type curAtomic_fetch_forVertexIndex = threadState_[threadId].cur.fetch_add(HOST_PER_THREAD_WORKING_VERTEX, ATOMIC_ORDER);
                if (curAtomic_fetch_forVertexIndex >= threadState_[threadId].end) break;
                const count_type curAtomic_fetch_forLastVertexIndex =
                    std::min(threadState_[threadId].end, (curAtomic_fetch_forVertexIndex + HOST_PER_THREAD_WORKING_VERTEX));
                for (count_type iter_id_forVertex = curAtomic_fetch_forVertexIndex; iter_id_forVertex < curAtomic_fetch_forLastVertexIndex;
                     iter_id_forVertex++)
                {
                    const vertex_id_type vid = frontier_in_[iter_id_forVertex];
                    countl_type nbrStartIndex = csr_offset_[vid];
                    const degree_type nbrSize = csr_offset_[vid + 1] - nbrStartIndex;
                    degree_type nbrSize_forCurVertex_thread = nbrSize;

                    if (iter_id_forVertex == threadState_[threadId].start)
                    {
                        nbrSize_forCurVertex_thread -= firstNbrIndex_InFirstVertex_thread;
                        nbrStartIndex += firstNbrIndex_InFirstVertex_thread;
                    }

                    if (iter_id_forVertex == (threadState_[threadId].end - 1))
                    {
                        nbrSize_forCurVertex_thread -= (nbrSize - lastNbrIndex_InLastVertex_thread - 1);
                    }

                    for (degree_type nbr_id = 0; nbr_id < nbrSize_forCurVertex_thread; nbr_id++)
                    {
                        vertex_id_type dest = csr_dest_[nbrStartIndex + nbr_id];
                        bool is_visited = (visitedBitset_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                        if (!is_visited)
                        {
                            vertexValue_[dest] = level_ + 1;
                            int new_val = 1 << (dest % INT_SIZE);
                            uint32_t old_val = visitedBitset_[dest / INT_SIZE].fetch_or(new_val, ATOMIC_ORDER);
                            bool is_notSink = (notSinkBitset_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                            if (is_notSink)
                            {
                                bool atomic_was_visited = (old_val >> (dest % INT_SIZE)) & 1;
                                if (!atomic_was_visited)
                                {
                                    frontierOut_thread[threadId][frontierCounter_thread] = dest;
                                    frontierCounter_thread++;

                                    if (frontierCounter_thread == HOST_THREAD_FLUSH_VALUE)
                                    {
                                        count_type old_index = frontierNum_out_.fetch_add(frontierCounter_thread, ATOMIC_ORDER);
                                        std::memcpy(frontier_out_ + old_index, frontierOut_thread[threadId],
                                                    frontierCounter_thread * sizeof(vertex_id_type));
                                        frontierCounter_thread = 0;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            threadState_[threadId].status = Balance_BFS_OPT_V8::ThreadStatus::VERTEX_STEALING;
            for (int steal_offset = 1; steal_offset < threads_req; steal_offset++)
            {
                const int threadId_help = (threadId + steal_offset) % threads_req;
                while (threadState_[threadId_help].status != Balance_BFS_OPT_V8::ThreadStatus::VERTEX_STEALING)
                {
                    const count_type curAtomic_fetch_forVertexIndex =
                        threadState_[threadId_help].cur.fetch_add(HOST_PER_THREAD_STEALING_VERTEX, ATOMIC_ORDER);
                    if (curAtomic_fetch_forVertexIndex >= threadState_[threadId_help].end) break;
                    const count_type curAtomic_fetch_forLastVertexIndex =
                        std::min(threadState_[threadId_help].end, (curAtomic_fetch_forVertexIndex + HOST_PER_THREAD_STEALING_VERTEX));
                    for (count_type iter_id_forVertex = curAtomic_fetch_forVertexIndex; iter_id_forVertex < curAtomic_fetch_forLastVertexIndex;
                         iter_id_forVertex++)
                    {
                        const vertex_id_type vid = frontier_in_[iter_id_forVertex];
                        countl_type nbrStartIndex = csr_offset_[vid];
                        const degree_type nbrSize = csr_offset_[vid + 1] - nbrStartIndex;
                        degree_type nbrSize_forCurVertex_thread = nbrSize;

                        if (iter_id_forVertex == threadState_[threadId_help].start)
                        {
                            nbrSize_forCurVertex_thread -= threadState_[threadId_help].firstNbrIndex_InFirstVertex_thread;
                            nbrStartIndex += threadState_[threadId_help].firstNbrIndex_InFirstVertex_thread;
                        }

                        if (iter_id_forVertex == (threadState_[threadId_help].end - 1))
                        {
                            nbrSize_forCurVertex_thread -= (nbrSize - threadState_[threadId_help].lastNbrIndex_InLastVertex_thread - 1);
                        }

                        for (degree_type nbr_id = 0; nbr_id < nbrSize_forCurVertex_thread; nbr_id++)
                        {
                            vertex_id_type dest = csr_dest_[nbrStartIndex + nbr_id];
                            bool is_visited = (visitedBitset_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                            if (!is_visited)
                            {
                                vertexValue_[dest] = level_ + 1;

                                int new_val = 1 << (dest % INT_SIZE);
                                uint32_t old_val = visitedBitset_[dest / INT_SIZE].fetch_or(new_val, ATOMIC_ORDER);
                                bool is_notSink = (notSinkBitset_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                if (is_notSink)
                                {
                                    bool atomic_was_visited = (old_val >> (dest % INT_SIZE)) & 1;
                                    if (!atomic_was_visited)
                                    {
                                        frontierOut_thread[threadId][frontierCounter_thread] = dest;
                                        frontierCounter_thread++;

                                        if (frontierCounter_thread == HOST_THREAD_FLUSH_VALUE)
                                        {
                                            count_type old_index = frontierNum_out_.fetch_add(frontierCounter_thread, ATOMIC_ORDER);
                                            std::memcpy(frontier_out_ + old_index, frontierOut_thread[threadId],
                                                        frontierCounter_thread * sizeof(vertex_id_type));
                                            frontierCounter_thread = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (frontierCounter_thread > 0)
            {
                count_type old_index = frontierNum_out_.fetch_add(frontierCounter_thread, ATOMIC_ORDER);
                std::memcpy(frontier_out_ + old_index, frontierOut_thread[threadId], frontierCounter_thread * sizeof(vertex_id_type));
            }
        }
    }

    count_type getVisitedIncAndSortedFrontierOut_Opt()
    {
        count_type activeVertexNum = 0;
        memset(frontierAppend_thread_, 0, sizeof(count_type) * frontierAppendNum_thread_);
        const count_type chunkNum_avg = visitedBitsetNum_ / frontierAppendNum_thread_;

#ifdef getVisitedIncAndSortedFrontierOut_debug
        count_type checkSum = 0;
        omp_par_reductionAdd(checkSum)
#else
        omp_par
#endif
        {
#ifdef getVisitedIncAndSortedFrontierOut_debug
            CPJ::Timer singleTime;
#endif
            const int threadId = omp_get_thread_num();
            const int level_cur = level_ + 1;
            const count_type stratIndex_thread = threadId * chunkNum_frontierAppend_eachThread;

            for (count_type index = 0; index < chunkNum_frontierAppend_eachThread; index++)
            {
                const count_type chunId_first = chunkNum_avg * (stratIndex_thread + index);
                const count_type chunkId_last = ((stratIndex_thread + index) == (frontierAppendNum_thread_ - 1))
                                                    ? visitedBitsetNum_
                                                    : (((stratIndex_thread + index) + 1) * chunkNum_avg);
                for (count_type chunk_id = chunId_first; chunk_id < chunkId_last; chunk_id++)
                {
                    if (visitedBitset_curIte_[chunk_id] != visitedBitset_[chunk_id])
                    {
                        visitedBitset_curIte_[chunk_id] ^= visitedBitset_[chunk_id];
                        if (visitedBitset_curIte_[chunk_id] != 0)
                        {
                            if constexpr (SINK_VERTEXVALUE_MODEL == 2)
                            {
                                uint32_t onlyActiveSink = visitedBitset_curIte_[chunk_id] & (~notSinkBitset_[chunk_id]);
                                vertex_id_type vertex_id_start = chunk_id * INT_SIZE;
                                if (onlyActiveSink == std::numeric_limits<uint32_t>::max())
                                {
                                    std::fill_n(vertexValue_ + vertex_id_start, INT_SIZE, level_cur);
                                }
                                else
                                {
                                    while (onlyActiveSink != 0)
                                    {
                                        if (onlyActiveSink & 1)
                                        {
                                            vertexValue_[vertex_id_start] = level_cur;
                                        }
                                        vertex_id_start++;
                                        onlyActiveSink = onlyActiveSink >> 1;
                                    }
                                }
                            }

                            visitedBitset_curIte_[chunk_id] = visitedBitset_curIte_[chunk_id] & (notSinkBitset_[chunk_id]);
                            if (visitedBitset_curIte_[chunk_id] == std::numeric_limits<uint32_t>::max())
                            {
                                frontierAppend_thread_[stratIndex_thread + index] += INT_SIZE;
                            }
                            else if (visitedBitset_curIte_[chunk_id] != 0)
                            {
                                frontierAppend_thread_[stratIndex_thread + index] += CPJ::Bits::popcount_folly(visitedBitset_curIte_[chunk_id]);
                            }
                        }
                    }
                    else
                    {
                        visitedBitset_curIte_[chunk_id] = 0;
                    }
                }
            }

#pragma omp barrier

            /* 获取Balance */

#pragma omp single
            {
                activeVertexNum = CPJ::SCAN::HOST::exclusive_scan_std(frontierAppend_thread_, frontierAppendNum_thread_, frontierAppend_thread_);
            }

            count_type startIndex_thread = 0;
            if (activeVertexNum != 0)
            {
                /* 现在再构建Balance */
                const count_type avg_thread = activeVertexNum / ThreadNum;
                const count_type vertexStartIndex_expect_thread = threadId * avg_thread;
                const count_type vertexEndIndex_expect_thread = ((threadId == (ThreadNum - 1)) ? activeVertexNum : (threadId + 1) * avg_thread);

                const count_type start_balance_thread =
                    std::distance(frontierAppend_thread_, std::upper_bound(frontierAppend_thread_, frontierAppend_thread_ + frontierAppendNum_thread_,
                                                                           vertexStartIndex_expect_thread));
                const count_type end_balance_thread =
                    std::distance(frontierAppend_thread_, std::upper_bound(frontierAppend_thread_, frontierAppend_thread_ + frontierAppendNum_thread_,
                                                                           vertexEndIndex_expect_thread));

                count_type firstVertexIndex_thread = start_balance_thread - 1;
                count_type lastVertexIndex_thread;
                if (vertexEndIndex_expect_thread < (activeVertexNum))
                {
                    lastVertexIndex_thread = end_balance_thread - 1;
                    if (frontierAppend_thread_[lastVertexIndex_thread] == (vertexEndIndex_expect_thread + 1)) lastVertexIndex_thread--;
                }
                else
                {
                    lastVertexIndex_thread = frontierAppendNum_thread_ - 1;
                }

                const degree_type firstNbrIndex_InFirstVertex_thread =
                    vertexStartIndex_expect_thread - frontierAppend_thread_[firstVertexIndex_thread];

                if (firstNbrIndex_InFirstVertex_thread != 0)
                {
                    firstVertexIndex_thread++;
                }
                startIndex_thread = frontierAppend_thread_[firstVertexIndex_thread];

                const count_type chunId_first_in =
                    (firstVertexIndex_thread / chunkNum_frontierAppend_eachThread * chunkNum_frontierAppend_eachThread +
                     (firstVertexIndex_thread % chunkNum_frontierAppend_eachThread)) *
                    chunkNum_avg;

                const count_type chunkId_last_in =
                    (lastVertexIndex_thread == (frontierAppendNum_thread_ - 1))
                        ? (visitedBitsetNum_ - 1)
                        : (((lastVertexIndex_thread + 1) / chunkNum_frontierAppend_eachThread * chunkNum_frontierAppend_eachThread +
                            ((lastVertexIndex_thread + 1) % chunkNum_frontierAppend_eachThread)) *
                               chunkNum_avg -
                           1);

                for (count_type chunk_id = chunId_first_in; chunk_id <= chunkId_last_in; chunk_id++)
                {
                    uint32_t word32 = visitedBitset_curIte_[chunk_id];
                    vertex_id_type vertex_id_start = chunk_id * INT_SIZE;
                    if (word32 == std::numeric_limits<uint32_t>::max())
                    {
                        std::iota(frontier_out_ + startIndex_thread, frontier_out_ + startIndex_thread + INT_SIZE, vertex_id_start);
                        startIndex_thread += INT_SIZE;
                        if constexpr (SINK_VERTEXVALUE_MODEL)
                        {
                            std::fill(vertexValue_ + vertex_id_start, vertexValue_ + vertex_id_start + INT_SIZE, level_cur);
                        }
                    }
                    else if (word32 == 0x80000000)
                    {
                        vertex_id_type vertexId = vertex_id_start + 31;
                        frontier_out_[startIndex_thread] = vertexId;
                        startIndex_thread++;
                        if constexpr (SINK_VERTEXVALUE_MODEL) vertexValue_[vertexId] = level_cur;
                    }
                    else
                    {
                        while (word32 != 0)
                        {
                            if (word32 & 1)
                            {
                                frontier_out_[startIndex_thread] = vertex_id_start;
                                if constexpr (SINK_VERTEXVALUE_MODEL)
                                {
                                    vertexValue_[vertex_id_start] = level_cur;
                                }
                                startIndex_thread++;
                            }
                            vertex_id_start++;
                            word32 = word32 >> 1;
                        }
                    }
                }
            }
        }
        return activeVertexNum;
    }
};

} // namespace CPJ

namespace CPJ {

struct Set_activeDegree_type
{
    thrust::device_ptr<vertex_id_type> frontier_in_thrust_temp_{nullptr};
    thrust::device_ptr<countl_type> csr_offset_thrust_temp_{nullptr};
    thrust::device_ptr<countl_type> frontier_degExSum_thrust_temp_{nullptr};

    Set_activeDegree_type(thrust::device_ptr<vertex_id_type> frontier_in_thrust_temp, thrust::device_ptr<countl_type> csr_offset_thrust_temp,
                          thrust::device_ptr<countl_type> frontier_degExSum_thrust_temp)
        : frontier_in_thrust_temp_(frontier_in_thrust_temp), csr_offset_thrust_temp_(csr_offset_thrust_temp),
          frontier_degExSum_thrust_temp_(frontier_degExSum_thrust_temp)
    {
    }

    __device__ __forceinline__ void operator()(const vertex_id_type& idx)
    {
        vertex_id_type vid = frontier_in_thrust_temp_[idx];
        countl_type deg = csr_offset_thrust_temp_[vid + 1] - csr_offset_thrust_temp_[vid];
        frontier_degExSum_thrust_temp_[idx] = deg;
    }
};

struct Set_blockTask_type
{
    __device__ __forceinline__ countl_type operator()(const countl_type& blockId) { return (TASK_PER_BLOCK * blockId); }
};

struct Set_sink_type
{
    thrust::device_ptr<uint8_t> sink_bitset_{nullptr};
    thrust::device_ptr<countl_type> csr_offset_thrust_temp_{nullptr};
    const count_type vertexNum_;

    Set_sink_type(thrust::device_ptr<uint8_t> sink_bitset, thrust::device_ptr<countl_type> csr_offset_thrust_temp, count_type vertexNum)
        : sink_bitset_(sink_bitset), csr_offset_thrust_temp_(csr_offset_thrust_temp), vertexNum_(vertexNum)
    {
    }

    __device__ void operator()(const size_t& idx)
    {
        size_t vid = idx * 8;
        uint8_t mask = 0;
#pragma unroll
        for (uint8_t i = 0; i < 8; i++, vid++)
        {
            uint8_t tmp = (vid < vertexNum_) ? ((!(csr_offset_thrust_temp_[vid + 1] - csr_offset_thrust_temp_[vid])) ? 1 : 0) : 0;
            mask |= (tmp << i);
        }

        sink_bitset_[idx] = mask;
    }
};

namespace BFS_DEVICE_SPACE {
template <typename ValType, typename IndexType>
__device__ __forceinline__ IndexType search(const ValType* vec, const ValType val, IndexType low, IndexType high)
{
    while (true)
    {
        if (low == high) return low;
        if ((low + 1) == high) return (vec[high] <= val) ? high : low;

        IndexType mid = low + (high - low) / 2;

        if (vec[mid] > val) high = mid - 1;
        else low = mid;
    }
}

__global__ void balance_model_kernel(vertex_data_type* __restrict__ vertexValue, const countl_type* __restrict__ csr_offset,
                                     const vertex_id_type* __restrict__ csr_dest, const int level,                              /* 基本数据 */
                                     const vertex_id_type* __restrict__ frontier_in, vertex_id_type* __restrict__ frontier_out, /* frontier */
                                     const count_type frontierNum, const countl_type totalDegree,
                                     count_type* frontierNum_next, /* frontier 常量信息 */
                                     const countl_type* __restrict__ frontier_degExSum_, const countl_type* __restrict__ frontier_balance_device,
                                     const uint8_t* __restrict__ sinkBitset, int* __restrict__ visitedBitset)
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
                    bsearch_idx = search(nbrSize_exsum_block, logical_tid, 0u, (vertexNum_forCurIterBlock - 1));
                    countl_type nbrIndex_inCurVertex = logical_tid - nbrSize_exsum_block[bsearch_idx];

                    vertex_id_type dest = csr_dest[nbr_offset_start[bsearch_idx] + nbrIndex_inCurVertex];
                    bool is_visited = (visitedBitset[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                    bool is_sink = (sinkBitset[dest / 8] >> (dest % 8)) & 1;

                    if (!is_visited)
                    {
                        vertexValue[dest] = level + 1;
                        if (is_sink)
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

void balance_model(const int& nBlock, vertex_data_type* __restrict__ vertexValue, const countl_type* __restrict__ csr_offset,
                   const vertex_id_type* __restrict__ csr_dest, const int level,                              /* 基本数据 */
                   const vertex_id_type* __restrict__ frontier_in, vertex_id_type* __restrict__ frontier_out, /* frontier */
                   const count_type frontierNum, const countl_type totalDegree, count_type* frontierNum_next, /* frontier 常量信息 */
                   const countl_type* __restrict__ frontier_degExSum_, const countl_type* __restrict__ frontier_balance_device_,
                   const uint8_t* __restrict__ sinkBitset, int* __restrict__ visitedBitset)
{
    CUDA_KERNEL_CALL(balance_model_kernel, nBlock, BLOCKSIZE_2, vertexValue, csr_offset, csr_dest, level, frontier_in, frontier_out, frontierNum,
                     totalDegree, frontierNum_next, frontier_degExSum_, frontier_balance_device_, sinkBitset, visitedBitset);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
} // namespace BFS_DEVICE_SPACE

class GPU_BFS
{
  private:
    count_type vertexNum_{0};
    countl_type edgeNum_{0};

    countl_type* csr_offset_host_{nullptr};
    vertex_id_type* csr_dest_host_{nullptr};
    degree_type* outDegree_host_{nullptr};

    countl_type* csr_offset_device_{nullptr};
    vertex_id_type* csr_dest_device_{nullptr};
    thrust::device_ptr<countl_type> csr_offset_thrust_{nullptr};

    vertex_id_type* frontier_in_device_{nullptr};
    vertex_id_type* frontier_out_device_{nullptr};
    countl_type* frontier_degExSum_device_{nullptr};
    countl_type* frontier_balance_device_{nullptr};
    thrust::device_ptr<vertex_id_type> frontier_in_thrust_{nullptr};
    thrust::device_ptr<vertex_id_type> frontier_out_thrust_{nullptr};
    thrust::device_ptr<countl_type> frontier_degExSum_thrust_{nullptr};
    thrust::device_ptr<countl_type> frontier_balance_thrust_{nullptr};

    vertex_data_type* vertexValue_device_{nullptr};
    thrust::device_ptr<vertex_data_type> vertexValue_thrust_{nullptr};
    vertex_data_type* vertexValue_host_{nullptr};

    const bool is_sortFrontier_;
    static constexpr bool ISSORT = true;
    void* cub_temp_device_{nullptr};
    size_t cub_alloc_size_;
    count_type frontierNum{0};
    count_type* frontierNum_next_device_;
    thrust::device_ptr<count_type> frontierNum_next_thrust_{nullptr};
    int num_vertex_bits_;

    uint8_t* sinkBitset_device_{nullptr};
    uint8_t* visitedBitset_device_{nullptr};
    const size_t bitset_size_{0};

    int level_{0};
    int ite_{0};
    size_t nBlock{0};

    /* Debug */
    static constexpr bool TEMP_TIME_DEBUG = false;

  public:
    vertex_data_type* getResult()
    {
        CUDA_CHECK(MALLOC_HOST(&vertexValue_host_, vertexNum_));
        CUDA_CHECK(D2H(vertexValue_host_, vertexValue_device_, vertexNum_));
        return vertexValue_host_;
    }

  public:
    GPU_BFS(CSR_Result_type& csrResult_, GPU_memory_type gpuMemoryType = GPU_memory_type::GPU_MEM)
        : vertexNum_(csrResult_.vertexNum), edgeNum_(csrResult_.edgeNum), outDegree_host_(csrResult_.outDegree),
          csr_offset_host_(csrResult_.csr_offset), csr_dest_host_(csrResult_.csr_dest),
          bitset_size_(sizeof(int) * (vertexNum_ + INT_SIZE - 1) / INT_SIZE),
          is_sortFrontier_((vertexNum_ <= std::numeric_limits<int>::max()) && ISSORT), num_vertex_bits_((int)log2((float)vertexNum_) + 1)
    {
        CPJ::Timer time;

        CUDA_CHECK(cudaSetDevice(FLAGS_useDeviceId));

        time.start();
        CUDA_CHECK(MALLOC_DEVICE(&csr_offset_device_, vertexNum_ + 1));
        CUDA_CHECK(H2D(csr_offset_device_, csr_offset_host_, vertexNum_ + 1));
        if (gpuMemoryType == GPU_memory_type::GPU_MEM)
        {
            CUDA_CHECK(MALLOC_DEVICE(&csr_dest_device_, edgeNum_));
            CUDA_CHECK(H2D(csr_dest_device_, csr_dest_host_, edgeNum_));
            Msg_info("GPU Used [DevicePtr]");
        }
        else if (gpuMemoryType == GPU_memory_type::UVM)
        {
            CUDA_CHECK(cudaMallocManaged((void**)&csr_dest_device_, edgeNum_ * sizeof(vertex_id_type)));
            CUDA_CHECK(cudaMemAdvise(csr_dest_device_, edgeNum_ * sizeof(vertex_id_type), FLAG_UVM, fLI::FLAGS_useDeviceId));
            std::memcpy(csr_dest_device_, csr_dest_host_, sizeof(vertex_id_type) * edgeNum_);
            Msg_info("GPU Used [UVM]");
        }
        else if (gpuMemoryType == GPU_memory_type::ZERO_COPY)
        {
            CUDA_CHECK(cudaMallocManaged((void**)&csr_dest_device_, edgeNum_ * sizeof(vertex_id_type)));
            CUDA_CHECK(cudaMemAdvise(csr_dest_device_, edgeNum_ * sizeof(vertex_id_type), FLAG_ZERO_COPY, fLI::FLAGS_useDeviceId));
            std::memcpy(csr_dest_device_, csr_dest_host_, sizeof(vertex_id_type) * edgeNum_);
            Msg_info("GPU Used [ZERO-COPY]");
        }
        else
        {
            assert_msg(false, "Error cudaMemoryAdvise");
        }

        CUDA_CHECK(MALLOC_DEVICE(&frontier_in_device_, vertexNum_));
        CUDA_CHECK(MALLOC_DEVICE(&frontier_out_device_, vertexNum_));
        CUDA_CHECK(MALLOC_DEVICE(&frontier_degExSum_device_, vertexNum_));
        CUDA_CHECK(MALLOC_DEVICE(&vertexValue_device_, vertexNum_));
        CUDA_CHECK(MALLOC_DEVICE(&frontier_balance_device_, vertexNum_)); //! 是否会超出范围
        CUDA_CHECK(MALLOC_DEVICE(&frontierNum_next_device_, 1));
        CUDA_CHECK(cudaMalloc(&sinkBitset_device_, bitset_size_));
        CUDA_CHECK(cudaMalloc(&visitedBitset_device_, bitset_size_));
        Msg_info("MALLOC_DEVICE and H2D finish, used time: %s", time.get_time_str().c_str());

        time.start();
        csr_offset_thrust_ = thrust::device_pointer_cast(csr_offset_device_);
        frontier_in_thrust_ = thrust::device_pointer_cast(frontier_in_device_);
        thrust::fill(frontier_in_thrust_, frontier_in_thrust_ + vertexNum_, 0);
        frontier_out_thrust_ = thrust::device_pointer_cast(frontier_out_device_);
        thrust::fill(frontier_out_thrust_, frontier_out_thrust_ + vertexNum_, 0);
        frontier_degExSum_thrust_ = thrust::device_pointer_cast(frontier_degExSum_device_);
        thrust::fill(frontier_degExSum_thrust_, frontier_degExSum_thrust_ + vertexNum_, 0);
        frontier_balance_thrust_ = thrust::device_pointer_cast(frontier_balance_device_);
        vertexValue_thrust_ = thrust::device_pointer_cast(vertexValue_device_);
        frontierNum_next_thrust_ = thrust::device_pointer_cast(frontierNum_next_device_);
        Msg_info("Thrust::device_pointer_cast finish, used time: %s", time.get_time_str().c_str());

        time.start();
        setSinkBitset();
        Msg_info("setSinkBitset finish, used time: %s", time.get_time_str().c_str());
    }

    void freeGPUMemory()
    {
        if (csr_offset_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(csr_offset_device_));
            csr_offset_device_ = nullptr;
        }
        if (csr_dest_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(csr_dest_device_));
            csr_dest_device_ = nullptr;
        }
        if (frontier_in_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(frontier_in_device_));
            frontier_in_device_ = nullptr;
        }
        if (frontier_out_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(frontier_out_device_));
            frontier_out_device_ = nullptr;
        }
        if (frontier_degExSum_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(frontier_degExSum_device_));
            frontier_degExSum_device_ = nullptr;
        }
        if (vertexValue_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(vertexValue_device_));
            vertexValue_device_ = nullptr;
        }
        if (vertexValue_host_ != nullptr)
        {
            CUDA_CHECK(FREE_HOST(vertexValue_host_));
            vertexValue_host_ = nullptr;
        }
        if (frontier_balance_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(frontier_balance_device_));
            frontier_balance_device_ = nullptr;
        }
        if (frontierNum_next_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(frontierNum_next_device_));
            frontierNum_next_device_ = nullptr;
        }
        if (sinkBitset_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(sinkBitset_device_));
            sinkBitset_device_ = nullptr;
        }
        if (visitedBitset_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(visitedBitset_device_));
            visitedBitset_device_ = nullptr;
        }
        if (cub_temp_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(cub_temp_device_));
            cub_temp_device_ = nullptr;
        }
    }

    ~GPU_BFS() { freeGPUMemory(); }

    std::vector<SpeedRecord_type> measureBFS(int64_t root)
    {
        std::vector<SpeedRecord_type> iteRecord_vec;
        iteRecord_vec.reserve(1000);
        iteRecord_vec.clear();

        assert_msg(root >= 0 && root < vertexNum_, "Error root, root = %ld", root);

        if ((csr_offset_host_[root + 1] - csr_offset_host_[root]) == 0) return iteRecord_vec;

        if (is_sortFrontier_)
        {
            cub::DeviceRadixSort::SortKeys(cub_temp_device_, cub_alloc_size_, static_cast<const vertex_id_type*>(frontier_in_device_),
                                           static_cast<vertex_id_type*>(frontier_out_device_), (int)vertexNum_);
            CUDA_CHECK(cudaMalloc(&cub_temp_device_, cub_alloc_size_));
        }

        thrust::fill(vertexValue_thrust_, vertexValue_thrust_ + vertexNum_, std::numeric_limits<vertex_data_type>::max());
        vertexValue_thrust_[root] = 0;
        frontier_in_thrust_[0] = root;
        frontierNum = 1;
        level_ = 0;
        ite_ = 0;

        setSinkBitset();

        thrust::device_ptr<uint8_t> visitedBitset_thrust = thrust::device_pointer_cast(visitedBitset_device_);
        thrust::fill(visitedBitset_thrust, visitedBitset_thrust + bitset_size_, 0);
        visitedBitset_thrust[root / 8] = 1U << (root % 8); // active

        CPJ::Timer totalTime;
        CPJ::Timer singleTime;
        CPJ::Timer tempTime;
        do
        {
            singleTime.start();
            ite_++;
            SpeedRecord_type speedRecord;
            speedRecord.iteId = ite_;

            setActiveDegree();
            countl_type degreeTotal = getActiveExSum();

            speedRecord.activeVertexNum = frontierNum;
            speedRecord.activeEdgeNum = degreeTotal;

            nBlock = (degreeTotal + TASK_PER_BLOCK - 1) / TASK_PER_BLOCK;
            thrust::counting_iterator<countl_type> cnt_iter(0);
            auto query_iter_first = thrust::make_transform_iterator(cnt_iter, Set_blockTask_type());
            auto query_iter_last = thrust::make_transform_iterator(cnt_iter + nBlock, Set_blockTask_type());
            assert_msg(vertexNum_ > (query_iter_last - query_iter_first),
                       "frontier_balance_thrust_ 返回超标, vertexNum_ = %zu, (query_iter_last - query_iter_first) = %zu", SCU64(vertexNum_),
                       SCU64((query_iter_last - query_iter_first)));
            thrust::upper_bound(thrust::device, frontier_degExSum_thrust_, frontier_degExSum_thrust_ + frontierNum, query_iter_first, query_iter_last,
                                frontier_balance_thrust_);
            nBlock = std::min(nBlock, SCU64(MAX_BLOCKS));
            BFS_DEVICE_SPACE::balance_model(nBlock, vertexValue_device_, csr_offset_device_, csr_dest_device_, level_, frontier_in_device_,
                                            frontier_out_device_, frontierNum, degreeTotal, frontierNum_next_device_, frontier_degExSum_device_,
                                            frontier_balance_device_, sinkBitset_device_, reinterpret_cast<int*>(visitedBitset_device_));

            count_type frontierNum_next = (count_type)frontierNum_next_thrust_[0];
            if (frontierNum_next == 0)
            {
                speedRecord.time_ms = singleTime.get_time_ms();
                speedRecord.total_time_ms = totalTime.get_time_ms();
                iteRecord_vec.push_back(speedRecord);
                // Msg_node("[Complete]: BFS-Opt -> iteration: %2d, Used time:: %5.2lf (ms)", ite_, speedRecord.total_time_ms);
                // Msg_info("[GPU-BFS] process end, (%s) usedGPUMem = %.2lf (GB)", GPU_memory_type_name[SCI32(fLI::FLAGS_gpuMemory)],
                //          BYTES_TO_GB(CPJ::MemoryInfo::getMemoryTotal_Device(FLAGS_useDeviceId) -
                //                      CPJ::MemoryInfo::getMemoryFree_Device(FLAGS_useDeviceId)));
                break;
            }

            if (is_sortFrontier_)
            {
                int start_bit = static_cast<int>(0.35 * num_vertex_bits_);
                cub::DeviceRadixSort::SortKeys(cub_temp_device_, cub_alloc_size_, static_cast<const vertex_id_type*>(frontier_out_device_),
                                               static_cast<vertex_id_type*>(frontier_in_device_), (int)frontierNum_next, start_bit, num_vertex_bits_);
            }
            level_++;
            frontierNum = frontierNum_next;
            CUDA_CHECK(MEMSET_DEVICE(frontierNum_next_device_, 1));
            speedRecord.time_ms = singleTime.get_time_ms();
            iteRecord_vec.push_back(speedRecord);
        }
        while (true);

        return iteRecord_vec;
    }

  private:
    void setActiveDegree()
    {
        thrust::counting_iterator<vertex_id_type> iter_first(0);
        thrust::counting_iterator<vertex_id_type> iter_last = iter_first + frontierNum;
        thrust::for_each(iter_first, iter_last, Set_activeDegree_type(frontier_in_thrust_, csr_offset_thrust_, frontier_degExSum_thrust_));
    }

    countl_type getActiveExSum()
    {
        countl_type totalDegree_temp = frontier_degExSum_thrust_[frontierNum - 1];
        thrust::exclusive_scan(frontier_degExSum_thrust_, frontier_degExSum_thrust_ + frontierNum, frontier_degExSum_thrust_);
        return totalDegree_temp + frontier_degExSum_thrust_[frontierNum - 1];
    }

    void setSinkBitset()
    {
        thrust::device_ptr<uint8_t> sinkBitset_thrust = thrust::device_pointer_cast(sinkBitset_device_);
        thrust::counting_iterator<size_t> iter_first(0);
        thrust::counting_iterator<size_t> iter_last = iter_first + bitset_size_;
        thrust::for_each(iter_first, iter_last, Set_sink_type(sinkBitset_thrust, csr_offset_thrust_, vertexNum_));
    }
};
} // namespace CPJ

namespace CPJ {

namespace Balance_SSSP {

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
} // namespace Balance_SSSP

class CPU_SSSP
{
  private:
    const count_type vertexNum_{0};
    const countl_type edgeNum_{0};
    const countl_type* csr_offset_{nullptr};
    const vertex_id_type* csr_dest_{nullptr};
    const edge_data_type* csr_weight_{nullptr};
    vertex_id_type* csr_destWeight_{nullptr};
    const bool is_SSSP_destWeight_{false};

    vertex_id_type* frontier_in_{nullptr};
    vertex_id_type* frontier_out_{nullptr};
    countl_type* frontier_degExSum_{nullptr};
    countl_type* frontier_degExSum_forPar_{nullptr};
    countl_type* frontier_balance_{nullptr};
    vertex_data_type* vertexValue_{nullptr};

    count_type frontierNum_in_{0};
    std::atomic<count_type> frontierNum_out_{0};
    countl_type* thread_edges_{nullptr};
    vertex_id_type** frontierOut_thread_{nullptr};

    std::atomic<uint32_t>* visitedBitset_{nullptr};
    const count_type visitedBitsetNum_{0};

    uint32_t* notSinkBitset_{nullptr};

    int ite_{0};
    Balance_SSSP::ThreadState_type* threadState_{nullptr};

    count_type* frontierAppend_thread_{nullptr};
    const count_type frontierAppendNum_thread_{0};
    static constexpr count_type CUSTOM_CHUNK_NUM_FRONTIER_APPEND_EACH_THREAD = 6;
    const count_type chunkNum_frontierAppend_eachThread{0};

    static constexpr bool ISSORT = true;
    static constexpr bool TEMP_TIME_DEBUG = false;
    static constexpr bool USED_STD_PAR = true;
    static constexpr int socketId_ = 0;
    static constexpr std::memory_order ATOMIC_ORDER = std::memory_order_relaxed;

  public:
    CPU_SSSP(CSR_Result_type& csrResult, const bool is_SSSP_destWeight)
        : vertexNum_(csrResult.vertexNum), edgeNum_(csrResult.edgeNum), csr_offset_(csrResult.csr_offset), csr_dest_(csrResult.csr_dest),
          csr_weight_(csrResult.csr_weight), visitedBitsetNum_((vertexNum_ + INT_SIZE - 1) / INT_SIZE),
          frontierAppendNum_thread_((ThreadNum * CUSTOM_CHUNK_NUM_FRONTIER_APPEND_EACH_THREAD) > visitedBitsetNum_
                                        ? ThreadNum
                                        : (ThreadNum * CUSTOM_CHUNK_NUM_FRONTIER_APPEND_EACH_THREAD)),
          chunkNum_frontierAppend_eachThread(frontierAppendNum_thread_ / ThreadNum), is_SSSP_destWeight_(is_SSSP_destWeight)
    {
        CPJ::Timer time;
        if (is_SSSP_destWeight_)
        {
            time.start();
            constexpr bool isSame = std::is_same<vertex_id_type, edge_data_type>::value;
            if constexpr (!isSame) assert_msg(false, "[is_SSSP_destWeight_] need <vertex_id_type> same as <edge_data_type>");
            assert_msg((edgeNum_ * 2) < std::numeric_limits<countl_type>::max(), "(edgeNum_ * 2) large than [countl_type] max()");
            csr_destWeight_ = CPJ::AllocMem::allocMem_memset<vertex_id_type>(edgeNum_ * 2);
            omp_par_for(countl_type edge_id = 0; edge_id < edgeNum_; edge_id++)
            {
                csr_destWeight_[edge_id * 2] = csr_dest_[edge_id];
                csr_destWeight_[edge_id * 2 + 1] = csr_weight_[edge_id];
            }
            Msg_info("csr_destWeight_ alloc and set finish, Used time: %s", time.get_time_str().c_str());
        }

        time.start();
        vertexValue_ = CPJ::AllocMem::allocMem<vertex_data_type>(vertexNum_, socketId_);

        Msg_info("VertexValue alloc finish, Used time: %s", time.get_time_str().c_str());

        time.start();
        frontier_in_ = CPJ::AllocMem::allocMem<vertex_id_type>(vertexNum_, socketId_);
        std::fill(frontier_in_, frontier_in_ + vertexNum_, 0);
        frontier_out_ = CPJ::AllocMem::allocMem<vertex_id_type>(vertexNum_, socketId_);
        std::fill(frontier_out_, frontier_out_ + vertexNum_, 0);
        frontier_degExSum_ = CPJ::AllocMem::allocMem<countl_type>(vertexNum_, socketId_);
        std::fill(frontier_degExSum_, frontier_degExSum_ + vertexNum_, 0); //
        if constexpr (USED_STD_PAR)
        {
            frontier_degExSum_forPar_ = CPJ::AllocMem::allocMem<countl_type>(vertexNum_, socketId_);
            std::fill(frontier_degExSum_forPar_, frontier_degExSum_forPar_ + vertexNum_, 0);
        }
        frontier_balance_ = CPJ::AllocMem::allocMem<countl_type>(ThreadNum, socketId_);
        memset(frontier_balance_, 0, ThreadNum * sizeof(countl_type));
        Msg_info("Frontier alloc finish, Used time: %s", time.get_time_str().c_str());

        time.start();
        visitedBitset_ = CPJ::AllocMem::allocMem<std::atomic<uint32_t>>(visitedBitsetNum_, socketId_);
        memset(visitedBitset_, 0, visitedBitsetNum_ * sizeof(uint32_t));
        Msg_info("VisitedBitset and VisitedBitset_curIte alloc finish, Used time: %s", time.get_time_str().c_str());

        time.start();
        notSinkBitset_ = CPJ::AllocMem::allocMem<uint32_t>(visitedBitsetNum_, socketId_);
        memset(notSinkBitset_, 0, visitedBitsetNum_ * sizeof(uint32_t));
        setSinkBitset();
        Msg_info("Set-SinkBitset finish, Used time: %s", time.get_time_str().c_str());

        time.start();
        thread_edges_ = CPJ::AllocMem::allocMem<countl_type>(ThreadNum + 1, socketId_);
        memset(thread_edges_, 0, (ThreadNum + 1) * sizeof(countl_type));
        Msg_info("Set-threadEdges finish, Used time: %s", time.get_time_str().c_str());

        time.start();
        frontierOut_thread_ = new vertex_id_type*[ThreadNum];
        for (int thread_id = 0; thread_id < ThreadNum; thread_id++)
        {
            frontierOut_thread_[thread_id] = new vertex_id_type[HOST_THREAD_FLUSH_VALUE];
            memset(frontierOut_thread_[thread_id], 0, HOST_THREAD_FLUSH_VALUE * sizeof(vertex_id_type));
        }
        Msg_info("Set-threadFrontierOut finish, Used time: %s", time.get_time_str().c_str());

        time.start();
        threadState_ = CPJ::AllocMem::allocMem<Balance_SSSP::ThreadState_type>(ThreadNum, socketId_);
        memset(threadState_, 0, sizeof(Balance_SSSP::ThreadState_type) * ThreadNum);
        Msg_info("Set-threadState finish, Used time: %s", time.get_time_str().c_str());

        time.start();
        frontierAppend_thread_ = CPJ::AllocMem::allocMem<count_type>(frontierAppendNum_thread_, socketId_);
        memset(frontierAppend_thread_, 0, sizeof(count_type) * frontierAppendNum_thread_);
        Msg_info("Set-frontierAppendNum_thread finish, Used time: %s", time.get_time_str().c_str());
    }

    ~CPU_SSSP() { freeAssistCPUMemory(); }

    std::vector<SpeedRecord_type> measureSSSP(int64_t root)
    {
        std::vector<SpeedRecord_type> iteRecord_vec;
        iteRecord_vec.reserve(1000);
        iteRecord_vec.clear();

        assert_msg(root >= 0 && root < vertexNum_, "Error root, root = %ld", root);
        if ((csr_offset_[root + 1] - csr_offset_[root]) == 0) return iteRecord_vec;

        std::fill(vertexValue_, vertexValue_ + vertexNum_, std::numeric_limits<vertex_data_type>::max());
        vertexValue_[root] = 0;

        frontier_in_[0] = root;
        frontierNum_in_ = 1;
        memset(visitedBitset_, 0, visitedBitsetNum_ * sizeof(uint32_t));
        visitedBitset_[root / INT_SIZE] = 1U << (root % INT_SIZE);

        ite_ = 0;
        CPJ::Timer totalTime;
        CPJ::Timer singleTime;
        CPJ::Timer tempTime;
        bool visitedBitset_increment_model = true;

        do
        {
            singleTime.start();
            ite_++;
            SpeedRecord_type speedRecord;
            speedRecord.iteId = ite_;

            setActiveDegree();
            countl_type degreeTotal = 0;
            if (USED_STD_PAR && (frontierNum_in_ > 10000))
            {
                degreeTotal = CPJ::SCAN::HOST::exclusive_scan_std_par(frontier_degExSum_, frontierNum_in_, frontier_degExSum_forPar_);
                std::swap(frontier_degExSum_forPar_, frontier_degExSum_);
            }
            else
            {
                degreeTotal = CPJ::SCAN::HOST::exclusive_scan_std(frontier_degExSum_, frontierNum_in_, frontier_degExSum_);
            }

            if constexpr (USED_STD_PAR) std::fill(std::execution::par_unseq, visitedBitset_, visitedBitset_ + visitedBitsetNum_, 0);
            else std::memset(visitedBitset_, 0, visitedBitsetNum_ * sizeof(uint32_t));

            speedRecord.activeVertexNum = frontierNum_in_;
            speedRecord.activeEdgeNum = degreeTotal;

            visitedBitset_increment_model = false;
            if (degreeTotal > 1000000) visitedBitset_increment_model = true;

            countl_type threads_req = (degreeTotal + HOST_PER_THREAD_MIN_EDGES_TASKS - 1) / HOST_PER_THREAD_MIN_EDGES_TASKS;
            threads_req = std::min(threads_req, static_cast<countl_type>(ThreadNum));
            countl_type threads_max_avg = (degreeTotal + threads_req - 1) / threads_req;
            thread_edges_[0] = 0;
            for (int thread_id = 1; thread_id < threads_req; thread_id++)
            {
                thread_edges_[thread_id] = thread_edges_[thread_id - 1] + threads_max_avg;
            }
            thread_edges_[threads_req] = degreeTotal;
            thrust::upper_bound(thrust::host, frontier_degExSum_, frontier_degExSum_ + frontierNum_in_, thread_edges_, thread_edges_ + threads_req,
                                frontier_balance_);
            balance_model(threads_req, degreeTotal, visitedBitset_increment_model);
            if (visitedBitset_increment_model)
            {
                frontierNum_out_ = getVisitedIncAndSortedFrontierOut_Opt();
            }

            if (frontierNum_out_ == 0)
            {
                speedRecord.time_ms = singleTime.get_time_ms();
                speedRecord.total_time_ms = totalTime.get_time_ms();
                iteRecord_vec.push_back(speedRecord);
                Msg_node("\t[Complete]: CPU_SSSP -> iteration: %2d, Used time: %5.2lf (ms)", ite_, speedRecord.total_time_ms);
                break;
            }

            if (ISSORT && !visitedBitset_increment_model)
            {
                if constexpr (USED_STD_PAR)
                {
                    std::sort(std::execution::par, frontier_out_, frontier_out_ + frontierNum_out_);
                }
                else
                {
                    std::sort(frontier_out_, frontier_out_ + frontierNum_out_);
                }
                auto it = std::unique(std::execution::par, frontier_out_, frontier_out_ + frontierNum_out_);
                assert_msg(std::distance(frontier_out_, it) == frontierNum_out_, "前 = %zu, 后 = %zu", SCU64(frontierNum_out_),
                           SCU64(std::distance(frontier_out_, it)));
            }
            frontierNum_in_ = frontierNum_out_;
            frontierNum_out_ = 0;
            std::swap(frontier_in_, frontier_out_);

            speedRecord.time_ms = singleTime.get_time_ms();
            iteRecord_vec.push_back(speedRecord);

            // Msg_info("\t[CPU_SSSP: The (%2d) iteration, activeNum = %zu, BitsetInc = %s, Used time: = %7.2lf (ms)", ite_, SCU64(frontierNum_in_),
            //          (visitedBitset_increment_model ? "true" : "false"), singleTime.get_time_ms());
        }
        while (true);

        return iteRecord_vec;
    }

    vertex_data_type* getResult() { return vertexValue_; }

    void balance_model(const int threads_req, const countl_type degreeTotal, const bool visitedBitset_increment_model)
    {
        if (visitedBitset_increment_model)
        {
            balance_model_inc(threads_req, degreeTotal);
        }
        else
        {
            balance_model_noInc(threads_req, degreeTotal);
        }
    }

    void balance_model_inc(const int threads_req, const countl_type degreeTotal)
    {

        omp_par_threads(threads_req)
        {
            const int threadId = omp_get_thread_num();
            const countl_type firstEdgeIndex_thread = thread_edges_[threadId];
            const countl_type lastEdgeIndex_thread = thread_edges_[threadId + 1] - 1;

            const count_type firstVertexIndex_thread = frontier_balance_[threadId] - 1;
            count_type lastVertexIndex_thread;
            if (lastEdgeIndex_thread < (degreeTotal - 1))
            {
                lastVertexIndex_thread = frontier_balance_[threadId + 1] - 1;
                if (frontier_degExSum_[lastVertexIndex_thread] == lastEdgeIndex_thread + 1) lastVertexIndex_thread--;
            }
            else
            {
                lastVertexIndex_thread = frontierNum_in_ - 1;
            }

            const count_type vertexNum_thread = lastVertexIndex_thread - firstVertexIndex_thread + 1;
            const degree_type firstNbrIndex_InFirstVertex_thread = firstEdgeIndex_thread - frontier_degExSum_[firstVertexIndex_thread];
            const degree_type lastNbrIndex_InLastVertex_thread = lastEdgeIndex_thread - frontier_degExSum_[lastVertexIndex_thread];

            threadState_[threadId].start = firstVertexIndex_thread;
            threadState_[threadId].cur = firstVertexIndex_thread;
            threadState_[threadId].end = firstVertexIndex_thread + vertexNum_thread;
            threadState_[threadId].firstNbrIndex_InFirstVertex_thread = firstNbrIndex_InFirstVertex_thread;
            threadState_[threadId].lastNbrIndex_InLastVertex_thread = lastNbrIndex_InLastVertex_thread;
            threadState_[threadId].status = Balance_SSSP::ThreadStatus::VERTEX_WORKING;

            while (true)
            {
                const count_type curAtomic_fetch_forVertexIndex = threadState_[threadId].cur.fetch_add(HOST_PER_THREAD_WORKING_VERTEX, ATOMIC_ORDER);
                if (curAtomic_fetch_forVertexIndex >= threadState_[threadId].end) break;
                const count_type curAtomic_fetch_forLastVertexIndex =
                    std::min(threadState_[threadId].end, (curAtomic_fetch_forVertexIndex + HOST_PER_THREAD_WORKING_VERTEX));
                for (count_type iter_id_forVertex = curAtomic_fetch_forVertexIndex; iter_id_forVertex < curAtomic_fetch_forLastVertexIndex;
                     iter_id_forVertex++)
                {
                    const vertex_id_type vid = frontier_in_[iter_id_forVertex];
                    countl_type nbrStartIndex = csr_offset_[vid];
                    const degree_type nbrSize = csr_offset_[vid + 1] - nbrStartIndex;
                    degree_type nbrSize_forCurVertex_thread = nbrSize;

                    if (iter_id_forVertex == threadState_[threadId].start)
                    {
                        nbrSize_forCurVertex_thread -= firstNbrIndex_InFirstVertex_thread;
                        nbrStartIndex += firstNbrIndex_InFirstVertex_thread;
                    }

                    if (iter_id_forVertex == (threadState_[threadId].end - 1))
                    {
                        nbrSize_forCurVertex_thread -= (nbrSize - lastNbrIndex_InLastVertex_thread - 1);
                    }

                    for (degree_type nbr_id = 0; nbr_id < nbrSize_forCurVertex_thread; nbr_id++)
                    {
                        countl_type nbr_index = (nbrStartIndex + nbr_id);
                        vertex_id_type dest;
                        edge_data_type weight;
                        if (is_SSSP_destWeight_)
                        {
                            nbr_index = nbr_index * 2;
                            dest = csr_destWeight_[nbr_index];
                            weight = csr_destWeight_[nbr_index + 1];
                        }
                        else
                        {
                            dest = csr_dest_[nbr_index];
                            weight = csr_weight_[nbr_index];
                        }
                        edge_data_type msg = vertexValue_[vid] + weight;
                        if (msg < vertexValue_[dest])
                        {
                            if (LinuxAtomic::write_min(&vertexValue_[dest], msg))
                            {
                                bool is_visited = (visitedBitset_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                if (!is_visited)
                                {
                                    int new_val = 1 << (dest % INT_SIZE);
                                    visitedBitset_[dest / INT_SIZE].fetch_or(new_val, ATOMIC_ORDER);
                                }
                            }
                        }
                    }
                }
            }

            threadState_[threadId].status = Balance_SSSP::ThreadStatus::VERTEX_STEALING;
            for (int steal_offset = 1; steal_offset < threads_req; steal_offset++)
            {
                const int threadId_help = (threadId + steal_offset) % threads_req;
                while (threadState_[threadId_help].status != Balance_SSSP::ThreadStatus::VERTEX_STEALING)
                {
                    const count_type curAtomic_fetch_forVertexIndex =
                        threadState_[threadId_help].cur.fetch_add(HOST_PER_THREAD_STEALING_VERTEX, ATOMIC_ORDER);
                    if (curAtomic_fetch_forVertexIndex >= threadState_[threadId_help].end) break;
                    const count_type curAtomic_fetch_forLastVertexIndex =
                        std::min(threadState_[threadId_help].end, (curAtomic_fetch_forVertexIndex + HOST_PER_THREAD_STEALING_VERTEX));
                    for (count_type iter_id_forVertex = curAtomic_fetch_forVertexIndex; iter_id_forVertex < curAtomic_fetch_forLastVertexIndex;
                         iter_id_forVertex++)
                    {
                        const vertex_id_type vid = frontier_in_[iter_id_forVertex];
                        countl_type nbrStartIndex = csr_offset_[vid];
                        const degree_type nbrSize = csr_offset_[vid + 1] - nbrStartIndex;
                        degree_type nbrSize_forCurVertex_thread = nbrSize;

                        if (iter_id_forVertex == threadState_[threadId_help].start)
                        {
                            nbrSize_forCurVertex_thread -= threadState_[threadId_help].firstNbrIndex_InFirstVertex_thread;
                            nbrStartIndex += threadState_[threadId_help].firstNbrIndex_InFirstVertex_thread;
                        }

                        if (iter_id_forVertex == (threadState_[threadId_help].end - 1))
                        {
                            nbrSize_forCurVertex_thread -= (nbrSize - threadState_[threadId_help].lastNbrIndex_InLastVertex_thread - 1);
                        }

                        for (degree_type nbr_id = 0; nbr_id < nbrSize_forCurVertex_thread; nbr_id++)
                        {
                            countl_type nbr_index = (nbrStartIndex + nbr_id);
                            vertex_id_type dest;
                            edge_data_type weight;
                            if (is_SSSP_destWeight_)
                            {
                                nbr_index = nbr_index * 2;
                                dest = csr_destWeight_[nbr_index];
                                weight = csr_destWeight_[nbr_index + 1];
                            }
                            else
                            {
                                dest = csr_dest_[nbr_index];
                                weight = csr_weight_[nbr_index];
                            }

                            edge_data_type msg = vertexValue_[vid] + weight;
                            if (msg < vertexValue_[dest])
                            {
                                if (LinuxAtomic::write_min(&vertexValue_[dest], msg))
                                {
                                    bool is_visited = (visitedBitset_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                    if (!is_visited)
                                    {
                                        int new_val = 1 << (dest % INT_SIZE);
                                        visitedBitset_[dest / INT_SIZE].fetch_or(new_val, ATOMIC_ORDER);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void balance_model_noInc(const int threads_req, const countl_type degreeTotal)
    {
        omp_par_threads(threads_req)
        {
            const int threadId = omp_get_thread_num();
            const countl_type firstEdgeIndex_thread = thread_edges_[threadId];
            const countl_type lastEdgeIndex_thread = thread_edges_[threadId + 1] - 1;

            const count_type firstVertexIndex_thread = frontier_balance_[threadId] - 1;
            count_type lastVertexIndex_thread;
            if (lastEdgeIndex_thread < (degreeTotal - 1))
            {
                lastVertexIndex_thread = frontier_balance_[threadId + 1] - 1;
                if (frontier_degExSum_[lastVertexIndex_thread] == lastEdgeIndex_thread + 1) lastVertexIndex_thread--;
            }
            else
            {
                lastVertexIndex_thread = frontierNum_in_ - 1;
            }

            const count_type vertexNum_thread = lastVertexIndex_thread - firstVertexIndex_thread + 1;
            const degree_type firstNbrIndex_InFirstVertex_thread = firstEdgeIndex_thread - frontier_degExSum_[firstVertexIndex_thread];
            const degree_type lastNbrIndex_InLastVertex_thread = lastEdgeIndex_thread - frontier_degExSum_[lastVertexIndex_thread];

            threadState_[threadId].start = firstVertexIndex_thread;
            threadState_[threadId].cur = firstVertexIndex_thread;
            threadState_[threadId].end = firstVertexIndex_thread + vertexNum_thread;
            threadState_[threadId].firstNbrIndex_InFirstVertex_thread = firstNbrIndex_InFirstVertex_thread;
            threadState_[threadId].lastNbrIndex_InLastVertex_thread = lastNbrIndex_InLastVertex_thread;
            threadState_[threadId].status = Balance_SSSP::ThreadStatus::VERTEX_WORKING;

            count_type frontierCounter_thread = 0;
            while (true)
            {
                const count_type curAtomic_fetch_forVertexIndex = threadState_[threadId].cur.fetch_add(HOST_PER_THREAD_WORKING_VERTEX, ATOMIC_ORDER);
                if (curAtomic_fetch_forVertexIndex >= threadState_[threadId].end) break;
                const count_type curAtomic_fetch_forLastVertexIndex =
                    std::min(threadState_[threadId].end, (curAtomic_fetch_forVertexIndex + HOST_PER_THREAD_WORKING_VERTEX));
                for (count_type iter_id_forVertex = curAtomic_fetch_forVertexIndex; iter_id_forVertex < curAtomic_fetch_forLastVertexIndex;
                     iter_id_forVertex++)
                {
                    const vertex_id_type vid = frontier_in_[iter_id_forVertex];
                    countl_type nbrStartIndex = csr_offset_[vid];
                    const degree_type nbrSize = csr_offset_[vid + 1] - nbrStartIndex;
                    degree_type nbrSize_forCurVertex_thread = nbrSize;

                    if (iter_id_forVertex == threadState_[threadId].start)
                    {
                        nbrSize_forCurVertex_thread -= firstNbrIndex_InFirstVertex_thread;
                        nbrStartIndex += firstNbrIndex_InFirstVertex_thread;
                    }

                    if (iter_id_forVertex == (threadState_[threadId].end - 1))
                    {
                        nbrSize_forCurVertex_thread -= (nbrSize - lastNbrIndex_InLastVertex_thread - 1);
                    }

                    for (degree_type nbr_id = 0; nbr_id < nbrSize_forCurVertex_thread; nbr_id++)
                    {
                        /* 获取到Dest */
                        countl_type nbr_index = (nbrStartIndex + nbr_id);
                        vertex_id_type dest;
                        edge_data_type weight;
                        if (is_SSSP_destWeight_)
                        {
                            nbr_index = nbr_index * 2;
                            dest = csr_destWeight_[nbr_index];
                            weight = csr_destWeight_[nbr_index + 1];
                        }
                        else
                        {
                            dest = csr_dest_[nbr_index];
                            weight = csr_weight_[nbr_index];
                        }

                        edge_data_type msg = vertexValue_[vid] + weight;
                        if (msg < vertexValue_[dest])
                        {
                            if (LinuxAtomic::write_min(&vertexValue_[dest], msg))

                            {
                                bool is_visited = (visitedBitset_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                if (!is_visited)
                                {
                                    int new_val = 1 << (dest % INT_SIZE);
                                    uint32_t old_val = visitedBitset_[dest / INT_SIZE].fetch_or(new_val, ATOMIC_ORDER);
                                    bool is_notSink = (notSinkBitset_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                    if (is_notSink)
                                    {
                                        bool atomic_was_visited = (old_val >> (dest % INT_SIZE)) & 1;
                                        if (!atomic_was_visited)
                                        {
                                            frontierOut_thread_[threadId][frontierCounter_thread] = dest;
                                            frontierCounter_thread++;

                                            if (frontierCounter_thread == HOST_THREAD_FLUSH_VALUE)
                                            {
                                                count_type old_index = frontierNum_out_.fetch_add(frontierCounter_thread, ATOMIC_ORDER);
                                                std::memcpy(frontier_out_ + old_index, frontierOut_thread_[threadId],
                                                            frontierCounter_thread * sizeof(vertex_id_type));
                                                frontierCounter_thread = 0;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            threadState_[threadId].status = Balance_SSSP::ThreadStatus::VERTEX_STEALING;
            for (int steal_offset = 1; steal_offset < threads_req; steal_offset++)
            {
                const int threadId_help = (threadId + steal_offset) % threads_req;
                while (threadState_[threadId_help].status != Balance_SSSP::ThreadStatus::VERTEX_STEALING)
                {
                    const count_type curAtomic_fetch_forVertexIndex =
                        threadState_[threadId_help].cur.fetch_add(HOST_PER_THREAD_STEALING_VERTEX, ATOMIC_ORDER);
                    if (curAtomic_fetch_forVertexIndex >= threadState_[threadId_help].end) break;
                    const count_type curAtomic_fetch_forLastVertexIndex =
                        std::min(threadState_[threadId_help].end, (curAtomic_fetch_forVertexIndex + HOST_PER_THREAD_STEALING_VERTEX));
                    for (count_type iter_id_forVertex = curAtomic_fetch_forVertexIndex; iter_id_forVertex < curAtomic_fetch_forLastVertexIndex;
                         iter_id_forVertex++)
                    {
                        const vertex_id_type vid = frontier_in_[iter_id_forVertex];
                        countl_type nbrStartIndex = csr_offset_[vid];
                        const degree_type nbrSize = csr_offset_[vid + 1] - nbrStartIndex;
                        degree_type nbrSize_forCurVertex_thread = nbrSize;

                        if (iter_id_forVertex == threadState_[threadId_help].start)
                        {
                            nbrSize_forCurVertex_thread -= threadState_[threadId_help].firstNbrIndex_InFirstVertex_thread;
                            nbrStartIndex += threadState_[threadId_help].firstNbrIndex_InFirstVertex_thread;
                        }

                        if (iter_id_forVertex == (threadState_[threadId_help].end - 1))
                        {
                            nbrSize_forCurVertex_thread -= (nbrSize - threadState_[threadId_help].lastNbrIndex_InLastVertex_thread - 1);
                        }
                        for (degree_type nbr_id = 0; nbr_id < nbrSize_forCurVertex_thread; nbr_id++)
                        {
                            countl_type nbr_index = (nbrStartIndex + nbr_id);
                            vertex_id_type dest;
                            edge_data_type weight;
                            if (is_SSSP_destWeight_)
                            {
                                nbr_index = nbr_index * 2;
                                dest = csr_destWeight_[nbr_index];
                                weight = csr_destWeight_[nbr_index + 1];
                            }
                            else
                            {
                                dest = csr_dest_[nbr_index];
                                weight = csr_weight_[nbr_index];
                            }

                            edge_data_type msg = vertexValue_[vid] + weight;
                            if (msg < vertexValue_[dest])
                            {
                                if (LinuxAtomic::write_min(&vertexValue_[dest], msg))
                                {
                                    bool is_visited = (visitedBitset_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                    if (!is_visited)
                                    {
                                        int new_val = 1 << (dest % INT_SIZE);
                                        uint32_t old_val = visitedBitset_[dest / INT_SIZE].fetch_or(new_val, ATOMIC_ORDER);
                                        bool is_notSink = (notSinkBitset_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                        if (is_notSink)
                                        {
                                            bool atomic_was_visited = (old_val >> (dest % INT_SIZE)) & 1;
                                            if (!atomic_was_visited)
                                            {
                                                frontierOut_thread_[threadId][frontierCounter_thread] = dest;
                                                frontierCounter_thread++;

                                                if (frontierCounter_thread == HOST_THREAD_FLUSH_VALUE)
                                                {
                                                    count_type old_index = frontierNum_out_.fetch_add(frontierCounter_thread, ATOMIC_ORDER);
                                                    std::memcpy(frontier_out_ + old_index, frontierOut_thread_[threadId],
                                                                frontierCounter_thread * sizeof(vertex_id_type));
                                                    frontierCounter_thread = 0;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (frontierCounter_thread > 0)
            {
                count_type old_index = frontierNum_out_.fetch_add(frontierCounter_thread, ATOMIC_ORDER);
                std::memcpy(frontier_out_ + old_index, frontierOut_thread_[threadId], frontierCounter_thread * sizeof(vertex_id_type));
            }
        }
    }

  private:
    void setSinkBitset()
    {
        omp_par_for(count_type bitSet_id = 0; bitSet_id < visitedBitsetNum_; bitSet_id++)
        {
            count_type vertexId = bitSet_id * INT_SIZE;
            uint32_t mask = 0;
#pragma unroll
            for (uint32_t bit_id = 0; bit_id < INT_SIZE; bit_id++)
            {
                uint32_t temp = (vertexId < vertexNum_) ? ((!(csr_offset_[vertexId + 1] - csr_offset_[vertexId])) ? 1 : 0) : 0;
                mask |= (temp << bit_id);
                vertexId++;
            }
            notSinkBitset_[bitSet_id] = (~mask);
        }
    }

    void setActiveDegree()
    {
        omp_par_for(count_type frontier_in_id = 0; frontier_in_id < frontierNum_in_; frontier_in_id++)
        {
            vertex_id_type vertexId = frontier_in_[frontier_in_id];
            countl_type degree = csr_offset_[vertexId + 1] - csr_offset_[vertexId];
            frontier_degExSum_[frontier_in_id] = degree;
        }
    }

  public:
    void freeSSSP_destWeight()
    {
        if (csr_destWeight_ != nullptr)
        {
            CPJ::AllocMem::freeMem(csr_destWeight_, edgeNum_ * 2);
            csr_destWeight_ = nullptr;
        }
    }

    void freeAssistCPUMemory()
    {

        if (vertexValue_ != nullptr)
        {
            CPJ::AllocMem::freeMem(vertexValue_, vertexNum_);
            vertexValue_ = nullptr;
        }

        if (frontier_in_ != nullptr)
        {
            CPJ::AllocMem::freeMem(frontier_in_, vertexNum_);
            frontier_in_ = nullptr;
        }
        if (frontier_out_ != nullptr)
        {
            CPJ::AllocMem::freeMem(frontier_out_, vertexNum_);
            frontier_out_ = nullptr;
        }
        if (frontier_degExSum_ != nullptr)
        {
            CPJ::AllocMem::freeMem(frontier_degExSum_, vertexNum_);
            frontier_degExSum_ = nullptr;
        }
        if (frontier_balance_ != nullptr)
        {
            CPJ::AllocMem::freeMem(frontier_balance_, ThreadNum);
            frontier_balance_ = nullptr;
        }
        if (notSinkBitset_ != nullptr)
        {
            CPJ::AllocMem::freeMem(notSinkBitset_, visitedBitsetNum_);
            notSinkBitset_ = nullptr;
        }
        if (thread_edges_ != nullptr)
        {
            CPJ::AllocMem::freeMem(thread_edges_, ThreadNum + 1);
            thread_edges_ = nullptr;
        }
        if (frontierOut_thread_ != nullptr)
        {
            for (int thread_id = 0; thread_id < ThreadNum; thread_id++) delete[] frontierOut_thread_[thread_id];
            delete[] frontierOut_thread_;
            frontierOut_thread_ = nullptr;
        }
        if (threadState_ != nullptr)
        {
            CPJ::AllocMem::freeMem(threadState_, ThreadNum);
            threadState_ = nullptr;
        }
        if (frontierAppend_thread_ != nullptr)
        {
            CPJ::AllocMem::freeMem(frontierAppend_thread_, frontierAppendNum_thread_);
            frontierAppend_thread_ = nullptr;
        }
    }

  private:
    count_type getVisitedIncAndSortedFrontierOut_Opt()
    {
        count_type activeVertexNum = 0;
        memset(frontierAppend_thread_, 0, sizeof(count_type) * frontierAppendNum_thread_);
        const count_type chunkNum_avg = visitedBitsetNum_ / frontierAppendNum_thread_;

        omp_par
        {
            const int threadId = omp_get_thread_num();
            const count_type stratIndex_thread = threadId * chunkNum_frontierAppend_eachThread;
            for (count_type index = 0; index < chunkNum_frontierAppend_eachThread; index++)
            {
                const count_type chunId_first = chunkNum_avg * (stratIndex_thread + index);
                const count_type chunkId_last = ((stratIndex_thread + index) == (frontierAppendNum_thread_ - 1))
                                                    ? visitedBitsetNum_
                                                    : (((stratIndex_thread + index) + 1) * chunkNum_avg);
                for (count_type chunk_id = chunId_first; chunk_id < chunkId_last; chunk_id++)
                {
                    if (visitedBitset_[chunk_id] != 0)
                    {
                        visitedBitset_[chunk_id] = visitedBitset_[chunk_id] & (notSinkBitset_[chunk_id]);
                        if (visitedBitset_[chunk_id] == std::numeric_limits<uint32_t>::max())
                        {
                            frontierAppend_thread_[stratIndex_thread + index] += INT_SIZE;
                        }
                        else if (visitedBitset_[chunk_id] != 0)
                        {
                            uint32_t temp = visitedBitset_[chunk_id];
                            frontierAppend_thread_[stratIndex_thread + index] += CPJ::Bits::popcount_folly(temp);
                        }
                    }
                }
            }
#pragma omp barrier

#pragma omp single
            {
                activeVertexNum = CPJ::SCAN::HOST::exclusive_scan_std(frontierAppend_thread_, frontierAppendNum_thread_, frontierAppend_thread_);
            }

            count_type startIndex_thread = 0;
            if (activeVertexNum != 0)
            {
                const count_type avg_thread = activeVertexNum / ThreadNum;
                const count_type vertexStartIndex_expect_thread = threadId * avg_thread;
                const count_type vertexEndIndex_expect_thread = ((threadId == (ThreadNum - 1)) ? activeVertexNum : (threadId + 1) * avg_thread);

                const count_type start_balance_thread =
                    std::distance(frontierAppend_thread_, std::upper_bound(frontierAppend_thread_, frontierAppend_thread_ + frontierAppendNum_thread_,
                                                                           vertexStartIndex_expect_thread));
                const count_type end_balance_thread =
                    std::distance(frontierAppend_thread_, std::upper_bound(frontierAppend_thread_, frontierAppend_thread_ + frontierAppendNum_thread_,
                                                                           vertexEndIndex_expect_thread));

                count_type firstVertexIndex_thread = start_balance_thread - 1;
                count_type lastVertexIndex_thread;
                if (vertexEndIndex_expect_thread < (activeVertexNum))
                {
                    lastVertexIndex_thread = end_balance_thread - 1;
                    if (frontierAppend_thread_[lastVertexIndex_thread] == (vertexEndIndex_expect_thread + 1)) lastVertexIndex_thread--;
                }
                else
                {
                    lastVertexIndex_thread = frontierAppendNum_thread_ - 1;
                }

                const degree_type firstNbrIndex_InFirstVertex_thread =
                    vertexStartIndex_expect_thread - frontierAppend_thread_[firstVertexIndex_thread];

                if (firstNbrIndex_InFirstVertex_thread != 0)
                {
                    firstVertexIndex_thread++;
                }

                startIndex_thread = frontierAppend_thread_[firstVertexIndex_thread];
                const count_type chunId_first_in =
                    (firstVertexIndex_thread / chunkNum_frontierAppend_eachThread * chunkNum_frontierAppend_eachThread +
                     (firstVertexIndex_thread % chunkNum_frontierAppend_eachThread)) *
                    chunkNum_avg;

                const count_type chunkId_last_in =
                    (lastVertexIndex_thread == (frontierAppendNum_thread_ - 1))
                        ? (visitedBitsetNum_ - 1)
                        : (((lastVertexIndex_thread + 1) / chunkNum_frontierAppend_eachThread * chunkNum_frontierAppend_eachThread +
                            ((lastVertexIndex_thread + 1) % chunkNum_frontierAppend_eachThread)) *
                               chunkNum_avg -
                           1);

                for (count_type chunk_id = chunId_first_in; chunk_id <= chunkId_last_in; chunk_id++)
                {
                    uint32_t word32 = visitedBitset_[chunk_id];
                    vertex_id_type vertex_id_start = chunk_id * INT_SIZE;
                    if (word32 == std::numeric_limits<uint32_t>::max())
                    {
                        std::iota(frontier_out_ + startIndex_thread, frontier_out_ + startIndex_thread + INT_SIZE, vertex_id_start);
                        startIndex_thread += INT_SIZE;
                    }
                    else if (word32 == 0x80000000)
                    {
                        vertex_id_type vertexId = vertex_id_start + 31;
                        frontier_out_[startIndex_thread] = vertexId;
                        startIndex_thread++;
                    }
                    else
                    {
                        while (word32 != 0)
                        {
                            if (word32 & 1)
                            {
                                frontier_out_[startIndex_thread] = vertex_id_start;
                                startIndex_thread++;
                            }
                            vertex_id_start++;
                            word32 = word32 >> 1;
                        }
                    }
                }
            }
        }
        return activeVertexNum;
    }
}; // end of class [CPU_SSSP]

} // namespace CPJ

namespace CPJ {

namespace SSSP_DEVICE_SPACE {

struct Set_sink_type
{
    thrust::device_ptr<uint8_t> sink_bitset_{nullptr};
    thrust::device_ptr<countl_type> csr_offset_thrust_temp_{nullptr};
    const count_type vertexNum_;

    Set_sink_type(thrust::device_ptr<uint8_t> sink_bitset, thrust::device_ptr<countl_type> csr_offset_thrust_temp, count_type vertexNum)
        : sink_bitset_(sink_bitset), csr_offset_thrust_temp_(csr_offset_thrust_temp), vertexNum_(vertexNum)
    {
    }

    __device__ void operator()(const size_t& idx)
    {
        size_t vid = idx * 8;
        uint8_t mask = 0;
#pragma unroll
        for (uint8_t i = 0; i < 8; i++, vid++)
        {
            uint8_t tmp = (vid < vertexNum_) ? ((!(csr_offset_thrust_temp_[vid + 1] - csr_offset_thrust_temp_[vid])) ? 1 : 0) : 0;
            mask |= (tmp << i);
        }

        sink_bitset_[idx] = mask;
    }
};

struct Set_activeDegree_type
{
    thrust::device_ptr<vertex_id_type> frontier_in_thrust_temp_{nullptr};
    thrust::device_ptr<countl_type> csr_offset_thrust_temp_{nullptr};
    thrust::device_ptr<countl_type> frontier_degExSum_thrust_temp_{nullptr};

    Set_activeDegree_type(thrust::device_ptr<vertex_id_type> frontier_in_thrust_temp, thrust::device_ptr<countl_type> csr_offset_thrust_temp,
                          thrust::device_ptr<countl_type> frontier_degExSum_thrust_temp)
        : frontier_in_thrust_temp_(frontier_in_thrust_temp), csr_offset_thrust_temp_(csr_offset_thrust_temp),
          frontier_degExSum_thrust_temp_(frontier_degExSum_thrust_temp)
    {
    }

    __device__ __forceinline__ void operator()(const vertex_id_type& idx)
    {
        vertex_id_type vid = frontier_in_thrust_temp_[idx];
        countl_type deg = csr_offset_thrust_temp_[vid + 1] - csr_offset_thrust_temp_[vid];
        frontier_degExSum_thrust_temp_[idx] = deg;
    }
};

struct Set_blockTask_type
{
    __device__ __forceinline__ countl_type operator()(const countl_type& blockId) { return (TASK_PER_BLOCK * blockId); }
};

template <typename ValType, typename IndexType>
__device__ __forceinline__ IndexType search_sssp(const ValType* vec, const ValType val, IndexType low, IndexType high)
{
    while (true)
    {
        if (low == high) return low; // we know it exists
        if ((low + 1) == high) return (vec[high] <= val) ? high : low;

        IndexType mid = low + (high - low) / 2;

        if (vec[mid] > val) high = mid - 1;
        else low = mid;
    }
}

__global__ void SSSP_balance_model_device_kernel_destWeight(vertex_data_type* __restrict__ vertexValue, const countl_type* __restrict__ csr_offset,
                                                            const vertex_id_type* __restrict__ csr_destWeight, /* 基本数据 */
                                                            const vertex_id_type* __restrict__ frontier_in,
                                                            vertex_id_type* __restrict__ frontier_out, /* frontier */
                                                            const count_type frontierNum, const countl_type totalDegree,
                                                            count_type* frontierNum_next, /* frontier 常量信息 */
                                                            const countl_type* __restrict__ frontier_degExSum_,
                                                            const countl_type* __restrict__ frontier_balance_device,
                                                            const uint8_t* __restrict__ sinkBitset, int* __restrict__ visitedBitset)
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
        countl_type lastNbrIndex_InLastVertex_inCurBlock = lastEdgeIndex_eachBlock - frontier_degExSum_[lastVertexIndex_eachBlock];

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
                    bsearch_idx = search_sssp(nbrSize_exsum_block, logical_tid, 0u, (vertexNum_forCurIterBlock - 1));
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
                                bool is_sink = (sinkBitset[dest / 8] >> (dest % 8)) & 1;
                                if (!is_sink)
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
                                 const uint8_t* __restrict__ sinkBitset, int* __restrict__ visitedBitset)
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
        countl_type lastNbrIndex_InLastVertex_inCurBlock = lastEdgeIndex_eachBlock - frontier_degExSum_[lastVertexIndex_eachBlock];

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
                    bsearch_idx = search_sssp(nbrSize_exsum_block, logical_tid, 0u, (vertexNum_forCurIterBlock - 1));
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
                                bool is_sink = (sinkBitset[dest / 8] >> (dest % 8)) & 1;
                                if (!is_sink)
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
                   const uint8_t* __restrict__ sinkBitset, int* __restrict__ visitedBitset, const bool is_SSSP_destWeight)
{

    if (is_SSSP_destWeight)
    {
        CUDA_KERNEL_CALL(SSSP_balance_model_device_kernel_destWeight, nBlock, BLOCKSIZE_2, vertexValue, csr_offset, csr_destWeight, frontier_in,
                         frontier_out, frontierNum, totalDegree, frontierNum_next, frontier_degExSum_, frontier_balance_device_, sinkBitset,
                         visitedBitset);
    }
    else
    {
        CUDA_KERNEL_CALL(SSSP_balance_model_device_kernel, nBlock, BLOCKSIZE_2, vertexValue, csr_offset, csr_dest, csr_weight, frontier_in,
                         frontier_out, frontierNum, totalDegree, frontierNum_next, frontier_degExSum_, frontier_balance_device_, sinkBitset,
                         visitedBitset);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
} // namespace SSSP_DEVICE_SPACE

class GPU_SSSP
{
  private:
    const count_type vertexNum_{0};
    const countl_type edgeNum_{0};

    countl_type* csr_offset_host_{nullptr};
    vertex_id_type* csr_dest_host_{nullptr};
    edge_data_type* csr_weight_host_{nullptr};
    vertex_id_type* csr_destWeight_host_{nullptr};
    const bool is_SSSP_destWeight_{false};

    countl_type* csr_offset_device_{nullptr};
    vertex_id_type* csr_dest_device_{nullptr};
    edge_data_type* csr_weight_device_{nullptr};
    vertex_id_type* csr_destWeight_device_{nullptr};
    thrust::device_ptr<countl_type> csr_offset_thrust_{nullptr};

    vertex_id_type* frontier_in_device_{nullptr};
    vertex_id_type* frontier_out_device_{nullptr};
    countl_type* frontier_degExSum_device_{nullptr};
    countl_type* frontier_balance_device_{nullptr};
    thrust::device_ptr<vertex_id_type> frontier_in_thrust_{nullptr};
    thrust::device_ptr<vertex_id_type> frontier_out_thrust_{nullptr};
    thrust::device_ptr<countl_type> frontier_degExSum_thrust_{nullptr};
    thrust::device_ptr<countl_type> frontier_balance_thrust_{nullptr};

    vertex_data_type* vertexValue_device_{nullptr};
    thrust::device_ptr<vertex_data_type> vertexValue_thrust_{nullptr};
    vertex_data_type* vertexValue_host_{nullptr};

    const bool is_sortFrontier_;
    static constexpr bool ISSORT = true;
    void* cub_temp_device_{nullptr};
    size_t cub_alloc_size_;
    count_type frontierNum_{0};
    count_type* frontierNum_next_device_;
    thrust::device_ptr<count_type> frontierNum_next_thrust_{nullptr};
    int num_vertex_bits_;

    uint8_t* sinkBitset_device_{nullptr};
    uint8_t* visitedBitset_device_{nullptr};
    const size_t bitset_size_{0};

    int ite_{0};
    size_t nBlock_{0};

    /* Debug */
    static constexpr bool TEMP_TIME_DEBUG = false;

  public:
    GPU_SSSP(CSR_Result_type& csrResult_, GPU_memory_type gpuMemoryType = GPU_memory_type::GPU_MEM, const bool is_SSSP_destWeight = true)
        : vertexNum_(csrResult_.vertexNum), edgeNum_(csrResult_.edgeNum), csr_offset_host_(csrResult_.csr_offset),
          csr_dest_host_(csrResult_.csr_dest), csr_weight_host_(csrResult_.csr_weight), is_SSSP_destWeight_(is_SSSP_destWeight),
          bitset_size_(sizeof(int) * (vertexNum_ + INT_SIZE - 1) / INT_SIZE),
          is_sortFrontier_((vertexNum_ <= std::numeric_limits<int>::max()) && ISSORT), num_vertex_bits_((int)log2((float)vertexNum_) + 1)
    {
        CPJ::Timer time;
        if (is_SSSP_destWeight_)
        {
            time.start();
            constexpr bool isSame = std::is_same<vertex_id_type, edge_data_type>::value;
            if constexpr (!isSame) assert_msg(false, "[is_SSSP_destWeight_] need <vertex_id_type> same as <edge_data_type>");
            assert_msg((edgeNum_ * 2) < std::numeric_limits<countl_type>::max(), "(edgeNum_ * 2) large than [countl_type] max()");
            csr_destWeight_host_ = CPJ::AllocMem::allocMem_memset<vertex_id_type>(edgeNum_ * 2);
            omp_par_for(countl_type edge_id = 0; edge_id < edgeNum_; edge_id++)
            {
                csr_destWeight_host_[edge_id * 2] = csr_dest_host_[edge_id];
                csr_destWeight_host_[edge_id * 2 + 1] = csr_weight_host_[edge_id];
            }
            Msg_info("csr_destWeight_host_ alloc and set finish, Used time: %s", time.get_time_str().c_str());

            time.start();
            CUDA_CHECK(MALLOC_DEVICE(&csr_offset_device_, vertexNum_ + 1));
            CUDA_CHECK(H2D(csr_offset_device_, csr_offset_host_, vertexNum_ + 1));
            if (gpuMemoryType == GPU_memory_type::GPU_MEM)
            {
                CUDA_CHECK(MALLOC_DEVICE(&csr_destWeight_device_, edgeNum_ * 2));
                CUDA_CHECK(H2D(csr_destWeight_device_, csr_destWeight_host_, edgeNum_ * 2));
                Msg_info("GPU Used [DevicePtr], Used time: %s", time.get_time_str().c_str());
            }
            else if (gpuMemoryType == GPU_memory_type::UVM)
            {
                CUDA_CHECK(MALLOC_UVM(&csr_destWeight_device_, edgeNum_ * 2, fLI::FLAGS_useDeviceId));
                CPJ::ManagedMem::memcpy_smart(csr_destWeight_device_, csr_destWeight_host_, edgeNum_ * 2);
                Msg_info("GPU Used [UVM], Used time: %s", time.get_time_str().c_str());
            }
            else if (gpuMemoryType == GPU_memory_type::ZERO_COPY)
            {
                CUDA_CHECK(MALLOC_ZEROCOPY(&csr_destWeight_device_, edgeNum_ * 2, fLI::FLAGS_useDeviceId));
                CPJ::ManagedMem::memcpy_smart(csr_destWeight_device_, csr_destWeight_host_, edgeNum_ * 2);
                Msg_info("GPU Used [ZERO-COPY], Used time: %s", time.get_time_str().c_str());
            }
            else
            {
                assert_msg(false, "Error gpuMemoryType");
            }
        }
        else
        {
            time.start();
            CUDA_CHECK(MALLOC_DEVICE(&csr_offset_device_, vertexNum_ + 1));
            CUDA_CHECK(H2D(csr_offset_device_, csr_offset_host_, vertexNum_ + 1));
            if (gpuMemoryType == GPU_memory_type::GPU_MEM)
            {
                CUDA_CHECK(MALLOC_DEVICE(&csr_dest_device_, edgeNum_));
                CUDA_CHECK(H2D(csr_dest_device_, csr_dest_host_, edgeNum_));
                CUDA_CHECK(MALLOC_DEVICE(&csr_weight_device_, edgeNum_));
                CUDA_CHECK(H2D(csr_weight_device_, csr_weight_host_, edgeNum_));
                Msg_info("GPU Used [DevicePtr], Used time: %s", time.get_time_str().c_str());
            }
            else if (gpuMemoryType == GPU_memory_type::UVM)
            {
                CUDA_CHECK(MALLOC_UVM(&csr_dest_device_, edgeNum_, fLI::FLAGS_useDeviceId));
                CPJ::ManagedMem::memcpy_smart(csr_dest_device_, csr_dest_host_, edgeNum_);
                CUDA_CHECK(MALLOC_UVM(&csr_weight_device_, edgeNum_, fLI::FLAGS_useDeviceId));
                CPJ::ManagedMem::memcpy_smart(csr_weight_device_, csr_weight_host_, edgeNum_);
                Msg_info("GPU Used [UVM], Used time: %s", time.get_time_str().c_str());
            }
            else if (gpuMemoryType == GPU_memory_type::ZERO_COPY)
            {
                CUDA_CHECK(MALLOC_ZEROCOPY(&csr_dest_device_, edgeNum_, fLI::FLAGS_useDeviceId));
                CPJ::ManagedMem::memcpy_smart(csr_dest_device_, csr_dest_host_, edgeNum_);
                CUDA_CHECK(MALLOC_ZEROCOPY(&csr_weight_device_, edgeNum_, fLI::FLAGS_useDeviceId));
                CPJ::ManagedMem::memcpy_smart(csr_weight_device_, csr_weight_host_, edgeNum_);
                Msg_info("GPU Used [ZERO-COPY], Used time: %s", time.get_time_str().c_str());
            }
            else
            {
                assert_msg(false, "Error gpuMemoryType");
            }
        }

        CUDA_CHECK(MALLOC_DEVICE(&frontier_in_device_, vertexNum_));
        CUDA_CHECK(MALLOC_DEVICE(&frontier_out_device_, vertexNum_));
        CUDA_CHECK(MALLOC_DEVICE(&frontier_degExSum_device_, vertexNum_));
        CUDA_CHECK(MALLOC_DEVICE(&vertexValue_device_, vertexNum_));
        CUDA_CHECK(MALLOC_DEVICE(&frontier_balance_device_, vertexNum_)); //! 是否会超出范围
        CUDA_CHECK(MALLOC_DEVICE(&frontierNum_next_device_, 1));
        CUDA_CHECK(cudaMalloc(&sinkBitset_device_, bitset_size_));
        CUDA_CHECK(cudaMalloc(&visitedBitset_device_, bitset_size_));
        Msg_info("MALLOC_DEVICE and H2D finish, used time: %s", time.get_time_str().c_str());

        time.start();
        csr_offset_thrust_ = thrust::device_pointer_cast(csr_offset_device_);
        frontier_in_thrust_ = thrust::device_pointer_cast(frontier_in_device_);
        thrust::fill(frontier_in_thrust_, frontier_in_thrust_ + vertexNum_, 0);
        frontier_out_thrust_ = thrust::device_pointer_cast(frontier_out_device_);
        thrust::fill(frontier_out_thrust_, frontier_out_thrust_ + vertexNum_, 0);
        frontier_degExSum_thrust_ = thrust::device_pointer_cast(frontier_degExSum_device_);
        thrust::fill(frontier_degExSum_thrust_, frontier_degExSum_thrust_ + vertexNum_, 0);
        frontier_balance_thrust_ = thrust::device_pointer_cast(frontier_balance_device_);
        vertexValue_thrust_ = thrust::device_pointer_cast(vertexValue_device_);
        frontierNum_next_thrust_ = thrust::device_pointer_cast(frontierNum_next_device_);
        Msg_info("Thrust::device_pointer_cast finish, used time: %s", time.get_time_str().c_str());

        time.start();
        setSinkBitset();
        Msg_info("setSinkBitset finish, used time: %s", time.get_time_str().c_str());
    }

    ~GPU_SSSP() { freeGPUMemory(); }

    std::vector<SpeedRecord_type> measureSSSP(int64_t root)
    {
        std::vector<SpeedRecord_type> iteRecord_vec;
        iteRecord_vec.reserve(1000);
        iteRecord_vec.clear();

        assert_msg(root >= 0 && root < vertexNum_, "Error root, root = %ld", root);
        if ((csr_offset_host_[root + 1] - csr_offset_host_[root]) == 0) return iteRecord_vec;

        if (is_sortFrontier_ && ISSORT)
        {
            cub::DeviceRadixSort::SortKeys(cub_temp_device_, cub_alloc_size_, static_cast<const vertex_id_type*>(frontier_in_device_),
                                           static_cast<vertex_id_type*>(frontier_out_device_), (int)vertexNum_);
            CUDA_CHECK(cudaMalloc(&cub_temp_device_, cub_alloc_size_));
        }

        thrust::fill(vertexValue_thrust_, vertexValue_thrust_ + vertexNum_, std::numeric_limits<vertex_data_type>::max());
        vertexValue_thrust_[root] = 0;
        frontier_in_thrust_[0] = root;
        frontierNum_ = 1;

        thrust::device_ptr<uint8_t> visitedBitset_thrust = thrust::device_pointer_cast(visitedBitset_device_);
        thrust::fill(visitedBitset_thrust, visitedBitset_thrust + bitset_size_, 0);
        visitedBitset_thrust[root / 8] = 1U << (root % 8);
        ite_ = 0;

        CPJ::Timer totalTime;
        CPJ::Timer singleTime;
        CPJ::Timer tempTime;
        do
        {
            singleTime.start();
            ite_++;
            SpeedRecord_type speedRecord;
            speedRecord.iteId = ite_;

            setActiveDegree();
            countl_type degreeTotal = getActiveExSum();

            speedRecord.activeVertexNum = frontierNum_;
            speedRecord.activeEdgeNum = degreeTotal;

            CUDA_CHECK(cudaMemset(visitedBitset_device_, 0, bitset_size_));

            nBlock_ = (degreeTotal + TASK_PER_BLOCK - 1) / TASK_PER_BLOCK;

            thrust::counting_iterator<countl_type> cnt_iter(0);
            auto query_iter_first = thrust::make_transform_iterator(cnt_iter, SSSP_DEVICE_SPACE::Set_blockTask_type());
            auto query_iter_last = thrust::make_transform_iterator(cnt_iter + nBlock_, SSSP_DEVICE_SPACE::Set_blockTask_type());
            assert_msg(vertexNum_ > (query_iter_last - query_iter_first),
                       "frontier_balance_thrust_ 返回超标, vertexNum_ = %zu, (query_iter_last - query_iter_first) = %zu", SCU64(vertexNum_),
                       SCU64((query_iter_last - query_iter_first)));
            thrust::upper_bound(thrust::device, frontier_degExSum_thrust_, frontier_degExSum_thrust_ + frontierNum_, query_iter_first,
                                query_iter_last, frontier_balance_thrust_);
            nBlock_ = std::min(nBlock_, SCU64(MAX_BLOCKS));
            SSSP_DEVICE_SPACE::balance_model(nBlock_, vertexValue_device_, csr_offset_device_, csr_dest_device_, csr_weight_device_,
                                             csr_destWeight_device_, frontier_in_device_, frontier_out_device_, frontierNum_, degreeTotal,
                                             frontierNum_next_device_, frontier_degExSum_device_, frontier_balance_device_, sinkBitset_device_,
                                             reinterpret_cast<int*>(visitedBitset_device_), is_SSSP_destWeight_);
            count_type frontierNum_next = (count_type)frontierNum_next_thrust_[0];
            if (frontierNum_next == 0)
            {
                speedRecord.time_ms = singleTime.get_time_ms();
                speedRecord.total_time_ms = totalTime.get_time_ms();
                iteRecord_vec.push_back(speedRecord);
                // Msg_node("\t[Complete]: GPU-SSSP -> iteration: %2d, Used time:: %5.2lf (ms)", ite_, speedRecord.total_time_ms);
                // Msg_info("[GPU-SSSP] process end, (%s) usedGPUMem = %.2lf (GB)", GPU_memory_type_name[SCI32(fLI::FLAGS_gpuMemory)],
                //          BYTES_TO_GB(CPJ::MemoryInfo::getMemoryFree_Device(FLAGS_useDeviceId)));
                break;
            }

            if (is_sortFrontier_ && ISSORT)
            {
                int start_bit = static_cast<int>(0.35 * num_vertex_bits_);
                cub::DeviceRadixSort::SortKeys(cub_temp_device_, cub_alloc_size_, static_cast<const vertex_id_type*>(frontier_out_device_),
                                               static_cast<vertex_id_type*>(frontier_in_device_), (int)frontierNum_next, start_bit, num_vertex_bits_);
            }

            frontierNum_ = frontierNum_next;
            CUDA_CHECK(MEMSET_DEVICE(frontierNum_next_device_, 1));

            speedRecord.time_ms = singleTime.get_time_ms();
            iteRecord_vec.push_back(speedRecord);
            // Msg_info("\t[GPU]: The (%2d) iteration, activeNum = %zu, Used time: = %7.2lf (ms)", ite_, SCU64(frontierNum_next),
            //          singleTime.get_time_ms());
        }
        while (true);

        return iteRecord_vec;
    }

    vertex_data_type* getResult()
    {
        CUDA_CHECK(MALLOC_HOST(&vertexValue_host_, vertexNum_));
        CUDA_CHECK(D2H(vertexValue_host_, vertexValue_device_, vertexNum_));
        return vertexValue_host_;
    }

    void freeGPUMemory()
    {
        if (csr_offset_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(csr_offset_device_));
            csr_offset_device_ = nullptr;
        }
        if (csr_dest_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(csr_dest_device_));
            csr_dest_device_ = nullptr;
        }
        if (csr_weight_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(csr_weight_device_));
            csr_weight_device_ = nullptr;
        }
        if (csr_destWeight_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(csr_destWeight_device_));
            csr_destWeight_device_ = nullptr;
        }
        if (frontier_in_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(frontier_in_device_));
            frontier_in_device_ = nullptr;
        }
        if (frontier_out_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(frontier_out_device_));
            frontier_out_device_ = nullptr;
        }
        if (frontier_degExSum_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(frontier_degExSum_device_));
            frontier_degExSum_device_ = nullptr;
        }
        if (vertexValue_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(vertexValue_device_));
            vertexValue_device_ = nullptr;
        }
        if (vertexValue_host_ != nullptr)
        {
            CUDA_CHECK(FREE_HOST(vertexValue_host_));
            vertexValue_host_ = nullptr;
        }
        if (frontier_balance_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(frontier_balance_device_));
            frontier_balance_device_ = nullptr;
        }
        if (frontierNum_next_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(frontierNum_next_device_));
            frontierNum_next_device_ = nullptr;
        }
        if (sinkBitset_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(sinkBitset_device_));
            sinkBitset_device_ = nullptr;
        }
        if (visitedBitset_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(visitedBitset_device_));
            visitedBitset_device_ = nullptr;
        }
        if (cub_temp_device_ != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(cub_temp_device_));
            cub_temp_device_ = nullptr;
        }
    }

  private:
    void setSinkBitset()
    {
        thrust::device_ptr<uint8_t> sinkBitset_thrust = thrust::device_pointer_cast(sinkBitset_device_);
        thrust::counting_iterator<size_t> iter_first(0);
        thrust::counting_iterator<size_t> iter_last = iter_first + bitset_size_;
        thrust::for_each(iter_first, iter_last, SSSP_DEVICE_SPACE::Set_sink_type(sinkBitset_thrust, csr_offset_thrust_, vertexNum_));
    }

    void setActiveDegree()
    {
        thrust::counting_iterator<vertex_id_type> iter_first(0);
        thrust::counting_iterator<vertex_id_type> iter_last = iter_first + frontierNum_;
        thrust::for_each(iter_first, iter_last,
                         SSSP_DEVICE_SPACE::Set_activeDegree_type(frontier_in_thrust_, csr_offset_thrust_, frontier_degExSum_thrust_));
    }

    countl_type getActiveExSum()
    {
        countl_type totalDegree_temp = frontier_degExSum_thrust_[frontierNum_ - 1];
        thrust::exclusive_scan(frontier_degExSum_thrust_, frontier_degExSum_thrust_ + frontierNum_, frontier_degExSum_thrust_);
        return totalDegree_temp + frontier_degExSum_thrust_[frontierNum_ - 1];
    }
};

} // namespace CPJ