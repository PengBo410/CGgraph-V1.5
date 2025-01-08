#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Console/console_bar.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Graph/getGraphCSR.hpp"
#include "Basic/Graph/graphFileList.hpp"
#include "Basic/IO/io_adapter_V1.hpp"
#include "Basic/Memory/alloc_CPJ.hpp"
#include "Basic/Other/scan_CPJ.hpp"
#include "Basic/Thread/omp_def.hpp"
#include "Basic/Timer/timer_CPJ.hpp"
#include "Basic/Type/data_type.hpp"
#include "rank.hpp"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <omp.h>
#include <pstl/glue_execution_defs.h>
#include <sstream>
#include <thrust/binary_search.h>

namespace CPJ {

namespace Balance_Reorder {

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

} // namespace Balance_Reorder

class ReorderGraph
{
  private:
    count_type vertexNum_{0};
    countl_type edgeNum_{0};

    countl_type* csr_offset_{nullptr};
    vertex_id_type* csr_dest_{nullptr};
    edge_data_type* csr_weight_{nullptr};
    degree_type* outDegree_{nullptr};

    static constexpr bool DEBUG_TIME = false;
    static constexpr bool DEBUG_PROGRESS = false;
    static constexpr bool DEBUG_PRINTF = false;
    Bar* bar{nullptr};

    vertex_id_type* rank_{nullptr};
    vertex_id_type* old2new_{nullptr};

    struct Reorder_type
    {
        vertex_id_type newId;
        vertex_id_type oldId;
    };

    const size_t REORDER_HOST_PER_THREAD_MIN_EDGES_TASKS = 6400;
    countl_type* thread_edges_{nullptr};
    countl_type* thread_balance_{nullptr};
    Balance_Reorder::ThreadState_type* threadState_{nullptr};
    static constexpr std::memory_order ATOMIC_ORDER = std::memory_order_relaxed; // std::memory_order_seq_cst;

    static constexpr bool REORDER_WEIGHT = true;

  public:
    ReorderGraph(CSR_Result_type& csr_result)
        : vertexNum_(csr_result.vertexNum), edgeNum_(csr_result.edgeNum), csr_offset_(csr_result.csr_offset), csr_dest_(csr_result.csr_dest),
          csr_weight_(csr_result.csr_weight)
    {
        assert_msg(csr_result.outDegree != nullptr, "We need CSR_Result_type provides outDegree");
        if constexpr (REORDER_WEIGHT)
        {
            assert_msg(csr_result.csr_weight != nullptr, "REORDER_WEIGHT need CSR_Result_type provides csrWeight");
        }

        CPJ::CGgraphReorder reorder(csr_result);
        rank_ = reorder.reoderV15();
        reorder.checkRankResult(rank_);

        outDegree_ = csr_result.outDegree;
        assert_msg(rank_ != nullptr, "In ReorderGraph Rank can not nullptr");

        thread_edges_ = CPJ::AllocMem::allocMem<countl_type>(ThreadNum + 1);
        memset(thread_edges_, 0, (ThreadNum + 1) * sizeof(countl_type));

        thread_balance_ = CPJ::AllocMem::allocMem<countl_type>(ThreadNum);
        memset(thread_balance_, 0, ThreadNum * sizeof(countl_type));

        old2new_ = CPJ::AllocMem::allocMem<vertex_id_type>(vertexNum_);
        memset(old2new_, 0, (vertexNum_) * sizeof(vertex_id_type));
    }

    CSR_Result_type doReorder(const std::string graphName = "")
    {
        CSR_Result_type csrResult_reorder;
        csrResult_reorder.vertexNum = vertexNum_;
        csrResult_reorder.edgeNum = edgeNum_;

        CPJ::Timer timer;

        if constexpr (DEBUG_TIME) timer.start();
        std::vector<Reorder_type> reorder_vec(vertexNum_);
        omp_par_for(count_type vertex_id = 0; vertex_id < vertexNum_; vertex_id++)
        {
            Reorder_type reorder;
            reorder.newId = vertex_id;
            reorder.oldId = rank_[vertex_id];
            reorder_vec[vertex_id] = reorder;
        }

        // clang-format off
        std::sort(std::execution::par_unseq, reorder_vec.begin(), reorder_vec.end(),
            [&](const Reorder_type& a, const Reorder_type& b)
            {
                return a.oldId < b.oldId;
            }
        );
        // clang-format on
        if constexpr (DEBUG_TIME) Msg_info("Prepare the reorder vec, used time: %s", timer.get_time_str().c_str());
        if constexpr (DEBUG_PRINTF)
        {
            std::stringstream ss;
            ss << "\n";
            for (count_type index = 0; index < std::min(static_cast<count_type>(10), static_cast<count_type>(reorder_vec.size())); index++)
            {
                ss << "[" << index << "] ->" << reorder_vec[index].newId << std::endl;
            }
            Msg_info("[oldId] -> [newId]: %s", ss.str().c_str()); // 3,1,7,5,6,0,4,2
        }

        if constexpr (DEBUG_TIME) timer.start();
        csrResult_reorder.outDegree = new degree_type[vertexNum_];
        memset(csrResult_reorder.outDegree, 0, sizeof(degree_type) * vertexNum_);
        omp_par_for(count_type vertex_id = 0; vertex_id < vertexNum_; vertex_id++)
        {
            csrResult_reorder.outDegree[vertex_id] = outDegree_[rank_[vertex_id]];
        }
        if constexpr (DEBUG_PRINTF)
        {
            std::stringstream ss;
            for (count_type index = 0; index < std::min(static_cast<count_type>(10), static_cast<count_type>(vertexNum_)); index++)
            {
                ss << csrResult_reorder.outDegree[index] << ", ";
            }
            Msg_info("csrResult_reorder.outDegree: %s", ss.str().c_str()); // 2,2,4,3,2,1,0,0
        }
        if constexpr (DEBUG_TIME) Msg_info("Build the csrResult_reorder.outDegree, used time: %s", timer.get_time_str().c_str());

        if constexpr (DEBUG_TIME) timer.start();
        csrResult_reorder.csr_offset = new countl_type[vertexNum_ + 1];

        countl_type* temp{nullptr};
        if constexpr (!std::is_same_v<countl_type, degree_type>)
        {
            temp = new countl_type[vertexNum_];
            std::copy(std::execution::par_unseq, csrResult_reorder.outDegree, csrResult_reorder.outDegree + vertexNum_, temp);
            csrResult_reorder.csr_offset[vertexNum_] = CPJ::SCAN::HOST::exclusive_scan_std_par(temp, vertexNum_, csrResult_reorder.csr_offset);
            delete[] temp;
        }
        else
        {
            csrResult_reorder.csr_offset[vertexNum_] =
                CPJ::SCAN::HOST::exclusive_scan_std_par(csrResult_reorder.outDegree, vertexNum_, csrResult_reorder.csr_offset);
        }
        assert_msg(csrResult_reorder.csr_offset[vertexNum_] == edgeNum_, "csrResult_reorder.csr_offset[vertexNum_] != edgeNum, (%zu) != (%zu)",
                   SCU64(csrResult_reorder.csr_offset[vertexNum_]), SCU64(edgeNum_));
        if constexpr (DEBUG_TIME) Msg_info("Build the csrResult_reorder.csr_offset, used time: %s", timer.get_time_str().c_str());
        if constexpr (DEBUG_PRINTF)
        {
            std::stringstream ss;
            for (count_type index = 0; index < std::min(static_cast<count_type>(10), static_cast<count_type>(vertexNum_ + 1)); index++)
            {
                ss << csrResult_reorder.csr_offset[index] << ", ";
            }
            Msg_info("csrResult_reorder.csr_offset: %s", ss.str().c_str()); // 0,2,4,8,11,13,14,14,14
        }

        /* 现在再构建Balance */
        countl_type threads_req = (edgeNum_ + REORDER_HOST_PER_THREAD_MIN_EDGES_TASKS - 1) / REORDER_HOST_PER_THREAD_MIN_EDGES_TASKS;
        threads_req = std::min(threads_req, static_cast<countl_type>(ThreadNum));
        countl_type threads_max_avg = (edgeNum_ + threads_req - 1) / threads_req;

        countl_type* thread_edges_ = CPJ::AllocMem::allocMem<countl_type>(ThreadNum + 1);
        memset(thread_edges_, 0, (ThreadNum + 1) * sizeof(countl_type));
        thread_edges_[0] = 0;
        for (int thread_id = 1; thread_id < threads_req; thread_id++)
        {
            thread_edges_[thread_id] = thread_edges_[thread_id - 1] + threads_max_avg;
        }
        thread_edges_[threads_req] = edgeNum_;
        if constexpr (DEBUG_PRINTF)
        {
            std::stringstream ss;
            for (count_type thread_id = 0; thread_id <= threads_req; thread_id++)
            {
                ss << thread_edges_[thread_id] << ", ";
            }
            Msg_info("thread_edges_: %s", ss.str().c_str()); // 0,3,6,9,12,14
        }
        thrust::upper_bound(thrust::host, csrResult_reorder.csr_offset, csrResult_reorder.csr_offset + vertexNum_, thread_edges_,
                            thread_edges_ + threads_req, thread_balance_);
        if constexpr (DEBUG_PRINTF)
        {
            std::stringstream ss;
            for (count_type thread_id = 0; thread_id < threads_req; thread_id++)
            {
                ss << thread_balance_[thread_id] << ", ";
            }
            Msg_info("thrust::upper_bound -> thread_balance_: %s", ss.str().c_str()); // 1,2,3,4,5
        }

        csrResult_reorder.csr_dest = new vertex_id_type[edgeNum_];
        memset(csrResult_reorder.csr_dest, 0, sizeof(vertex_id_type) * edgeNum_);
        if constexpr (REORDER_WEIGHT)
        {
            csrResult_reorder.csr_weight = new edge_data_type[edgeNum_];
            memset(csrResult_reorder.csr_weight, 0, sizeof(edge_data_type) * edgeNum_);
        }
        omp_par_threads(threads_req)
        {
            CPJ::Timer* timer_thread{nullptr};
            if constexpr (DEBUG_PROGRESS)
            {
                timer_thread = new CPJ::Timer();
                timer_thread->start();
            }

            const int threadId = omp_get_thread_num();
            const countl_type firstEdgeIndex_thread = thread_edges_[threadId];        // 期望的first_edge
            const countl_type lastEdgeIndex_thread = thread_edges_[threadId + 1] - 1; // 期望的last_edge

            const count_type firstVertexIndex_thread = thread_balance_[threadId] - 1;
            count_type lastVertexIndex_thread;
            if (lastEdgeIndex_thread < (edgeNum_ - 1))
            {
                lastVertexIndex_thread = thread_balance_[threadId + 1] - 1;
                if (csrResult_reorder.csr_offset[lastVertexIndex_thread] == lastEdgeIndex_thread + 1) lastVertexIndex_thread--;
            }
            else
            {
                lastVertexIndex_thread = vertexNum_ - 1;
            }

            const count_type vertexNum_thread = lastVertexIndex_thread - firstVertexIndex_thread + 1;
            const degree_type firstNbrIndex_InFirstVertex_thread = firstEdgeIndex_thread - csrResult_reorder.csr_offset[firstVertexIndex_thread];
            const degree_type lastNbrIndex_InLastVertex_thread = lastEdgeIndex_thread - csrResult_reorder.csr_offset[lastVertexIndex_thread];

            // Msg_info("T[%d]: 负责处理(%d)个顶点, 点[%zu, %zu], 边：[%zu, %zu]", threadId, vertexNum_thread, SCU64(firstVertexIndex_thread),
            //          SCU64(lastVertexIndex_thread), SCU64(firstNbrIndex_InFirstVertex_thread), SCU64(lastNbrIndex_InLastVertex_thread));

            countl_type edgeWriteCount = 0;
            for (count_type index = firstVertexIndex_thread; index < firstVertexIndex_thread + vertexNum_thread; index++)
            {
                const vertex_id_type vid = rank_[index];
                countl_type nbrStartIndex = csr_offset_[vid];
                const degree_type nbrSize = csr_offset_[vid + 1] - nbrStartIndex;
                degree_type nbrSize_forCurVertex_thread = nbrSize;

                if (index == firstVertexIndex_thread)
                {
                    nbrSize_forCurVertex_thread -= firstNbrIndex_InFirstVertex_thread;
                    nbrStartIndex += firstNbrIndex_InFirstVertex_thread;
                }

                if (index == (firstVertexIndex_thread + vertexNum_thread - 1))
                {
                    nbrSize_forCurVertex_thread -= (nbrSize - lastNbrIndex_InLastVertex_thread - 1);
                }

                for (degree_type nbr_id = 0; nbr_id < nbrSize_forCurVertex_thread; nbr_id++)
                {
                    vertex_id_type dest_old = csr_dest_[nbrStartIndex + nbr_id];
                    vertex_id_type dest_new = reorder_vec[dest_old].newId;
                    csrResult_reorder.csr_dest[firstEdgeIndex_thread + edgeWriteCount] = dest_new;
                    if constexpr (REORDER_WEIGHT)
                    {
                        edge_data_type weight = csr_weight_[nbrStartIndex + nbr_id];
                        csrResult_reorder.csr_weight[firstEdgeIndex_thread + edgeWriteCount] = weight;
                    }
                    edgeWriteCount++;

                    // @see match (below the file)
                    // Msg_info("==>T[%d] 处理了顶点(%u)的第(%u)边: %u", threadId, vid, nbrStartIndex + nbr_id - csr_offset_[vid], dest);
                }
            }

            if constexpr (DEBUG_PROGRESS)
            {
                if constexpr (!REORDER_WEIGHT)
                {
                    Msg_info("T[%d]从位置(%zu)开始共写入(%zu)条边到csrResult_reorder.csr_dest, 用时: %s", threadId, SCU64(firstEdgeIndex_thread),
                             SCU64(edgeWriteCount), timer_thread->get_time_str().c_str());
                }
                else
                {
                    Msg_info("T[%d]从位置(%zu)开始共写入(%zu)条边到csrResult_reorder.csr_dest和csrResult_reorder.csr_weight, 用时: %s", threadId,
                             SCU64(firstEdgeIndex_thread), SCU64(edgeWriteCount), timer_thread->get_time_str().c_str());
                }
            }
        } // end of [omp_par_threads]

        omp_par_for(count_type vertex_id = 0; vertex_id < vertexNum_; vertex_id++) { old2new_[vertex_id] = reorder_vec[vertex_id].newId; }
        std::vector<Reorder_type>().swap(reorder_vec);
        assert_msg(reorder_vec.size() == 0, "After free reorder_vec, its size need to 0, size = %zu", reorder_vec.size());

        // 写入到文件
        {
            std::string old2new_path = getCGgraph_reorder_old2newFile(graphName);
            if (!CPJ::FS::isExist(old2new_path))
            {
                writeArrayToFile(old2new_, vertexNum_, old2new_path);
            }
            else
            {
                vertex_id_type* old2new_check = readArrayFromFile<vertex_id_type>(old2new_path);
                omp_par_for(count_type vertex_id = 0; vertex_id < vertexNum_; vertex_id++)
                {
                    assert_msg(old2new_[vertex_id] == old2new_check[vertex_id], "old2new_[%zu] = %zu, old2new_check[%zu] = %zu", SCU64(vertex_id),
                               SCU64(old2new_[vertex_id]), SCU64(vertex_id), SCU64(old2new_check[vertex_id]));
                }
                Msg_finish("File[%s] alread exist, check finshed!", old2new_path.c_str());
                free(old2new_check);
            }
        }

        {
            std::string rank_path = getCGgraph_reorder_rankFile(graphName);
            if (!CPJ::FS::isExist(rank_path))
            {
                writeArrayToFile(rank_, vertexNum_, rank_path);
            }
            else
            {
                vertex_id_type* rank_check = readArrayFromFile<vertex_id_type>(rank_path);
                omp_par_for(count_type vertex_id = 0; vertex_id < vertexNum_; vertex_id++)
                {
                    assert_msg(rank_[vertex_id] == rank_check[vertex_id], "rank_[%zu] = %zu, rank_check[%zu] = %zu", SCU64(vertex_id),
                               SCU64(rank_[vertex_id]), SCU64(vertex_id), SCU64(rank_check[vertex_id]));
                }
                Msg_finish("File[%s] alread exist, check finshed!", rank_path.c_str());
                free(rank_check);
            }
            CPJ::AllocMem::freeMem(rank_, vertexNum_);
        }

        {
            std::string csrOffset_path = getCGgraph_reorder_csrOffset(graphName);
            if (!CPJ::FS::isExist(csrOffset_path))
            {
                writeArrayToFile(csrResult_reorder.csr_offset, (vertexNum_ + 1), csrOffset_path);
            }
            else
            {
                countl_type* csr_offset_check = readArrayFromFile<countl_type>(csrOffset_path);
                omp_par_for(count_type vertex_id = 0; vertex_id <= vertexNum_; vertex_id++)
                {
                    assert_msg(csrResult_reorder.csr_offset[vertex_id] == csr_offset_check[vertex_id],
                               "csrResult_reorder.csr_offset[%zu] = %zu, csr_offset_check[%zu] = %zu", SCU64(vertex_id),
                               SCU64(csrResult_reorder.csr_offset[vertex_id]), SCU64(vertex_id), SCU64(csr_offset_check[vertex_id]));
                }
                Msg_finish("File[%s] alread exist, check finshed!", csrOffset_path.c_str());
                free(csr_offset_check);
            }
        }

        {
            std::string csrDest_path = getCGgraph_reorder_csrDest(graphName);
            if (!CPJ::FS::isExist(csrDest_path))
            {
                writeArrayToFile(csrResult_reorder.csr_dest, edgeNum_, csrDest_path);
            }
            else
            {
                vertex_id_type* csr_dest_check = readArrayFromFile<vertex_id_type>(csrDest_path);
                omp_par_for(countl_type edge_id = 0; edge_id < edgeNum_; edge_id++)
                {
                    assert_msg(csrResult_reorder.csr_dest[edge_id] == csr_dest_check[edge_id],
                               "csrResult_reorder.csr_dest[%zu] = %zu, csr_dest_check[%zu] = %zu", SCU64(edge_id),
                               SCU64(csrResult_reorder.csr_dest[edge_id]), SCU64(edge_id), SCU64(csr_dest_check[edge_id]));
                }
                Msg_finish("File[%s] alread exist, check finshed!", csrDest_path.c_str());
                free(csr_dest_check);
            }
        }

        if constexpr (REORDER_WEIGHT)
        {
            std::string csrWeight_path = getCGgraph_reorder_csrWeight(graphName);
            if (!CPJ::FS::isExist(csrWeight_path))
            {
                writeArrayToFile(csrResult_reorder.csr_weight, edgeNum_, csrWeight_path);
            }
            else
            {
                vertex_id_type* csr_weight_check = readArrayFromFile<edge_data_type>(csrWeight_path);
                omp_par_for(countl_type edge_id = 0; edge_id < edgeNum_; edge_id++)
                {
                    assert_msg(csrResult_reorder.csr_weight[edge_id] == csr_weight_check[edge_id],
                               "csrResult_reorder.csr_weight[%zu] = %zu, csr_weight_check[%zu] = %zu", SCU64(edge_id),
                               SCU64(csrResult_reorder.csr_weight[edge_id]), SCU64(edge_id), SCU64(csr_weight_check[edge_id]));
                }
                Msg_finish("File[%s] alread exist, check finshed!", csrWeight_path.c_str());
                free(csr_weight_check);
            }
        }

        return csrResult_reorder;
    }

    // vertex_id_type* getRank() { return rank_; }
    vertex_id_type* getOld2New() { return old2new_; }

    void freeOldCSR()
    {
        if (csr_offset_ != nullptr)
        {
            delete[] csr_offset_;
            csr_offset_ = nullptr;
        }
        if (csr_dest_ != nullptr)
        {
            delete[] csr_dest_;
            csr_dest_ = nullptr;
        }
        if (csr_weight_ != nullptr)
        {
            delete[] csr_weight_;
            csr_weight_ = nullptr;
        }
        if (outDegree_ != nullptr)
        {
            delete[] outDegree_;
            outDegree_ = nullptr;
        }
    }

  private:
    template <typename T>
    void writeArrayToFile(T* array, size_t arrLen, std::string& filePath)
    {
        assert_msg(array != nullptr, "Write nullptr array to file[%s]", filePath.c_str());

        CPJ::Timer write_timer;
        CPJ::IOAdaptor ioAdapter(filePath);
        ioAdapter.openFile("w");
        ioAdapter.writeBinFile_sync(array, arrLen * sizeof(T), omp_get_max_threads());
        ioAdapter.closeFile();
        Msg_info("Write array to [%s] finished, Used time: %s", filePath.c_str(), write_timer.get_time_str().c_str());
    }

    template <typename T>
    T* readArrayFromFile(std::string& filePath)
    {
        CPJ::Timer timer;
        CPJ::IOAdaptor ioAdapter(filePath);
        ioAdapter.openFile();
        T* array = ioAdapter.readBinFileEntire_sync<T>(omp_get_max_threads());
        ioAdapter.closeFile();
        Msg_info("Read array from[%s] finished, Used time: %s", filePath.c_str(), timer.get_time_str().c_str());

        return array;
    }
};

/* @see match

T[0]: 负责处理(2)个顶点, 点[0, 1], 边：[0, 0]
T[0] 处理了顶点(5)的第(0)边: 1
T[0] 处理了顶点(5)的第(1)边: 3
T[0] 处理了顶点(1)的第(0)边: 2

T[1]: 负责处理(2)个顶点, 点[1, 2], 边：[1, 1]
T[1] 处理了顶点(1)的第(1)边: 6
T[1] 处理了顶点(7)的第(0)边: 0
T[1] 处理了顶点(7)的第(1)边: 2

T[2]: 负责处理(2)个顶点, 点[2, 3], 边：[2, 0]
T[2] 处理了顶点(7)的第(2)边: 4
T[2] 处理了顶点(7)的第(3)边: 5
T[2] 处理了顶点(0)的第(0)边: 1

T[3]: 负责处理(2)个顶点, 点[3, 4], 边：[1, 0]
T[3] 处理了顶点(0)的第(1)边: 4
T[3] 处理了顶点(0)的第(2)边: 7
T[3] 处理了顶点(6)的第(0)边: 0

T[4]: 负责处理(4)个顶点, 点[4, 7], 边：[1, 4294967295] -> [reorderGraph.hpp:221 行]
T[4] 处理了顶点(6)的第(1)边: 7 -> [reorderGraph.hpp:259 行]
T[4] 处理了顶点(3)的第(0)边: 6 -> [reorderGraph.hpp:259 行]

*/

} // namespace CPJ