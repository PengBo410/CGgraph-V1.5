#pragma once

#include "Basic/Bitmap/fixed_bitset.cuh"
#include "Basic/Console/console_bar.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/IO/io_adapter_V1.hpp"
#include "Basic/Memory/alloc_CPJ.hpp"
#include "Basic/Other/checkDuplicate.hpp"
#include "Basic/Timer/timer_CPJ.hpp"
#include "Basic/Type/data_type.hpp"

#include <pstl/glue_execution_defs.h>
#include <queue>

namespace CPJ {

class CGgraphReorder
{
  private:
    count_type vertexNum_{0};
    countl_type edgeNum_{0};

    countl_type* csr_offset_{nullptr};
    vertex_id_type* csr_dest_{nullptr};
    degree_type* outDegree_{nullptr};

    bool* isVisited_{nullptr};
    bool isRCMFinished_{false};
    Fixed_Bitset* bitset{nullptr};

    static constexpr bool DEBUG_TIME = false;
    double pushQueueTime_{0.0};
    double traverseNbrTime_{0.0};
    double sortNbrTime_{0.0};
    double copyNbrToQueueTime_{0.0};
    double whileTime_{0.0};

    static constexpr bool DEBUG_PROGRESS = true;
    Bar* bar{nullptr};

    struct NbrDegree_type
    {
        vertex_id_type vertexId{0};
        degree_type vertexDegree{0};
    };

  public:
    enum class RCM_type { SEQ, PAR, MAX_VALUE };

  public:
    CGgraphReorder(CSR_Result_type& csr_result)
        : vertexNum_(csr_result.vertexNum), edgeNum_(csr_result.edgeNum), csr_offset_(csr_result.csr_offset), csr_dest_(csr_result.csr_dest)
    {
        assert_msg(csr_result.outDegree != nullptr, "We need CSR_Result_type provides outDegree");
        outDegree_ = csr_result.outDegree;

        isVisited_ = CPJ::AllocMem::allocMem<bool>(vertexNum_);
        memset(isVisited_, 0, vertexNum_ * sizeof(bool));
        bitset = new Fixed_Bitset(vertexNum_);
        bitset->clear_smart();

        if constexpr (DEBUG_PROGRESS) bar = new Bar(static_cast<bool>(vertexNum_ >= 5000000));
    }

    vertex_id_type* reoderV15()
    {
        std::queue<vertex_id_type> rcm_queue;
        CPJ::Timer timer_total;
        CPJ::Timer timer;

        vertex_id_type* rank = CPJ::AllocMem::allocMem<vertex_id_type>(vertexNum_);
        memset(rank, 0, sizeof(vertex_id_type) * vertexNum_);
        count_type alreadyRankVertex = 0;

        if constexpr (DEBUG_TIME) timer.start();
        std::vector<NbrDegree_type> sortDegree_vec;
        sortDegree_vec.resize(vertexNum_);
        omp_par_for(count_type vertex_id = 0; vertex_id < vertexNum_; vertex_id++)
        {
            NbrDegree_type nbrDegree;
            nbrDegree.vertexId = vertex_id;
            nbrDegree.vertexDegree = outDegree_[vertex_id];
            sortDegree_vec[vertex_id] = nbrDegree;
        }
        // clang-format off
        std::sort(std::execution::par_unseq, sortDegree_vec.begin(), sortDegree_vec.end(),
            [&](const NbrDegree_type& a, const NbrDegree_type& b)
            {
                return std::tie(a.vertexDegree, a.vertexId) < std::tie(b.vertexDegree, b.vertexId);
            }
        );
        // clang-format on
        if constexpr (DEBUG_TIME) Msg_info("Sort NbrDegree, used time: %s", timer.get_time_str().c_str());

        std::vector<NbrDegree_type> sortNbr_vec;
        sortNbr_vec.reserve(sortDegree_vec[vertexNum_ - 1].vertexDegree);
        timer_total.start();
        for (count_type outVertex_id = 0; outVertex_id < vertexNum_; outVertex_id++)
        {
            if (!bitset->get(sortDegree_vec[outVertex_id].vertexId))
            {
                bitset->set_bit_unsync(sortDegree_vec[outVertex_id].vertexId);
                if (sortDegree_vec[outVertex_id].vertexDegree == 0)
                {
                    rank[alreadyRankVertex++] = sortDegree_vec[outVertex_id].vertexId;
                    continue;
                }
                else rcm_queue.push(sortDegree_vec[outVertex_id].vertexId);
            }

            while (!rcm_queue.empty())
            {
                if constexpr (DEBUG_TIME) timer.start();
                vertex_id_type firstVertexId = rcm_queue.front();
                rcm_queue.pop();
                rank[alreadyRankVertex++] = firstVertexId;
                if constexpr (DEBUG_PROGRESS)
                {
                    bar->progress(alreadyRankVertex, vertexNum_, "Rank progress: ");
                }

                sortNbr_vec.clear();
                assert_msg(sortNbr_vec.size() == 0, "nbrList_vec需要初始化为0, 当前size = %zu", sortNbr_vec.size());
                for (countl_type nbr_index = csr_offset_[firstVertexId]; nbr_index < csr_offset_[firstVertexId + 1]; nbr_index++)
                {
                    vertex_id_type nbrId = csr_dest_[nbr_index];
                    if (!bitset->get(nbrId))
                    {
                        NbrDegree_type nbrDegree;
                        nbrDegree.vertexId = nbrId;
                        nbrDegree.vertexDegree = outDegree_[nbrId];
                        sortNbr_vec.emplace_back(nbrDegree);
                        bitset->set_bit_unsync(nbrId);
                    }
                }
                assert_msg(sortNbr_vec.size() <= sortDegree_vec[vertexNum_ - 1].vertexDegree, "sortNbr_vec预申请内存小了");

                // clang-format off
                std::sort(std::execution::par_unseq, sortNbr_vec.begin(), sortNbr_vec.end(),
                [&](const NbrDegree_type& a, const NbrDegree_type& b)
                    {
                        return std::tie(a.vertexDegree, a.vertexId) < std::tie(b.vertexDegree, b.vertexId);
                    }
                );
                // clang-format on

                for (count_type index = 0; index < sortNbr_vec.size(); index++)
                {
                    rcm_queue.push(sortNbr_vec[index].vertexId);
                }
            }
        }
        if constexpr (DEBUG_PROGRESS) bar->finish();

        constexpr bool isCheck = false;
        if constexpr (isCheck)
        {
            timer_total.start();
            for (count_type arrLen_bitset = 0; arrLen_bitset < (bitset->arrlen - 1); arrLen_bitset++)
            {
                assert_msg(bitset->array[arrLen_bitset] == std::numeric_limits<Fixed_Bitset::array_type>::max(), "最后bitset[0, %zu)应该全部为true",
                           (bitset->arrlen - 1) * sizeof(Fixed_Bitset::array_type) * 8);
            }
            for (count_type bitIndex_bitset = (bitset->arrlen - 1) * sizeof(Fixed_Bitset::array_type) * 8; bitIndex_bitset < vertexNum_;
                 bitIndex_bitset++)
            {
                assert_msg(bitset->get(bitIndex_bitset), "最后bitset[%zu, %zu]应该全部为true",
                           (bitset->arrlen - 1) * sizeof(Fixed_Bitset::array_type) * 8, SCU64(vertexNum_));
            }
            Msg_info("Bitset finishe check, Used time: %s", timer_total.get_time_str().c_str());
        }

        vertex_id_type temp;
        omp_par_for(count_type i = 0; i < vertexNum_ / 2; i++)
        {
            temp = rank[i];
            rank[i] = rank[vertexNum_ - i - 1];
            rank[vertexNum_ - i - 1] = temp;
        }
        Msg_info("Reorder finished, Used time: %s", timer_total.get_time_str().c_str());

        return rank;
    }

    void writeRankToFile(vertex_id_type* rank, std::string& filePath)
    {
        CPJ::Timer timer;
        CPJ::IOAdaptor ioAdapter(filePath);
        ioAdapter.openFile("w");
        ioAdapter.writeBinFile_sync(rank, vertexNum_ * sizeof(vertex_id_type), omp_get_max_threads());
        ioAdapter.closeFile();
        Msg_info("Rank write to [%s] finished, Used time: %s", filePath.c_str(), timer.get_time_str().c_str());
    }

    vertex_id_type* readRankFromFile(std::string& filePath)
    {
        CPJ::Timer timer;
        CPJ::IOAdaptor ioAdapter(filePath);
        ioAdapter.openFile();
        vertex_id_type* rank = ioAdapter.readBinFileEntire_sync<vertex_id_type>(omp_get_max_threads());
        ioAdapter.closeFile();
        Msg_info("Read Rank from [%s] finished, Used time: %s", filePath.c_str(), timer.get_time_str().c_str());

        return rank;
    }

    void checkRankResult(vertex_id_type* rank)
    {
        CPJ::Timer timer;

        bool hasDuplicates = CPJ::has_duplicates(rank, vertexNum_);
        assert_msg(!hasDuplicates, "Rank error");

        omp_par_for(count_type index = 0; index < vertexNum_; index++)
        {
            assert_msg(rank[index] < vertexNum_, "Rank error, error element: %zu", SCU64(rank[index]));
        }
        Msg_finish("Rank Result finish check, Used time: %s", timer.get_time_str().c_str());
    }
};
} // namespace CPJ