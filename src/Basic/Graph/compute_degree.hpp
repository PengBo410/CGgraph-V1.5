#pragma once

#include "Basic/Graph/Balance/threadSteal.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Timer/timer_CPJ.hpp"
#include "Basic/Type/data_type.hpp"

namespace CPJ {

class Compute_degree
{

  public:
    count_type vertexNum;
    countl_type edgeNum;
    count_type zeroOutDegreeNum;
    count_type noZeroOutDegreeNum;

    countl_type* csr_offset;
    vertex_id_type* csr_dest;
    edge_data_type* csr_weight;
    degree_type* outDegree;

    degree_type* inDegree;
    bool hasIndegree = false;
    bool hasOutdegree = false;

  public:
    Compute_degree(const CSR_Result_type& csrResult) : vertexNum(0), edgeNum(0), zeroOutDegreeNum(0), noZeroOutDegreeNum(0)
    {
        // Get Graph CSR
        vertexNum = csrResult.vertexNum;
        edgeNum = csrResult.edgeNum;
        csr_offset = csrResult.csr_offset;
        csr_dest = csrResult.csr_dest;
        csr_weight = csrResult.csr_weight;

        noZeroOutDegreeNum = csrResult.noZeroOutDegreeNum;
        zeroOutDegreeNum = vertexNum - noZeroOutDegreeNum;
        Msg_info("zeroOutDegreeNum = %zu (%.2lf), noZeroOutDegreeNum = %zu (%.2f)", static_cast<uint64_t>(zeroOutDegreeNum),
                 (double)zeroOutDegreeNum / vertexNum * 100, static_cast<uint64_t>(noZeroOutDegreeNum), (double)noZeroOutDegreeNum / vertexNum * 100);

        CPJ::Timer constructTime;
        initGraph();
        Msg_info("Compute-outDegree: Used time: %.2lf (ms)", constructTime.get_time_ms());
    }

    degree_type* getIndegree()
    {
        if (hasIndegree) return inDegree;
        else
        {
            CPJ::Timer constructTime;
            compute_inDegree();
            Msg_info("Compute-inDegree: Used time: %.2lf (ms)", constructTime.get_time_ms());

            if (hasIndegree) return inDegree;
            else
            {
                assert_msg(false, "inDegree has not finish");
                return nullptr;
            }
        }
    }

    degree_type* getOutdegree()
    {
        if (hasOutdegree) return outDegree;
        else
        {
            assert_msg(false, "outDegree has not finish");
            return nullptr;
        }
    }

  private:
    /* **********************************************************
     * Func: Host Function , Init Graph
     * **********************************************************/
    void initGraph()
    {
        if (vertexNum >= std::numeric_limits<count_type>::max())
        {
            assert_msg(false, "vertexNum >= count_type:max()");
        }
        if (edgeNum >= std::numeric_limits<countl_type>::max())
        {
            assert_msg(false, "vertexNum >= countl_type:max()");
        }

        // Init outDegree
        inDegree = new degree_type[vertexNum];
        memset(inDegree, 0, sizeof(degree_type) * vertexNum);

        outDegree = new degree_type[vertexNum];
        memset(outDegree, 0, sizeof(degree_type) * vertexNum);

        // Get outDegree
        hasOutdegree = true;
        omp_parallel_for(vertex_id_type vertexId = 0; vertexId < vertexNum; vertexId++)
        {
            outDegree[vertexId] = csr_offset[vertexId + 1] - csr_offset[vertexId];
        }
        // check
        bool check = true;
        if (check)
        {
            countl_type edgeNum_total = 0;
#pragma omp parallel for reduction(+ : edgeNum_total)
            for (count_type vertexId = 0; vertexId < vertexNum; vertexId++)
            {
                edgeNum_total += outDegree[vertexId];
            }

            assert_msg((edgeNum_total == edgeNum), "edgeNum_total = %zu", static_cast<uint64_t>(edgeNum_total));
        }

    } // end of func [initGraph()]

    /* **********************************************************
     * Func: Host Function , Compute-inDegree
     * **********************************************************/
    void compute_inDegree()
    {
        CPJ::Balance::ThreadSteal* taskSteal = new Balance::ThreadSteal();
        taskSteal->vertexLevel<size_t, size_t>(
            static_cast<size_t>(noZeroOutDegreeNum),
            [&](size_t& current, size_t& local_workloads)
            {
            size_t end = current + VERTEXWORK_CHUNK;
            size_t length = VERTEXWORK_CHUNK;
            if (end >= noZeroOutDegreeNum) length = noZeroOutDegreeNum - current;

            for (size_t in = 0; in < length; in++)
            {
                countl_type nbr_first = csr_offset[current + in];
                countl_type nbr_end = csr_offset[current + in + 1];

                for (countl_type nbrId = nbr_first; nbrId < nbr_end; nbrId++)
                {
                    vertex_id_type dest = csr_dest[nbrId];
                    __sync_fetch_and_add(&inDegree[dest], 1);
                }
            }
            },
            VERTEXWORK_CHUNK);

        hasIndegree = true;

        // check
        bool check = true;
        if (check)
        {
            countl_type edgeNum_total = 0;
#pragma omp parallel for reduction(+ : edgeNum_total)
            for (count_type vertexId = 0; vertexId < vertexNum; vertexId++)
            {
                edgeNum_total += inDegree[vertexId];
            }

            assert_msg((edgeNum_total == edgeNum), "edgeNum_total = %zu", static_cast<uint64_t>(edgeNum_total));
        }
    }

}; // end of class [Compute_indegree]

} // namespace CPJ