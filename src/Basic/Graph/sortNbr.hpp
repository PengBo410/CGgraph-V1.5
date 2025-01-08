#pragma once

#include "Basic/Graph/Balance/threadSteal.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Timer/timer_CPJ.hpp"

namespace CPJ {

#define VERTEXWORK_CHUNK 64

void sortNbr(CSR_Result_type& csrResult_old, bool asc = true, bool check = true)
{
    Balance::ThreadSteal* threadSteal = new Balance::ThreadSteal(); // compute task
    size_t totalWorkloads = 0;

    CPJ::Timer sort_time;
    totalWorkloads = threadSteal->vertexLevel<size_t, size_t>(
        static_cast<size_t>(csrResult_old.vertexNum),
        [&](size_t& current, size_t& local_workloads)
        {
        size_t end = current + VERTEXWORK_CHUNK;
        size_t length = VERTEXWORK_CHUNK;
        if (end >= csrResult_old.vertexNum) length = csrResult_old.vertexNum - current;

        for (size_t in = 0; in < length; in++)
        {
            countl_type nbr_first = csrResult_old.csr_offset[current + in];
            countl_type nbr_end = csrResult_old.csr_offset[current + in + 1];

            std::vector<std::pair<vertex_id_type, edge_data_type>> nbr;
            nbr.resize(nbr_end - nbr_first);

            for (countl_type i = nbr_first; i < nbr_end; i++)
            {
                nbr[i - nbr_first] = std::make_pair(csrResult_old.csr_dest[i], csrResult_old.csr_weight[i]);
            }

            sort(nbr.begin(), nbr.end(),
                 [&](const std::pair<vertex_id_type, edge_data_type>& a, const std::pair<vertex_id_type, edge_data_type>& b) -> bool
                 {
                if (asc)
                {
                    if (a.first < b.first) return true;
                    else return false;
                }
                else
                {
                    if (a.first > b.first) return true;
                    else return false;
                }
            });

            for (countl_type i = nbr_first; i < nbr_end; i++)
            {
                csrResult_old.csr_dest[i] = nbr[i - nbr_first].first;
                csrResult_old.csr_weight[i] = nbr[i - nbr_first].second;
            }
        }
        },
        64);
    Msg_info("nbrSort_taskSteal Used time: %.2f(ms), totalWorkloads = %lu", sort_time.get_time_ms(), totalWorkloads);

    if (check)
    {
        sort_time.start();
        totalWorkloads = threadSteal->vertexLevel<size_t, size_t>(
            static_cast<size_t>(csrResult_old.vertexNum),
            [&](size_t& current, size_t& local_workloads)
            {
            size_t end = current + VERTEXWORK_CHUNK;
            size_t length = VERTEXWORK_CHUNK;
            if (end >= csrResult_old.vertexNum) length = csrResult_old.vertexNum - current;

            for (size_t in = 0; in < length; in++)
            {
                countl_type nbr_first = csrResult_old.csr_offset[current + in];
                countl_type nbr_end = csrResult_old.csr_offset[current + in + 1];

                if ((nbr_end - nbr_first) > 1)
                {
                    for (countl_type nbrId = nbr_first; nbrId < (nbr_end - 1); nbrId++)
                    {
                        if (asc)
                        {
                            assert_msg((csrResult_old.csr_dest[nbrId] < csrResult_old.csr_dest[nbrId + 1]), "nbrSort is not sort");
                        }
                        else
                        {
                            assert_msg((csrResult_old.csr_dest[nbrId] > csrResult_old.csr_dest[nbrId + 1]), "nbrSort is not sort");
                        }
                    }
                }
            }
            },
            VERTEXWORK_CHUNK);
        Msg_finish("nbrSort_taskSteal Finished Checked, Used time: %.2lf(ms)", sort_time.get_time_ms());
    }
}

void sortNbr_noWeight(CSR_Result_type& csrResult_old, bool asc = true, bool check = true)
{
    Balance::ThreadSteal* threadSteal = new Balance::ThreadSteal(); // compute task
    size_t totalWorkloads = 0;

    CPJ::Timer sort_time;
    totalWorkloads = threadSteal->vertexLevel<size_t, size_t>(
        static_cast<size_t>(csrResult_old.vertexNum),
        [&](size_t& current, size_t& local_workloads)
        {
        size_t end = current + VERTEXWORK_CHUNK;
        size_t length = VERTEXWORK_CHUNK;
        if (end >= csrResult_old.vertexNum) length = csrResult_old.vertexNum - current;

        for (size_t in = 0; in < length; in++)
        {
            countl_type nbr_first = csrResult_old.csr_offset[current + in];
            countl_type nbr_end = csrResult_old.csr_offset[current + in + 1];

            std::vector<vertex_id_type> nbr;
            nbr.resize(nbr_end - nbr_first);

            for (countl_type i = nbr_first; i < nbr_end; i++)
            {
                nbr[i - nbr_first] = csrResult_old.csr_dest[i];
            }

            sort(nbr.begin(), nbr.end(),
                 [&](const vertex_id_type& a, const vertex_id_type& b) -> bool
                 {
                if (asc)
                {
                    if (a < b) return true;
                    else return false;
                }
                else
                {
                    if (a > b) return true;
                    else return false;
                }
            });

            for (countl_type i = nbr_first; i < nbr_end; i++)
            {
                csrResult_old.csr_dest[i] = nbr[i - nbr_first];
            }
        }
        },
        64);
    Msg_info("Function [%s] Used time: %.2f(ms), totalWorkloads = %lu", __FUNCTION__, sort_time.get_time_ms(), totalWorkloads);

    if (check)
    {
        sort_time.start();
        totalWorkloads = threadSteal->vertexLevel<size_t, size_t>(
            static_cast<size_t>(csrResult_old.vertexNum),
            [&](size_t& current, size_t& local_workloads)
            {
            size_t end = current + VERTEXWORK_CHUNK;
            size_t length = VERTEXWORK_CHUNK;
            if (end >= csrResult_old.vertexNum) length = csrResult_old.vertexNum - current;

            for (size_t in = 0; in < length; in++)
            {
                countl_type nbr_first = csrResult_old.csr_offset[current + in];
                countl_type nbr_end = csrResult_old.csr_offset[current + in + 1];

                if ((nbr_end - nbr_first) > 1)
                {
                    for (countl_type nbrId = nbr_first; nbrId < (nbr_end - 1); nbrId++)
                    {
                        if (asc)
                        {
                            assert_msg((csrResult_old.csr_dest[nbrId] < csrResult_old.csr_dest[nbrId + 1]), "nbrSort is not sort");
                        }
                        else
                        {
                            assert_msg((csrResult_old.csr_dest[nbrId] > csrResult_old.csr_dest[nbrId + 1]), "nbrSort is not sort");
                        }
                    }
                }
            }
            },
            VERTEXWORK_CHUNK);
        Msg_finish("Function [%s] Finished Checked, Used time: %.2lf(ms)", __FUNCTION__, sort_time.get_time_ms());
    }
}

} // namespace CPJ