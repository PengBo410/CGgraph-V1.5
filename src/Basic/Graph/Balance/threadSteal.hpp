#pragma once

#include "Basic/Graph/Balance/threadState.hpp"

namespace CPJ {

namespace Balance {

class ThreadSteal
{

  private:
    ThreadState* threadState;

  public:
    ThreadState::ThreadState_type** thread_state;

    ThreadSteal()
    {
        omp_set_dynamic(0);
        threadState = new ThreadState();
        thread_state = threadState->thread_state;
    }

  public:
    template <typename result_type, typename workNum_type>
    result_type vertexLevel(workNum_type work, std::function<void(size_t&, result_type&)> enCoderTask, size_t alignSize = 64)
    {
        splitTask<workNum_type>(work, alignSize);

        result_type totalWorkloads = 0;
#pragma omp parallel reduction(+ : totalWorkloads)
        {
            size_t thread_id = omp_get_thread_num();
            result_type totalTask_local = 0;

            /*************************************
             *   2.1.【VERTEX_WORKING】
             *************************************/
            while (true)
            {
                size_t vertexId_current = __sync_fetch_and_add(&thread_state[thread_id]->cur, 64);
                if (vertexId_current >= thread_state[thread_id]->end) break;

                enCoderTask(vertexId_current,
                            totalTask_local); //[vertexId_current, vertexId_current + 64)

            } // end of [2.1.Vertex Working]

            /*************************************
             *   2.2.【VERTEX_STEALING】
             *************************************/
            thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
            for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++)
            {
                count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
                while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING)
                {
                    size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, 64);
                    if (vertexId_current >= thread_state[threadId_help]->end) break;

                    enCoderTask(vertexId_current,
                                totalTask_local); //[vertexId_current, vertexId_current + 64)
                }
            } // end of [2.2.VERTEX_STEALING]

            totalWorkloads += totalTask_local;
        }

        return totalWorkloads;
    }

  private:
    /* ======================================================================================*
     *                              [taskSteal_splitTask]
     * ======================================================================================*/
    template <typename T>
    void splitTask(T& workSize, size_t alignSize = 1, bool fillWord = false)
    {
        size_t bitNum = 8 * sizeof(size_t);
        if (fillWord) alignSize = bitNum;
        T taskSize = workSize;
        for (count_type threadId = 0; threadId < ThreadNum; threadId++)
        {
            if (fillWord && WORD_MOD(taskSize) != 0) taskSize = (taskSize / bitNum + 1) * bitNum;
            thread_state[threadId]->start = (taskSize / ThreadNum) / alignSize * alignSize * threadId;
            thread_state[threadId]->cur = thread_state[threadId]->start;
            thread_state[threadId]->end = (taskSize / ThreadNum) / alignSize * alignSize * (threadId + 1);
            thread_state[threadId]->edgeDonate.vertex = 0;
            thread_state[threadId]->edgeDonate.edge_cur = 0;
            thread_state[threadId]->edgeDonate.edge_end = 0;
            thread_state[threadId]->edgeDonate.edge_socket = 0;
            if (threadId == (ThreadNum - 1)) thread_state[threadId]->end = taskSize;
            thread_state[threadId]->status = ThreadState::VERTEX_WORKING;
        }
    }

}; // end of class [ThreadSteal]

} // namespace Balance
} // namespace CPJ