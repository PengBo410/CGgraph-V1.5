#pragma once

#include <iomanip>

#include "Basic/Thread/omp_def.hpp"
#include "Basic/Type/data_type.hpp"

namespace CPJ {

class ThreadState {

private:
#ifdef LOCK
    std::vector<simple_spinlock> threadLock;
#endif // LOCK

public:
    enum ThreadStatus { VERTEX_WORKING,
        EDGE_DONATE,
        EDGE_HELP,
        VERTEX_STEALING };

    struct Edge_donate_type {
        vertex_id_type vertex = 0;
        countl_type edge_cur = 0;
        countl_type edge_end = 0;
        count_type edge_socket = 0;
    };

    struct ThreadState_type {
        uint64_t start = 0;
        uint64_t cur = 0;
        uint64_t end = 0;

        Edge_donate_type edgeDonate;

        ThreadStatus status;
    } CACHE_ALIGNED;

public:
    ThreadState_type** thread_state;

public:
    /*======================================================================================*
     *                                   【con】
     *======================================================================================*/
    ThreadState()
    {
        thread_state = new ThreadState_type*[ThreadNum];
        for (count_type threadId = 0; threadId < ThreadNum; threadId++) {
#ifdef NUMA_AWARE
            thread_state[threadId] = (ThreadState_type*)numa_alloc_onnode(sizeof(ThreadState_type), getThreadSocketId(threadId));
#else
            thread_state[threadId] = (ThreadState_type*)malloc(sizeof(ThreadState_type));
#endif // NUMA_AWARE
        }

#ifdef LOCK
        threadLock.resize(ThreadNum);
#endif // LOCK

#ifdef THREAD_STATE_DEBUG
        Msg_info("ThreadNum (%2u), sizeof(ThreadState_type) = %2u", ThreadNum, sizeof(ThreadState_type));
#endif // THREAD_STATE_DEBUG
    }

    void init_threadState()
    {
        for (count_type threadId = 0; threadId < ThreadNum; threadId++) {
            thread_state[threadId]->cur = thread_state[threadId]->start;
            thread_state[threadId]->edgeDonate.edge_cur = 0;
            thread_state[threadId]->edgeDonate.edge_end = 0;
            thread_state[threadId]->edgeDonate.edge_socket = 0;
            thread_state[threadId]->edgeDonate.vertex = 0;
            thread_state[threadId]->status = VERTEX_WORKING;
        }
    }

    void log_threadTask()
    {
        std::stringstream ss;
        ss << "\nprint:\n";
        for (count_type threadId = 0; threadId < ThreadNum; threadId++) {
            ss << "thread[" << std::setw(2) << threadId << "]task num (" << std::setw(10)
               << (thread_state[threadId]->end - thread_state[threadId]->cur) << "), range: [" << std::setw(10) << thread_state[threadId]->cur << ","
               << std::setw(10) << thread_state[threadId]->end << ")" << std::endl;
        }
        // logstream(LOG_INFO) << ss.str() << std::endl;
        std::cout << ss.str() << std::endl;
    }
};
}