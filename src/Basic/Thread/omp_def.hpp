#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include <chrono>
#include <iostream>
#include <numa.h>
#include <omp.h>
#include <thread>

#define omp_parallel _Pragma("omp parallel")
#define omp_parallel_for _Pragma("omp parallel for") for
#define omp_parallel_for_1 _Pragma("omp parallel for schedule (static,1)") for
#define omp_parallel_for_256 _Pragma("omp parallel for schedule (static,256)") for
// #define omp_parallel_for_threads(threadNum) _Pragma("omp parallel for num_threads(#threadNum)") for

#define OMP_STRINGIFY(x) #x
#define OMP_TOSTRING(x) OMP_STRINGIFY(x)
#define omp_par_for _Pragma("omp parallel for") for
#define omp_par_for_threads(threadNum) _Pragma(OMP_TOSTRING(omp parallel for num_threads(threadNum))) for
#define omp_par _Pragma("omp parallel")
#define omp_par_threads(threadNum) _Pragma(OMP_TOSTRING(omp parallel num_threads(threadNum)))
#define omp_par_for_reductionAdd(sum) _Pragma(OMP_TOSTRING(omp parallel for reduction(+: sum))) for
#define omp_par_for_threads_reductionAdd(threadNum, sum) _Pragma(OMP_TOSTRING(omp parallel for num_threads(threadNum) reduction(+: sum))) for
#define omp_par_reductionAdd(sum) _Pragma(OMP_TOSTRING(omp parallel reduction(+ : sum)))
#define omp_par_threads_reductionAdd(threadNum, sum) _Pragma(OMP_TOSTRING(omp parallel num_threads(threadNum) reduction(+ : sum)))
#define omp_par_for_staticChunk(chunkSize) _Pragma(OMP_TOSTRING(omp parallel for schedule (static,chunkSize))) for
#define omp_par_for_threads_staticChunk(threadNum, chunkSize) _Pragma(OMP_TOSTRING(omp parallel for num_threads(threadNum) schedule (static,chunkSize))) for
// #define omp_par_staticChunk(chunkSize) _Pragma(OMP_TOSTRING(omp parallel schedule(static, chunkSize)))

// 无openMP时c++使用
// #define parallel_for for
// #define parallel_for_1 for
// #define parallel_for_256 for

/***********************************************************************
 *                              【CPU INFO】
 ***********************************************************************/
static uint64_t ThreadNum = omp_get_max_threads();
static uint64_t SocketNum = numa_num_configured_nodes();
static uint64_t ThreadPerSocket = ThreadNum / SocketNum;

inline uint64_t getThreadSocketId(uint64_t threadId) { return threadId / ThreadPerSocket; }

inline uint64_t getThreadSocketOffset(uint64_t threadId) { return threadId % ThreadPerSocket; }

static void reSetThreadNum(uint64_t threadNum)
{
    omp_set_num_threads(threadNum);
    ThreadNum = threadNum;
    ThreadPerSocket = ThreadNum / SocketNum;
}
