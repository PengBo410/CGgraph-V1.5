#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include <thread>

inline void bindCore(int coreId)
{
    auto coreNum = std::thread::hardware_concurrency();
    assert_msg(coreId < coreNum, "You coreId >= coreNum");

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(coreId, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    assert_msg(rc == 0, "The current thread can't bind core (%d)!", static_cast<uint32_t>(coreId));
}