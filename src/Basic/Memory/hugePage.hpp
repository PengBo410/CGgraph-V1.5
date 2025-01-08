
#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Other/errnoMsg.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory.h>
#include <string>
#include <sys/mman.h>

#include <fcntl.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

class HugePage
{
  public:
    static constexpr uint64_t PAGE_SIZE = 4096;         // 4KB
    static constexpr uint64_t HUGE_PAGE_SIZE = 2097152; // 2MB

    uint64_t TOTAL_INIT_HUGEPAGE_NUM = 0; // 程序初始时获取到的total_hugePageNum, 这个值只会被计算一次, 不会随着allocate hugePage而变化
    uint64_t FREE_INIT_HUGEPAGE_NUM = 0; // 程序初始时获取到的free_hugePageNum, 这个值只会被计算一次, 不会随着allocate hugePage而变化
    uint64_t RSVD_INIT_HUGEPAGE_NUM = 0; // 程序初始时获取到的rsvd_hugePageNum, 这个值只会被计算一次, 不会随着allocate hugePage而变化

  private:
    static constexpr bool DEBUG = false;

  public:
    HugePage() { init(); }

  public:
    void init()
    {
        if constexpr (DEBUG) Msg_warn("The function [init()] of Class [HugePage] Only Call Once");

        uint64_t pageSize = sysconf(_SC_PAGESIZE);
        uint64_t hugepageSize = getLableValue<uint64_t>("Hugepagesize:");
        assert_msg((pageSize != 0) && (pageSize == PAGE_SIZE),
                   "The detected pagesize does not match the predefined pagesize, detected = %zu (bytes), predefined = %zu (bytes)", pageSize,
                   PAGE_SIZE);
        assert_msg((pageSize != 0) && (KB(hugepageSize) == HUGE_PAGE_SIZE),
                   "The detected hugepageSize does not match the predefined hugepageSize, detected = %zu (bytes), predefined = %zu (bytes)",
                   KB(hugepageSize), HUGE_PAGE_SIZE);

        TOTAL_INIT_HUGEPAGE_NUM = getLableValue<uint64_t>("HugePages_Total:");
        FREE_INIT_HUGEPAGE_NUM = getLableValue<uint64_t>("HugePages_Free:");
        RSVD_INIT_HUGEPAGE_NUM = getLableValue<uint64_t>("HugePages_Rsvd:");
    }

  private:
    template <typename T>
    T getLableValue(std::string lable)
    {
        std::ifstream memInfo("/proc/meminfo");
        assert_msg(memInfo.is_open(), "Can not open [/proc/meminfo]");
        T value;

        std::string line;
        while (std::getline(memInfo, line))
        {
            if (line.compare(0, lable.length(), lable) == 0)
            {
                // 找到第一个非空格字符的位置
                line = line.substr(lable.length());
                size_t firstNonSpace = line.find_first_not_of(' ');
                std::istringstream iss(line.substr(firstNonSpace));
                iss >> value; // 自动跳过空格、换行符等空白字符，并一直读取到遇到非数字字符为止
                if constexpr (DEBUG) Msg_info("%s = %zu", lable.c_str(), SCU64(value));
                break;
            }
        }
        memInfo.close();
        return value;
    }

  public:
    // The follow function will call IO operator
    uint64_t getTotalHugePageNum() { return getLableValue<uint64_t>("HugePages_Total:"); }
    uint64_t getFreeHugePageNum() { return getLableValue<uint64_t>("HugePages_Free:"); }
    uint64_t getRsvdHugePageNum() { return getLableValue<uint64_t>("HugePages_Rsvd:"); }
};

template <typename T>
inline T* allocateHugePage(size_t bytes)
{
    T* res = (T*)mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    assert_msg_clear(res != MAP_FAILED, "Failed to allocateHugePage, errMsg = %s", getErrorMsg(errno).c_str());
    return res;
clear:
    STOP;
    return nullptr;
}

/* *********************************************************************************************************
 * @description: 在申请内存的同时, 在分配内存的时候提前把页面映射好，从而避免运行时的缺页异常。
 *               对应于: MAP_POPULATE
 * @param [size_t] bytes
 * @return [*]
 * *********************************************************************************************************/
template <typename T>
inline T* allocateHugePage_mapPhy(size_t bytes)
{
    T* res = (T*)mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
    assert_msg_clear(res != MAP_FAILED, "Failed to allocateHugePage, errMsg = %s", getErrorMsg(errno).c_str());
    return res;
clear:
    STOP;
    return nullptr;
}

template <typename T>
inline bool freeHugePage(T* mapped_addr, size_t bytes)
{
    int errCode = munmap(mapped_addr, bytes);
    assert_msg_clear(errCode == 0, "Failed to free Huge Page, errMsg = %s", getErrorMsg(errno).c_str());
    return true;
clear:
    STOP;
    return false;
}

inline HugePage& global_hugePage()
{
    static HugePage hugePage;
    return hugePage;
}

inline uint64_t PAGE_SIZE() { return global_hugePage().PAGE_SIZE; }
inline uint64_t HUGE_PAGE_SIZE() { return global_hugePage().HUGE_PAGE_SIZE; }
inline uint64_t HUGE_PAGE_NUM_FREE_INIT() { return global_hugePage().FREE_INIT_HUGEPAGE_NUM; }
inline uint64_t HUGE_PAGE_NUM_TOTAL_INIT() { return global_hugePage().TOTAL_INIT_HUGEPAGE_NUM; }
inline uint64_t HUGE_PAGE_NUM_RSVD_INIT() { return global_hugePage().RSVD_INIT_HUGEPAGE_NUM; }
uint32_t HUGE_PAGE_NUM_FREE() { return global_hugePage().getFreeHugePageNum(); }   // Will Call IO
uint32_t HUGE_PAGE_NUM_TOTAL() { return global_hugePage().getTotalHugePageNum(); } // Will Call IO
uint32_t HUGE_PAGE_NUM_RSVD() { return global_hugePage().getRsvdHugePageNum(); }   // Will Call IO