#pragma once

#include "Basic/CUDA/cuda_check.cuh"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Memory/hugePage.hpp"
#include <cstddef>
#include <cstdio>
#include <fstream> // std::ifstream
#include <sstream> // std::istringstream
#include <string>
#include <sys/resource.h>

namespace CPJ {

class MemoryInfo {
public:
    MemoryInfo() { }

    //! Device
    static size_t getMemoryTotal_Device(int useDeviceId = 0)
    {
        size_t freeMemory_device_ = 0;
        size_t totalMemory_device_ = 0;
        getGPUMemory(freeMemory_device_, totalMemory_device_, useDeviceId);
        return totalMemory_device_;
    }

    static size_t getMemoryFree_Device(int useDeviceId = 0)
    {
        size_t freeMemory_device_ = 0;
        size_t totalMemory_device_ = 0;
        getGPUMemory(freeMemory_device_, totalMemory_device_, useDeviceId);
        return freeMemory_device_;
    }

    static size_t getMemoryAvailable_Device(int useDeviceId = 0)
    {
        size_t freeMemory_device_ = 0;
        size_t totalMemory_device_ = 0;
        getGPUMemory(freeMemory_device_, totalMemory_device_, useDeviceId);
        return freeMemory_device_;
    }

    static std::string getMemoryInfo_Device(int useDeviceId = 0)
    {
        size_t freeMemory_device_ = 0;
        size_t totalMemory_device_ = 0;
        CUDA_CHECK(cudaSetDevice(useDeviceId));
        CUDA_CHECK(cudaMemGetInfo(&freeMemory_device_, &totalMemory_device_));
        char buffer[256];
        std::sprintf(buffer,
            "The Device[%d]: total memory = %.2lf (GB), used memory = %.2lf (GB), free memory = %.2lf (GB), available memory = %.2lf (GB)",
            useDeviceId, BYTES_TO_GB(SCU64(totalMemory_device_)), BYTES_TO_GB(SCU64(totalMemory_device_ - freeMemory_device_)),
            BYTES_TO_GB(SCD(freeMemory_device_)), BYTES_TO_GB(SCD(freeMemory_device_)));
        return std::string(buffer);
    }

    //! Host
    static size_t getMemoryTotal_Host()
    {
        size_t totalMemory_host_ = 0;
        size_t freeMemory_host_ = 0;
        size_t availableMemory_host_ = 0;
        getMemoryInfo(totalMemory_host_, freeMemory_host_, availableMemory_host_);
        return totalMemory_host_;
    }
    /* 这个是未分配的 */
    static size_t getMemoryFree_Host()
    {
        size_t totalMemory_host_ = 0;
        size_t freeMemory_host_ = 0;
        size_t availableMemory_host_ = 0;
        getMemoryInfo(totalMemory_host_, freeMemory_host_, availableMemory_host_);
        return freeMemory_host_;
    }
    /* 这个是实际可用的 */
    static size_t getMemoryAvailable_Host()
    {
        size_t totalMemory_host_ = 0;
        size_t freeMemory_host_ = 0;
        size_t availableMemory_host_ = 0;
        getMemoryInfo(totalMemory_host_, freeMemory_host_, availableMemory_host_);
        return availableMemory_host_;
    }

    static std::string getMemoryInfo_Host()
    {
        size_t totalMemory_host_ = 0;
        size_t freeMemory_host_ = 0;
        size_t availableMemory_host_ = 0;
        getMemoryInfo(totalMemory_host_, freeMemory_host_, availableMemory_host_);
        size_t usedMemory = totalMemory_host_ - freeMemory_host_;

        char buffer[256];
        std::sprintf(buffer, "The Host: total memory = %.2lf (GB), used memory = %.2lf (GB), free memory = %.2lf (GB), available memory = %.2lf (GB)",
            BYTES_TO_GB(totalMemory_host_), BYTES_TO_GB(usedMemory), BYTES_TO_GB(freeMemory_host_), BYTES_TO_GB(availableMemory_host_));
        return std::string(buffer);
    }

    static size_t getHugePageTotal_Host() { return global_hugePage().getTotalHugePageNum() * global_hugePage().HUGE_PAGE_SIZE; }
    static size_t getHugePageFree_Host() { return global_hugePage().getFreeHugePageNum() * global_hugePage().HUGE_PAGE_SIZE; }
    static size_t getHugePageRsvd_Host() { return global_hugePage().getRsvdHugePageNum() * global_hugePage().HUGE_PAGE_SIZE; }

    static std::string getHugePageInfo_Host()
    {
        char buffer[256];
        std::sprintf(buffer,
            "The HugePage: total HugePage = %.2lf (GB), used HugePage = %.2lf (GB), free memory = %.2lf (GB), available memory = %.2lf (GB)",
            BYTES_TO_GB(getHugePageTotal_Host()), BYTES_TO_GB(getHugePageTotal_Host() - getHugePageFree_Host()),
            BYTES_TO_GB(getHugePageFree_Host()), BYTES_TO_GB(getHugePageFree_Host()));
        return std::string(buffer);
    }

private:
    static void getGPUMemory(size_t& freeMemory, size_t& totalMemory, int useDeviceId = 0)
    {
        CUDA_CHECK(cudaSetDevice(useDeviceId));
        CUDA_CHECK(cudaMemGetInfo(&freeMemory, &totalMemory));
    }

    static void getMemoryInfo(size_t& totalMemory, size_t& freeMemory, size_t& availableMemory)
    {
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal:") == 0) {
                totalMemory = parseLine(line);
            } else if (line.find("MemFree:") == 0) {
                freeMemory = parseLine(line);
            } else if (line.find("MemAvailable:") == 0) {
                availableMemory = parseLine(line);
            }
        }
    }

    static size_t parseLine(const std::string& line)
    {
        size_t value = 0;
        std::istringstream iss(line);
        std::string key;
        std::string unit;
        iss >> key >> value >> unit;
        return value * 1024; // The value is in kB, converting to bytes
    }
};

class Monitor {
private:
    struct rusage start_usage;
    struct rusage end_usage;

    size_t start_vMemory = 0;
    size_t end_vMemory = 0;

public:
    Monitor()
    {
        start_getUsage();
        start_vMemory = getCurrentVirtualMemorySize();
    }

    // 以KB为单位
    size_t getCur_maxPhysicalMemory_KB()
    {
        end_endUsage();
        return static_cast<size_t>(end_usage.ru_maxrss);
    }

    // free 会减少虚拟内存
    size_t getCur_virualMemory_KB()
    {
        end_getVmemory();
        return (end_vMemory - start_vMemory) / 1024;
    }

    void start_getUsage() { getrusage(RUSAGE_SELF, &start_usage); }
    void end_endUsage() { getrusage(RUSAGE_SELF, &end_usage); }
    void start_getVmemory() { start_vMemory = getCurrentVirtualMemorySize(); }
    void end_getVmemory() { end_vMemory = getCurrentVirtualMemorySize(); }

private:
    std::size_t getCurrentVirtualMemorySize()
    {
        std::ifstream stat_stream("/proc/self/statm");
        std::size_t size;
        stat_stream >> size;
        return size * sysconf(_SC_PAGESIZE); // sysconf(_SC_PAGESIZE) 获取每一页的大小
    }
};

class MemUsage {
public:
    static double getCur_maxPhyUsage_KB()
    {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        return static_cast<double>(usage.ru_maxrss);
    }
    /* 当前程序使用的 */
    static double getCur_maxPhyUsage_MB() { return (getCur_maxPhyUsage_KB() / 1024); }
    static double getCur_maxPhyUsage_GB() { return (getCur_maxPhyUsage_KB() / 1024 / 1024); }

    static size_t getCur_virUsage_BYTE()
    {
        std::ifstream stat_stream("/proc/self/statm", std::ios_base::in);
        std::size_t size;
        stat_stream >> size;
        return size * sysconf(_SC_PAGESIZE);
    }
    static double getCur_virUsage_MB() { return BYTES_TO_MB(SCD(getCur_virUsage_BYTE())); }
    static double getCur_virUsage_GB() { return BYTES_TO_GB(SCD(getCur_virUsage_BYTE())); }

private:
    static void printfMemoryUsage()
    {
        std::ifstream mem_file("/proc/self/stat", std::ios_base::in);
        std::string ignore; // 用于忽略不需要的字段。
        int64_t pagenum; // 用于存储页面数量
        uint64_t vm_byte; // 用于存储虚拟内存字节数。
        double vm_usage; // 用于存储虚拟内存使用量（以MB为单位）
        const double BYTES_TO_MB = 1.0 / (1024 * 1024);
        mem_file >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> vm_byte >> pagenum;
        vm_usage = vm_byte * BYTES_TO_MB;

        double pm_usage;
        pm_usage = pagenum * getpagesize() * BYTES_TO_MB;
        Msg_info("alloc_size: 物理内存(%.2lf) MB, 虚拟内存(%.2lf) MB", pm_usage, vm_usage);
    }
};

} // namespace CPJ