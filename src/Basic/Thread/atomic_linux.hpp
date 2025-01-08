#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include <assert.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace LinuxAtomic {

template <class T>
inline bool cas(T* ptr, T old_val, T new_val)
{
    if constexpr (sizeof(T) == 8)
    {
        return __sync_bool_compare_and_swap((long*)ptr, *((long*)&old_val), *((long*)&new_val));
    }
    else if constexpr (sizeof(T) == 4)
    {
        return __sync_bool_compare_and_swap((int*)ptr, *((int*)&old_val), *((int*)&new_val));
    }
    else
    {
        assert(false);
        return static_cast<bool>(0);
    }
}

template <class T>
inline bool write_min(T* ptr, T val)
{
    volatile T curr_val;
    bool done = false;
    do
    {
        curr_val = *ptr;
    }
    while (curr_val > val && !(done = cas(ptr, curr_val, val)));
    return done;
}

template <class T>
inline bool cas_std(std::atomic<T>* ptr, T old_val, T new_val, std::memory_order memOrder)
{
    // 使用std::atomic的compare_exchange_strong进行原子比较与交换
    return ptr->compare_exchange_strong(old_val, new_val, memOrder);
}

template <class T>
inline bool write_min_std(std::atomic<T>* ptr, T val, std::memory_order memOrder)
{
    T curr_val;
    bool done = false;
    do
    {
        curr_val = ptr->load(); // 读取当前值
    }
    while (curr_val > val && !(done = cas_std(ptr, curr_val, val, memOrder))); // 若当前值大于val，尝试进行CAS
    return done;
}

template <class ET>
inline bool write_max(ET* a, ET b)
{
    ET c;
    bool r = 0;
    do
    {
        c = *a;
    }
    while (c < b && !(r = cas(a, c, b)));
    return r;
}

template <class T>
inline void write_add(T* ptr, T val)
{
    volatile T new_val, old_val;
    do
    {
        old_val = *ptr;
        new_val = old_val + val;
    }
    while (!cas(ptr, old_val, new_val));
}

/* **********************************************************
 *  Func: 原子比较两个值的大小
 *        若   (ptr > val) return true;
 *        否则             return false
 * **********************************************************/
template <class T>
inline bool atomic_large(T* ptr, T val)
{
    volatile T curr_val;
    bool done = false;

    do
    {
        curr_val = *ptr;
        done = curr_val > val;
    }
    while (!cas(ptr, curr_val, curr_val));

    return done;
}

/* **********************************************************
 *  Func: 原子比较两个值的大小
 *        若   (ptr >= val) return true;
 *        否则             return false
 * **********************************************************/
template <class T>
inline bool atomic_largeEqu(T* ptr, T val)
{
    volatile T curr_val;
    bool done = false;

    do
    {
        curr_val = *ptr;
        done = curr_val >= val;
    }
    while (!cas(ptr, curr_val, curr_val));

    return done;
}

/* **********************************************************
 *  Func: 原子比较两个值的大小
 *        若   (ptr <= val) return true;
 *        否则             return false
 * **********************************************************/
template <class T>
inline bool atomic_smallEqu(T* ptr, T val)
{
    volatile T curr_val;
    bool done = false;

    do
    {
        curr_val = *ptr;
        done = curr_val <= val;
    }
    while (!cas(ptr, curr_val, curr_val));

    return done;
}

/* **********************************************************
 *  Func: 原子比较两个值的大小,并返回差值
 * **********************************************************/
template <class T>
inline int64_t atomic_length(T* ptr, T val)
{
    volatile T curr_val;
    int64_t length = 0;

    do
    {
        curr_val = *ptr;
        length = static_cast<int64_t>(curr_val) - static_cast<int64_t>(val);
    }
    while (!cas(ptr, curr_val, curr_val));

    return length;
}

namespace CUDA {

template <typename T, size_t n>
struct AtomicAddIntegerImpl;

// Code from Pytorch
template <typename T>
struct AtomicAddIntegerImpl<T, 1>
{
    inline __device__ T operator()(T* address, T val)
    {
        size_t offset = (size_t)address & 3;
        uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
        uint32_t old = *address_as_ui;
        uint32_t shift = offset * 8;
        uint32_t old_byte;
        uint32_t newval;
        uint32_t assumed;

        do
        {
            assumed = old;
            old_byte = (old >> shift) & 0xff;
            // preserve size in initial cast. Casting directly to uint32_t pads
            // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
            newval = static_cast<uint8_t>(val + old_byte);
            newval = (old & ~(0x000000ff << shift)) | (newval << shift);
            old = atomicCAS(address_as_ui, assumed, newval);
        }
        while (assumed != old);

        return old_byte;
    }
};

template <typename T>
struct AtomicAddIntegerImpl<T, 2>
{
    inline __device__ T operator()(T* address, T val)
    {
        size_t offset = (size_t)address & 2;
        uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
        bool is_32_align = offset;
        uint32_t old = *address_as_ui;
        uint32_t old_bytes;
        uint32_t newval;
        uint32_t assumed;

        do
        {
            assumed = old;
            old_bytes = is_32_align ? old >> 16 : old & 0xffff;
            // preserve size in initial cast. Casting directly to uint32_t pads
            // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
            newval = static_cast<uint16_t>(val + old_bytes);
            newval = is_32_align ? (old & 0xffff) | (newval << 16) : (old & 0xffff0000) | newval;
            old = atomicCAS(address_as_ui, assumed, newval);
        }
        while (assumed != old);

        return old_bytes;
    }
};

inline __device__ uint8_t atomicAdd(uint8_t* address, uint8_t val) { return AtomicAddIntegerImpl<uint8_t, sizeof(uint8_t)>()(address, val); }

inline __device__ uint16_t atomicAdd(uint16_t* address, uint16_t val) { return AtomicAddIntegerImpl<uint16_t, sizeof(uint16_t)>()(address, val); }

inline __device__ uint32_t atomicAdd(uint32_t* address, uint32_t val) { return ::atomicAdd(address, val); }

inline __device__ uint64_t atomicAdd(uint64_t* address, uint64_t val)
{
    static_assert(sizeof(uint64_t) == sizeof(unsigned long long));
    return ::atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

//!--- atomicOr

template <typename T, size_t n>
struct AtomicOrIntegerImpl;

// uint8_t 版本的实现
template <typename T>
struct AtomicOrIntegerImpl<T, 1>
{
    inline __device__ T operator()(T* address, T val)
    {
        size_t offset = (size_t)address & 3;
        uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
        uint32_t old = *address_as_ui;
        uint32_t shift = offset * 8;
        uint32_t old_byte;
        uint32_t newval;
        uint32_t assumed;

        do
        {
            assumed = old;
            old_byte = (old >> shift) & 0xff;
            newval = old_byte | val;
            newval = (old & ~(0x000000ff << shift)) | (newval << shift);
            old = atomicCAS(address_as_ui, assumed, newval);
        }
        while (assumed != old);

        return old_byte;
    }
};

// uint16_t 版本的实现
template <typename T>
struct AtomicOrIntegerImpl<T, 2>
{
    inline __device__ T operator()(T* address, T val)
    {
        size_t offset = (size_t)address & 2;
        uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
        bool is_32_align = offset;
        uint32_t old = *address_as_ui;
        uint32_t old_bytes;
        uint32_t newval;
        uint32_t assumed;

        do
        {
            assumed = old;
            old_bytes = is_32_align ? old >> 16 : old & 0xffff;
            newval = old_bytes | val;
            newval = is_32_align ? (old & 0xffff) | (newval << 16) : (old & 0xffff0000) | newval;
            old = atomicCAS(address_as_ui, assumed, newval);
        }
        while (assumed != old);

        return old_bytes;
    }
};

// 包装函数，提供更简单的接口
inline __device__ uint8_t atomicOr(uint8_t* address, uint8_t val) { return AtomicOrIntegerImpl<uint8_t, sizeof(uint8_t)>()(address, val); }

inline __device__ uint16_t atomicOr(uint16_t* address, uint16_t val) { return AtomicOrIntegerImpl<uint16_t, sizeof(uint16_t)>()(address, val); }

inline __device__ uint32_t atomicOr(uint32_t* address, uint32_t val) { return ::atomicOr(address, val); }

inline __device__ uint32_t atomicOr(int* address, int val) { return ::atomicOr(address, val); }

inline __device__ uint64_t atomicOr(uint64_t* address, uint64_t val)
{
    static_assert(sizeof(uint64_t) == sizeof(unsigned long long));
    return ::atomicOr((unsigned long long*)address, (unsigned long long)val);
}

inline __device__ uint32_t atomicMin(uint32_t* address, uint32_t val) { return ::atomicMin(address, val); }

inline __device__ uint64_t atomicMin(uint64_t* address, uint64_t val)
{
    static_assert(sizeof(uint64_t) == sizeof(unsigned long long), "uint64_t and unsigned long long must be of the same size.");
    return ::atomicMin((unsigned long long*)address, (unsigned long long)val);
}

} // namespace CUDA

} // namespace LinuxAtomic