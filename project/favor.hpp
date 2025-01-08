#pragma once

#include "Basic/Type/data_type.hpp"
#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cub/cub.cuh>

__device__ __forceinline__ void sync_block() { __syncthreads(); }

template <typename T>
__device__ __forceinline__ T cuda_popc(T& a)
{
    if constexpr ((sizeof(T) == 1) || (sizeof(T) == 4))
    {
        return __popc(a);
    }
    else if constexpr (sizeof(T) == 8)
    {
        return __popcll(a);
    }
    else
    {
        assert(false);
    }
}

template <typename T>
__device__ __forceinline__ T cuda_min(const T& a, const T& b)
{
    return min(a, b);
}

// #define CUDA_KERNEL_CALL(kernel, nBlock, blockSize, ...) kernel<<<nBlock, blockSize>>>(__VA_ARGS__);

#ifdef __clang__
#define CUDA_KERNEL_CALL(kernel, nBlock, blockSize, ...) kernel(__VA_ARGS__)
#else
#define CUDA_KERNEL_CALL(kernel, nBlock, blockSize, ...) kernel<<<nBlock, blockSize>>>(__VA_ARGS__)
#endif

template <typename T>
using Thrust_CountIterator = thrust::counting_iterator<T>;