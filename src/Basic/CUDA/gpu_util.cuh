#pragma once

#include "cuda_check.cuh"
#include "cuda_include.cuh"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <sys/time.h>
#include <type_traits>

#define FUNC

#define FLAG_H2D cudaMemcpyHostToDevice
#define FLAG_D2H cudaMemcpyDeviceToHost
#define FLAG_UVM cudaMemAdviseSetReadMostly
#define FLAG_ZERO_COPY cudaMemAdviseSetAccessedBy

#ifdef FUNC

/***********************************************************************************************
 *  Call: CUDA_CHECK(MALLOC_HOST(&csr_offset_host_, (noZeroOutDegreeNum + 1)));
 *  Call: CUDA_CHECK(MEMSET_HOST(csr_dest_host_, edgeNum_device_vec[deviceId]));
 *  Call: CUDA_CHECK(MEMSET_DEVICE(csr_dest_device_, edgeNum_device_vec[deviceId]));
 *  Call: CUDA_CHECK(H2D(csr_offset_device_, csr_offset_host_, (noZeroOutDegreeNum + 1)));
 ***********************************************************************************************/

template <typename T>
cudaError_t H2D(T* device_pointer, T* host_pointer, uint64_t size)
{
    return cudaMemcpy((device_pointer), (host_pointer), sizeof(T) * (size), FLAG_H2D);
}

template <typename T>
cudaError_t D2H(T* host_pointer, T* device_pointer, uint64_t size)
{
    return cudaMemcpy((host_pointer), (device_pointer), sizeof(T) * (size), FLAG_D2H);
}

template <typename T>
cudaError_t MALLOC_HOST(T** host_pointer, uint64_t size)
{
    return cudaMallocHost((void**)host_pointer, sizeof(T) * (size));
}

template <typename T>
cudaError_t MALLOC_DEVICE(T** device_pointer, uint64_t size)
{
    return cudaMalloc((void**)device_pointer, sizeof(T) * (size));
}

template <typename T>
void MEMSET_HOST(T* host_pointer, uint64_t size)
{
    memset(host_pointer, 0, sizeof(T) * (size));
}

template <typename T>
cudaError_t MALLOC_UVM(T** device_pointer, uint64_t size, int useDeviceId = 0)
{
    CUDA_CHECK(cudaMallocManaged((void**)device_pointer, size * sizeof(T)));
    return cudaMemAdvise(*device_pointer, size * sizeof(T), FLAG_UVM, useDeviceId);
}

template <typename T>
cudaError_t MALLOC_ZEROCOPY(T** device_pointer, uint64_t size, int useDeviceId = 0)
{
    CUDA_CHECK(cudaMallocManaged((void**)device_pointer, size * sizeof(T)));
    return cudaMemAdvise(*device_pointer, size * sizeof(T), FLAG_ZERO_COPY, useDeviceId);
}

template <typename T>
cudaError_t MEMSET_DEVICE(T* device_pointer, uint64_t size)
{
    return cudaMemset(device_pointer, 0, sizeof(T) * (size));
}

template <typename T>
cudaError_t FREE_HOST(T* host_pointer)
{
    return cudaFreeHost(host_pointer);
}

template <typename T>
cudaError_t FREE_DEVICE(T* device_pointer)
{
    return cudaFree(device_pointer);
}

#else

#define H2D(device_pointer, host_pointer, size) CUDA_CHECK(cudaMemcpy((device_pointer), (host_pointer), sizeof(*(host_pointer)) * (size), FLAG_H2D))
#define D2H(host_pointer, device_pointer, size) CUDA_CHECK(cudaMemcpy((host_pointer), (device_pointer), sizeof(*(device_pointer)) * (size), FLAG_D2H))

#define MALLOC_HOST_U32(host_pointer, size) CUDA_CHECK(cudaMallocHost((void**)&host_pointer, sizeof(uint32_t) * (size)))
#define MALLOC_HOST_U64(host_pointer, size) CUDA_CHECK(cudaMallocHost((void**)&host_pointer, sizeof(uint64_t) * (size)))
#define MALLOC_HOST_I32(host_pointer, size) CUDA_CHECK(cudaMallocHost((void**)&host_pointer, sizeof(int32_t) * (size)))
#define MALLOC_HOST_I64(host_pointer, size) CUDA_CHECK(cudaMallocHost((void**)&host_pointer, sizeof(int64_t) * (size)))
#define MALLOC_HOST_FLOAT(host_pointer, size) CUDA_CHECK(cudaMallocHost((void**)&host_pointer, sizeof(float) * (size)))
#define MALLOC_HOST_DOUBLE(host_pointer, size) CUDA_CHECK(cudaMallocHost((void**)&host_pointer, sizeof(double) * (size)))
#define MEMSET_HOST_U32(host_pointer, size) memset((void**)&host_pointer, 0, sizeof(uint32_t) * (size))
#define MEMSET_HOST_U64(host_pointer, size) memset((void**)&host_pointer, 0, sizeof(uint64_t) * (size))

#define MALLOC_DEVICE_U32(device_pointer, size) CUDA_CHECK(cudaMalloc((void**)&device_pointer, sizeof(uint32_t) * (size)))
#define MALLOC_DEVICE_U64(device_pointer, size) CUDA_CHECK(cudaMalloc((void**)&device_pointer, sizeof(uint64_t) * (size)))
#define MALLOC_DEVICE_I32(device_pointer, size) CUDA_CHECK(cudaMalloc((void**)&device_pointer, sizeof(int32_t) * (size)))
#define MALLOC_DEVICE_I64(device_pointer, size) CUDA_CHECK(cudaMalloc((void**)&device_pointer, sizeof(int64_t) * (size)))
#define MALLOC_DEVICE_FLOAT(device_pointer, size) CUDA_CHECK(cudaMalloc((void**)&device_pointer, sizeof(float) * (size)))
#define MALLOC_DEVICE_DOUBLE(device_pointer, size) CUDA_CHECK(cudaMalloc((void**)&device_pointer, sizeof(double) * (size)))
#define MEMSET_DEVICE_U32(device_pointer, size) CUDA_CHECK(cudaMemset((void**)&device_pointer, 0, sizeof(uint32_t) * (size)))
#define MEMSET_DEVICE_U64(device_pointer, size) CUDA_CHECK(cudaMemset((void**)&device_pointer, 0, sizeof(uint64_t) * (size)))

#define FREE_HOST(host_pointer) CUDA_CHECK(cudaFreeHost((void**)&host_pointer))
#define FREE_DEVICE(device_pointer) CUDA_CHECK(cudaFree((void**)&device_pointer))

#endif

#define SYNC_HD                                                                                                                                      \
    CUDA_CHECK(cudaGetLastError());                                                                                                                  \
    CUDA_CHECK(cudaDeviceSynchronize());

#define SYNC_BLOCK __syncthreads();
