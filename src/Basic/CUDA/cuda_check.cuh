#ifndef CUDA_CHECK_CUH
#define CUDA_CHECK_CUH

#include "cuda_include.cuh"
#include <cstdio>
#include <libgen.h>
#include <vector>

#define ESC_START "\033["
#define ESC_END "\033[0m"
#define COLOR_ERROR "31;48;1m" // 31:是红色

// #include <vector_types.h>

static void HandleError(char* file, int line, cudaError_t err)
{
    printf(ESC_START COLOR_ERROR "<CUDA-ERROR>: (%d) %s -> {%s : %u 行}" ESC_END "\n", err, cudaGetErrorString(err), basename((char*)file), line);
    exit(1);
}

// static void HandleError(char* file, int line, CUresult err)
// {
// 	const char* err_str;
// 	cuGetErrorString(err, &err_str);

// 	printf("ERROR in %s:%d: %s (%d)\n", file, line,
// 		err_str == nullptr ? "UNKNOWN ERROR VALUE" : err_str, err);
// 	exit(1);
// }

// CUDA assertions
#define GY_CUDA_CHECK(err)                           \
    do {                                             \
        cudaError_t errr = (err);                    \
        if (errr != cudaSuccess) {                   \
            ::HandleError(__FILE__, __LINE__, errr); \
        }                                            \
    } while (0)

// #define CUDA_CHECK(err) do {                                  \
//     cudaError_t errr = (err);                                        \
//     if(errr != cudaSuccess)                                          \
//     {                                                                \
//         ::HandleError(__FILE__, __LINE__, errr);             \
//     }                                                                \
// } while(0)

#define CUDA_CHECK(err)                                                                                                                                 \
    do {                                                                                                                                                \
        cudaError_t errr = (err);                                                                                                                       \
        if (errr != cudaSuccess) {                                                                                                                      \
            printf(ESC_START COLOR_ERROR "<CUDA-ERROR>: (%d) %s -> {%s : %u 行}" ESC_END "\n", err, cudaGetErrorString(err), basename((char*)__FILE__), \
                __LINE__);                                                                                                                              \
            exit(1);                                                                                                                                    \
        }                                                                                                                                               \
    } while (0)

// #define CUDA_CHECK_test \
// do{\
// 	printf(ESC_START COLOR_ERROR "<CUDA-ERROR>: -> {%s : %u 行}" ESC_END "\n", \
//  			 basename((char *)__FILE__), __LINE__); \
// 		exit(1);\
// }while(0)

// inline void CUDA_CHECK(cudaError_t err) {
//     if (err != cudaSuccess) {
// 		printf(ESC_START COLOR_ERROR "<CUDA-ERROR>: (%d) %s -> {%s : %u 行}" ESC_END "\n",
// 			err, cudaGetErrorString(err), basename((char *)__FILE__), __LINE__);
// 		exit(1);
//     }
// }

#define GY_CUDA_DAPI_CHECK(err)                          \
    do {                                                 \
        CUresult errr = (err);                           \
        if (errr != ::CUDA_SUCCESS) {                    \
            ::GY::HandleError(__FILE__, __LINE__, errr); \
        }                                                \
    } while (0)

#define __deviceSync__ GY_CUDA_CHECK(cudaDeviceSynchronize())
#define __deviceSync                            \
    {                                           \
        GY_CUDA_CHECK(cudaGetLastError());      \
        GY_CUDA_CHECK(cudaDeviceSynchronize()); \
    }

/*
        device<<<10,256>>>();
        GY_CUDA_CHECK(cudaGetLastError());
        GY_CUDA_CHECK(cudaDeviceSynchronize());

        printf("blockSize = %d\n", roundUp);// 2560
        printf("gridDim.x = %d\n", gridDim.x);//nBlock = 10
        printf("blockDim.x = %d\n", blockDim.x);//blockSize = 256
*/

#endif // CUDA_CHECK_CUH
