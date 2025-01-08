#pragma once

#define CUDA_INCLUDE_TEMP  //这里我们自定义一个全路径，避免VSCode的红色波浪线
                           //但是这个需要根据自己的服务器路径配置，直接取消定义也不会错误

#ifdef CUDA_INCLUDE_TEMP

    #include </usr/local/cuda-11.7/include/cuda_runtime.h>
    #include </usr/local/cuda-11.7/include/cuda.h>
    #include </usr/local/cuda-11.7/include/device_launch_parameters.h>
#else
    #include <cuda_runtime.h>
    #include <cuda.h>
    #include <device_launch_parameters.h>
#endif