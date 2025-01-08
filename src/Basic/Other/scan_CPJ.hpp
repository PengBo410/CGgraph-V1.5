#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include <execution>
#include <numeric>
#include <pstl/glue_execution_defs.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <type_traits>

namespace CPJ {
namespace SCAN {

namespace HOST {

template <typename T, typename SIZE_T>
inline T exclusive_scan_omp(T* array_first, const SIZE_T array_length, T* result)
{
    T scan_a = 0;
#pragma omp simd reduction(inscan, + : scan_a)
    for (int i = 0; i < array_length; i++)
    {
        result[i] = scan_a;
#pragma omp scan exclusive(scan_a)
        scan_a += array_first[i];
    }

    return scan_a;
}

template <typename T, typename SIZE_T>
inline T exclusive_scan_thrust(T* array_first, const SIZE_T array_length, T* result)
{
    T totalDegree_temp = array_first[array_length - 1];
    thrust::exclusive_scan(thrust::host, array_first, array_first + array_length, result, static_cast<T>(0));
    totalDegree_temp = totalDegree_temp + result[array_length - 1];

    return totalDegree_temp;
}

template <typename T, typename RT, typename SIZE_T>
inline RT exclusive_scan_std(T* array_first, const SIZE_T array_length, RT* result)
{
    RT totalDegree_temp = array_first[array_length - 1];
    std::exclusive_scan(/*std::execution::par,*/ array_first, array_first + array_length, result, static_cast<RT>(0));
    totalDegree_temp = totalDegree_temp + result[array_length - 1];

    return totalDegree_temp;
}

template <typename T, typename RT, typename SIZE_T>
inline RT exclusive_scan_std_par(T* array_first, const SIZE_T array_length, RT* result)
{
    bool isSame = std::is_same_v<T, RT>;
    assert_msg(isSame, "array_first and result need same type");
    assert_msg(reinterpret_cast<void*>(array_first) != reinterpret_cast<void*>(result),
               "Par version [array_first] and [result] cannot point to the same array in Func [%s]", __FUNCTION__);
    RT totalDegree_temp = array_first[array_length - 1];
    std::exclusive_scan(std::execution::par_unseq, array_first, array_first + array_length, result, static_cast<RT>(0));
    totalDegree_temp = totalDegree_temp + result[array_length - 1];

    return totalDegree_temp;
}

} // namespace HOST

namespace Device {

template <typename T, typename SIZE_T>
inline T exclusive_scan_thrust(thrust::device_ptr<T> array_first, const SIZE_T array_length, thrust::device_ptr<T> result)
{
    T totalDegree_temp = array_first[array_length - 1];
    thrust::exclusive_scan(thrust::device, array_first, array_first + array_length, result, static_cast<T>(0));
    totalDegree_temp = totalDegree_temp + result[array_length - 1];

    return totalDegree_temp;
}

template <typename T, typename SIZE_T>
inline T exclusive_scan_thrust(T* array_first, const SIZE_T array_length, T* result)
{

    thrust::device_ptr<T> array_first_thrust = thrust::device_pointer_cast(array_first);
    thrust::device_ptr<T> result_thrust = thrust::device_pointer_cast(result);
    T totalDegree_temp = array_first_thrust[array_length - 1];
    thrust::exclusive_scan(thrust::device, array_first_thrust, array_first_thrust + array_length, result_thrust, static_cast<T>(0));
    totalDegree_temp = totalDegree_temp + result_thrust[array_length - 1];

    return totalDegree_temp;
}

template <typename T, typename SIZE_T, typename R>
inline R exclusive_scan_thrust_mut(T* array_first, const SIZE_T array_length, R* result)
{
    thrust::device_ptr<T> array_first_thrust = thrust::device_pointer_cast(array_first);
    thrust::device_ptr<R> result_thrust = thrust::device_pointer_cast(result);
    R totalDegree_temp = array_first_thrust[array_length - 1];
    thrust::exclusive_scan(thrust::device, array_first_thrust, array_first_thrust + array_length, result_thrust, static_cast<R>(0));
    totalDegree_temp = totalDegree_temp + result_thrust[array_length - 1];

    return totalDegree_temp;
}

} // namespace Device

} // namespace SCAN
} // namespace CPJ