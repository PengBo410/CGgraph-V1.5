#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include "Basic/IO/io_adapter_V1.hpp"
#include "Basic/Other/random_CPJ.hpp"
#include "Basic/Type/data_type.hpp"
#include <algorithm>
#include <cstddef>
#include <execution>
#include <pstl/glue_execution_defs.h>
#include <thrust/sort.h>

namespace CPJ {
namespace SORT {

constexpr size_t SORT_LIMITED_TASKS = 1024;

class MergeSort
{
  public:
    template <typename SIZE_T, typename ARRAY_T>
    static void mergeSort(ARRAY_T* X, SIZE_T n, ARRAY_T* tmp)
    {
        if (n < 2) return;

#pragma omp task shared(X) if (n > SORT_LIMITED_TASKS)
        mergeSort(X, n / 2, tmp);

#pragma omp task shared(X) if (n > SORT_LIMITED_TASKS)
        mergeSort(X + (n / 2), n - (n / 2), tmp + n / 2);

#pragma omp taskwait
        mergeSortAux(X, n, tmp);
    }

  private:
    template <typename SIZE_T, typename ARRAY_T>
    static void mergeSortAux(ARRAY_T* X, SIZE_T n, ARRAY_T* tmp)
    {
        SIZE_T i = 0;
        SIZE_T j = n / 2;
        SIZE_T ti = 0;

        while (i < n / 2 && j < n)
        {
            if (X[i] < X[j])
            {
                tmp[ti] = X[i];
                ti++;
                i++;
            }
            else
            {
                tmp[ti] = X[j];
                ti++;
                j++;
            }
        }
        while (i < n / 2)
        { /* finish up lower half */
            tmp[ti] = X[i];
            ti++;
            i++;
        }
        while (j < n)
        { /* finish up upper half */
            tmp[ti] = X[j];
            ti++;
            j++;
        }
        // Msg_check("N = %zd", n);
        memcpy(X, tmp, n * sizeof(ARRAY_T));
    }
};

// 较快的排序
class MergeSortRecursive
{
  public:
    /* *********************************************************************************************************
     * @description: 快速的并行排序
     * @param [std::vector<ARRAY_T>&*] vec 要排序的vector
     * @return [*]
     * *********************************************************************************************************/
    /**********************************************************************************************************
     * @description:
     * @return [*]
     **********************************************************************************************************/
    template <typename ARRAY_T>
    static void mergeSortRecursiveVec(std::vector<ARRAY_T>& vec)
    {
#pragma omp parallel
#pragma omp single
        mergeSortRecursive_vec(vec, static_cast<decltype(vec.size())>(0), vec.size() - 1);
    }

    /* *********************************************************************************************************
     * @description: 快速的并行排序
     *               https://cw.fel.cvut.cz/old/_media/courses/b4m35pag/lab6_slides_advanced_openmp.pdf
     * @param [ARRAY_T*] array 要排序的数组
     * @param [SIZE_T] arrayLength 数据的长度
     * @return [*]
     * *********************************************************************************************************/
    template <typename SIZE_T, typename ARRAY_T>
    static void mergeSortRecursive_array(ARRAY_T* array, const SIZE_T arrayLength)
    {
#pragma omp parallel
#pragma omp single
        mergeSortRecursive_array(array, static_cast<SIZE_T>(0), static_cast<SIZE_T>(arrayLength - 1));
    }

  private:
    template <typename SIZE_T, typename ARRAY_T>
    static void mergeSortRecursive_vec(std::vector<ARRAY_T>& v, SIZE_T left, SIZE_T right)
    {
        if (left < right)
        {
            if (right - left >= 32)
            {
                SIZE_T mid = (left + right) / 2;
#pragma omp taskgroup
                {
#pragma omp task shared(v) untied if (right - left >= (1 << 14))
                    mergeSortRecursive_vec(v, left, mid);
#pragma omp task shared(v) untied if (right - left >= (1 << 14))
                    mergeSortRecursive_vec(v, mid + 1, right);
#pragma omp taskyield
                }
                std::inplace_merge(v.begin() + left, v.begin() + mid + 1, v.begin() + right + 1);
            }
            else
            {
                std::sort(v.begin() + left, v.begin() + right + 1);
            }
        }
    }

    template <typename SIZE_T, typename ARRAY_T>
    static void mergeSortRecursive_array(ARRAY_T* array, SIZE_T left, SIZE_T right)
    {
        if (left < right)
        {
            if (right - left >= 32)
            {
                SIZE_T mid = (left + right) / 2;
#pragma omp taskgroup
                {
#pragma omp task shared(array) untied if (right - left >= (1 << 14))
                    mergeSortRecursive_array(array, left, mid);
#pragma omp task shared(array) untied if (right - left >= (1 << 14))
                    mergeSortRecursive_array(array, mid + 1, right);
#pragma omp taskyield
                }
                std::inplace_merge(array + left, array + mid + 1, array + right + 1);
            }
            else
            {
                std::sort(array + left, array + right + 1);
            }
        }
    }
};
} // namespace SORT
} // namespace CPJ

template <typename SIZE_T, typename ARRAY_T>
void validate_sort(ARRAY_T* data, SIZE_T n)
{
#pragma omp parallel for
    for (SIZE_T i = 0; i < n - 1; i++)
    {
        assert_msg(data[i] < data[i + 1], "Sort error, data[%zu] = %zu, data[%zu] = %zu", SCU64(i), SCU64(data[i]), SCU64(i + 1), SCU64(data[i + 1]));
    }
    assert_msg(data[n - 1] == (n - 1), "data[n-1] = %zu, (n-1) = %zu", SCU64(data[n - 1]), SCU64(n - 1));
    Msg_finish("Validate Sort Finsih");
}

/* *
 * [INFOS-0]: OMP Max Threads [24] -> [OpenMPMergeSort.hpp:185 行]
 * [INFOS-0]: - std::sort seq Used time: 2.90 (s) -> [OpenMPMergeSort.hpp:218 行]
 * [FINSH-0]: Validate Sort Finsih -> [OpenMPMergeSort.hpp:175 行]
 * [INFOS-0]: - std::sort parallel Used time: 200.09 (ms) -> [OpenMPMergeSort.hpp:227 行]
 * [FINSH-0]: Validate Sort Finsih -> [OpenMPMergeSort.hpp:175 行]
 * [INFOS-0]: - MergeSort::mergeSort Used time: 1.04 (s) -> [OpenMPMergeSort.hpp:246 行]
 * [FINSH-0]: Validate Sort Finsih -> [OpenMPMergeSort.hpp:175 行]
 * [INFOS-0]: - CPJ::SORT::MergeSortV2::mergeSortRecursive_array Used time: 643.22 (ms) -> [OpenMPMergeSort.hpp:262 行]
 * [FINSH-0]: Validate Sort Finsih -> [OpenMPMergeSort.hpp:175 行]
 * [INFOS-0]: Hello, from 2023-10-8-CPJ! -> [CGgraphV1.5.cu:23 行]
 * => std::sort的并行排序是最快的
 * */
void test_sort_MergeSort()
{
    typedef vertex_id_type array_type;
    typedef count_type size_type;
    typedef count_type qsort_type;

    // omp_set_nested(1);
    Msg_info("OMP Max Threads [%u]", omp_get_max_threads());

    size_type sortNum = 30000000;
    std::string filePath = "Sort_rawData_" + std::to_string(sortNum) + "_" + std::string((sizeof(array_type) == 4) ? "u32.bin" : "u64.bin");
    array_type* sort_data{nullptr};
    std::vector<array_type> randomArray;
    bool is_exist = CPJ::FS::isExist(filePath);
    if (!is_exist)
    {
        randomArray = CPJ::Random::generateRandomNumbers_NoRepeat(static_cast<size_type>(0), sortNum, sortNum);
        sort_data = randomArray.data();
        CPJ::IOAdaptor ioAdapter(filePath);
        ioAdapter.openFile("w");
        ioAdapter.writeBinFile_sync(randomArray.data(), randomArray.size() * sizeof(array_type), omp_get_max_threads());
        ioAdapter.closeFile();
    }
    else
    {
        CPJ::IOAdaptor ioAdapter(filePath);
        ioAdapter.openFile();
        sort_data = ioAdapter.readBinFileEntire_sync<array_type>(omp_get_max_threads());
        ioAdapter.closeFile();
    }

    // int testNum = 3;
    CPJ::Timer time;

    // 预热
    {
        std::unique_ptr<array_type[]> resultArray(new array_type[sortNum]);
        std::memcpy(resultArray.get(), sort_data, sortNum * sizeof(array_type));
        time.start();
        std::sort(resultArray.get(), resultArray.get() + sortNum);
        Msg_info("- std::sort seq Used time: %s", time.get_time_str().c_str());
        validate_sort(resultArray.get(), sortNum);
    }

    {
        std::unique_ptr<array_type[]> resultArray(new array_type[sortNum]);
        std::memcpy(resultArray.get(), sort_data, sortNum * sizeof(array_type));
        time.start();
        std::sort(std::execution::par_unseq, resultArray.get(), resultArray.get() + sortNum);
        Msg_info("- std::sort parallel Used time: %s", time.get_time_str().c_str());
        validate_sort(resultArray.get(), sortNum);
    }

    {
        // omp_set_dynamic(0);                         /** Explicitly disable dynamic teams **/
        // omp_set_num_threads(omp_get_max_threads()); /** Use N threads for all parallel regions **/
        // omp_set_nested(1);
        array_type* tmp = new array_type[sortNum];

        std::unique_ptr<array_type[]> resultArray(new array_type[sortNum]);
        std::memcpy(resultArray.get(), sort_data, sortNum * sizeof(array_type));

        time.start();
#pragma omp parallel
        {
#pragma omp single
            CPJ::SORT::MergeSort::mergeSort(resultArray.get(), static_cast<qsort_type>(sortNum), tmp);
        }
        Msg_info("- MergeSort::mergeSort Used time: %s", time.get_time_str().c_str());
        validate_sort(tmp, sortNum);
    }

    {
        // omp_set_dynamic(0);                         /** Explicitly disable dynamic teams **/
        omp_set_num_threads(omp_get_max_threads()); /** Use N threads for all parallel regions **/
        // omp_set_nested(1);

        std::unique_ptr<array_type[]> resultArray(new array_type[sortNum]);
        std::memcpy(resultArray.get(), sort_data, sortNum * sizeof(array_type));
        // std::vector<array_type> vec(resultArray.get(), resultArray.get() + sortNum);

        time.start();
        CPJ::SORT::MergeSortRecursive::mergeSortRecursive_array(resultArray.get(), sortNum);

        Msg_info("- CPJ::SORT::MergeSortV2::mergeSortRecursive_array Used time: %s", time.get_time_str().c_str());
        validate_sort(resultArray.get(), sortNum);
    }
}