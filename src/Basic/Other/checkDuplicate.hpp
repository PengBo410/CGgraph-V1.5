#pragma once

#include <iterator>
#include <pstl/glue_execution_defs.h>
#include <vector>

namespace CPJ {

template <typename Iterator>
bool has_duplicates(Iterator begin, Iterator end, bool USE_STD_PAR = true)
{
    using T = typename std::iterator_traits<Iterator>::value_type;
    std::vector<T> values(begin, end);

    if (USE_STD_PAR) std::sort(std::execution::par_unseq, values.begin(), values.end());
    else std::sort(values.begin(), values.end());

    return (std::adjacent_find(values.begin(), values.end()) != values.end());
}

template <typename T>
bool has_duplicates(const T* array, const size_t arrayLen, bool USE_STD_PAR = true)
{
    std::vector<T> values(array, array + arrayLen);

    if (USE_STD_PAR) std::sort(std::execution::par_unseq, values.begin(), values.end());
    else std::sort(values.begin(), values.end());

    return (std::adjacent_find(values.begin(), values.end()) != values.end());
}
} // namespace CPJ