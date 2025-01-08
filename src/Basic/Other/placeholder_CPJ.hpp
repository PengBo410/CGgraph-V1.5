#pragma once

#include "Basic/Other/traits_CPJ.hpp"
#include <cassert>
#include <cstdlib>
#include <iomanip>

namespace CPJ {

template <typename T>
static inline constexpr const char* TYPE_HOLDER()
{
    if constexpr (CPJ::Traits::is_float_type<T>())
    {
        return "%.2f";
    }
    else if constexpr (CPJ::Traits::is_double_type<T>())
    {
        return "%.2lf";
    }
    else if constexpr (CPJ::Traits::is_int8_t<T>())
    {
        return "%hhd";
    }
    else if constexpr (CPJ::Traits::is_uint8_t<T>())
    {
        return "%hhu";
    }
    else if constexpr (CPJ::Traits::is_int_type<T>())
    {
        return "%d";
    }
    else if constexpr (CPJ::Traits::is_u32_type<T>())
    {
        return "%u";
    }
    else if constexpr (CPJ::Traits::is_i64_type<T>())
    {
        return "%ld";
    }
    else if constexpr (CPJ::Traits::is_u64_type<T>())
    {
        return "%zu";
    }
    else if constexpr (CPJ::Traits::is_ssize_type<T>())
    {
        return "%zd";
    }
    else if constexpr (CPJ::Traits::is_char_type<T>())
    {
        return "'%c'";
    }
    else if constexpr (CPJ::Traits::is_chars_type<T>())
    {
        return "\"%s\"";
        // return std::quoted("%s")._M_string;
    }
    else if constexpr (CPJ::Traits::is_string_type<T>())
    {
        return "\"%s\"";
    }
    else
    {
        printf("[Error]: Can not set placeholder automatically");
        assert(false);
        std::exit(1);
        return "%s";
    }
}

} // namespace CPJ