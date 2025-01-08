#pragma once

/* *********************************************************************************************************
 * @description: 抽时间确定怎么实现 (我们想获取一个函数返回值的类型, 但是重载函数会有一定的问题)
 *               参考资料:
 *               // https://stackoverflow.com/questions/40135561/decltype-for-overloaded-member-function
 *               // https://stackoverflow.com/questions/22291737/why-cant-decltype-work-with-overloaded-functions
 * @return [*]
 * *********************************************************************************************************/

#include <cstdint>
#include <limits>
#include <string>
#include <sys/types.h>
#include <type_traits>

namespace CPJ {

class Traits
{
  public:
    template <typename T>
    static constexpr bool is_float_point()
    {
        if constexpr (std::is_floating_point_v<std::remove_cv_t<std::remove_reference_t<T>>>) return true;
        return false;
    }

    template <typename T>
    static constexpr bool is_float_type()
    {
        if constexpr (std::is_floating_point_v<std::remove_cv_t<std::remove_reference_t<T>>> &&
                      std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, float>)
            return true;
        return false;
    }

    template <typename T>
    static constexpr bool is_double_type()
    {
        if constexpr (std::is_floating_point_v<std::remove_cv_t<std::remove_reference_t<T>>> &&
                      std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, double>)
            return true;
        return false;
    }

    template <typename T>
    static constexpr bool is_pointer()
    {
        return std::is_pointer_v<T>;
        // return std::is_pointer_v<std::remove_cv_t<std::remove_reference_t<T>>>;
    }

    template <typename T>
    static constexpr bool is_integral_point()
    {
        if constexpr (std::is_integral_v<std::remove_cv_t<std::remove_reference_t<T>>>) return true;
        return false;
    }

    template <typename T>
    static constexpr bool is_uint8_t()
    {
        return std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, uint8_t>;
    }

    template <typename T>
    static constexpr bool is_int8_t()
    {
        return std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, int8_t>;
    }

    template <typename T>
    static constexpr bool is_int_type()
    {
        if constexpr (is_integral_point<T>() && std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, int>) return true;
        return false;
    }

    template <typename T>
    static constexpr bool is_u32_type()
    {
        if constexpr (is_integral_point<T>() && std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, uint32_t>) return true;
        return false;
    }

    template <typename T>
    static constexpr bool is_i64_type()
    {
        if constexpr (is_integral_point<T>() && std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, int64_t>) return true;
        return false;
    }

    template <typename T>
    static constexpr bool is_u64_type()
    {
        if constexpr (is_integral_point<T>() && std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, uint64_t>) return true;
        return false;
    }

    template <typename T>
    static constexpr bool is_ssize_type()
    {
        if constexpr (is_integral_point<T>() && std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, ssize_t>) return true;
        return false;
    }

    template <typename T>
    static constexpr bool is_char_type()
    {
        if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, char>) return true;
        return false;
    }

    template <typename T>
    static constexpr bool is_chars_type()
    {
        if constexpr (is_pointer<T>())
        {
            using PlainType = std::remove_pointer_t<T>;
            if constexpr (is_char_type<PlainType>())
            {
                return true;
            }
        }
        return false;
    }

    template <typename T>
    static constexpr bool is_string_type()
    {
        if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, std::string>) return true;
        return false;
    }

    /* *********************************************************************************************************
     * @description: 浮点类型既可以表示正数也可以表示负数, 但是不应该参与判断
     * @return [*]
     * *********************************************************************************************************/
    template <typename T>
    static constexpr bool is_signed()
    {
        if constexpr (std::numeric_limits<std::remove_cv_t<std::remove_reference_t<T>>>::is_signed) return true;
        return false;
    }

    template <typename Dst, typename Src>
    static constexpr std::make_signed_t<Dst> bits_to_signed(Src const s)
    {
        static_assert(std::is_signed<Dst>::value, "unsigned type");
        return to_signed(static_cast<std::make_unsigned_t<Dst>>(to_unsigned(s)));
    }
    template <typename Dst, typename Src>
    static constexpr std::make_unsigned_t<Dst> bits_to_unsigned(Src const s)
    {
        static_assert(std::is_unsigned<Dst>::value, "signed type");
        return static_cast<Dst>(to_unsigned(s));
    }

    struct to_signed_fn
    {
        template <typename..., typename T>
        constexpr auto operator()(T const& t) const noexcept -> typename std::make_signed<T>::type
        {
            using S = typename std::make_signed<T>::type;
            constexpr auto m = static_cast<T>(std::numeric_limits<S>::max());
            return m < t ? -static_cast<S>(~t) + S{-1} : static_cast<S>(t);
        }
    };
    inline static constexpr to_signed_fn to_signed{};

    struct to_unsigned_fn
    {
        template <typename..., typename T>
        constexpr auto operator()(T const& t) const noexcept -> typename std::make_unsigned<T>::type
        {
            using U = typename std::make_unsigned<T>::type;
            return static_cast<U>(t);
        }
    };
    inline static constexpr to_unsigned_fn to_unsigned{};

    // namespace detail {

    // } // namespace detail
};
} // namespace CPJ