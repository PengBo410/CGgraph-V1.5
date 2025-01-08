#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Other/traits_CPJ.hpp"
#include <cstddef>
#include <cstdint>

namespace CPJ {

namespace Bits {

constexpr unsigned int bitCount[256] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2,
                                        3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3,
                                        3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5,
                                        6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4,
                                        3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4,
                                        5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6,
                                        6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

template <typename T>
inline constexpr unsigned int popcount_folly(T const v)
{
    using U0 = unsigned int;
    using U1 = unsigned long int;
    using U2 = unsigned long long int;
    static_assert(sizeof(T) <= sizeof(U2), "[popcount_folly] over-sized type");
    static_assert(std::is_integral<T>::value, "[popcount_folly] non-integral type");
    static_assert(!std::is_same<T, bool>::value, "[popcount_folly] can not bool type");

    // clang-format off
        return static_cast<unsigned int>(
            sizeof(T) <= sizeof(U0) ? __builtin_popcount(CPJ::Traits::bits_to_unsigned<U0>(v)) :
            sizeof(T) <= sizeof(U1) ? __builtin_popcountl(CPJ::Traits::bits_to_unsigned<U1>(v)) :
            sizeof(T) <= sizeof(U2) ? __builtin_popcountll(CPJ::Traits::bits_to_unsigned<U2>(v)) :
            0);
    // clang-format on
}

template <typename T>
inline constexpr unsigned int popcount_countBits(T const v)
{
    static_assert(sizeof(T) <= sizeof(uint64_t), "[popCount_countBits] over-sized type");
    static_assert(std::is_integral<T>::value, "[popCount_countBits] non-integral type");
    static_assert(!std::is_same<T, bool>::value, "[popCount_countBits] can not bool type");

    if constexpr (CPJ::Traits::is_uint8_t<T>())
    {
        return static_cast<unsigned int>(bitCount[v]);
    }
    else if constexpr (CPJ::Traits::is_u32_type<T>())
    {
        return bitCount[(v & 0x000000ffu)] + bitCount[(v & 0x0000ff00u) >> 8] + bitCount[(v & 0x00ff0000u) >> 16] + bitCount[(v & 0xff000000u) >> 24];
    }
    else if constexpr (CPJ::Traits::is_u64_type<T>())
    {
        return bitCount[(v & 0x00000000000000FFu)] + bitCount[(v & 0x000000000000FF00u) >> 8] + bitCount[(v & 0x0000000000FF0000u) >> 16] +
               bitCount[(v & 0x00000000FF000000u) >> 24] + bitCount[(v & 0x000000FF00000000u) >> 32] + bitCount[(v & 0x0000FF0000000000u) >> 40] +
               bitCount[(v & 0x00FF000000000000u) >> 48] + bitCount[(v & 0xFF00000000000000u) >> 56];
    }
    else
    {
        assert_msg(false, "[popCount_countBits] get Unknown type");
        return 0;
    }
}

template <typename T>
inline constexpr unsigned int popcount_asm(T const v)
{
    static_assert(sizeof(T) <= sizeof(uint64_t), "[popCount_countBits] over-sized type");
    static_assert(std::is_integral<T>::value, "[popCount_countBits] non-integral type");
    static_assert(!std::is_same<T, bool>::value, "[popCount_countBits] can not bool type");

    if constexpr (CPJ::Traits::is_uint8_t<T>())
    {
        unsigned int count{0};
        uint32_t value = static_cast<uint32_t>(v); // 将 uint8_t 扩展为 uint32_t
        asm("popcnt %1,%0" : "=r"(count) : "rm"(value) : "cc");
        return static_cast<unsigned int>(count);
    }
    else if constexpr (CPJ::Traits::is_u32_type<T>())
    {
        unsigned int count{0};
        asm("popcnt %1,%0" : "=r"(count) : "rm"(static_cast<uint32_t>(v)) : "cc");
        return count;
    }
    else if constexpr (CPJ::Traits::is_u64_type<T>())
    {
        uint64_t count{0};
        // asm("popcnt %1,%0" : "=r"(count) : "rm"(static_cast<uint64_t>(v)) : "cc");
        asm("popcnt %1, %0" : "=r"(count) : "r"(v) : "cc");
        return static_cast<uint32_t>(count);
    }
    else
    {
        assert_msg(false, "[popcount_asm] get Unknown type");
        return 0;
    }
}

template <typename T>
inline constexpr unsigned int popcount_std(T const v)
{
    static_assert(sizeof(T) <= sizeof(uint64_t), "[popCount_countBits] over-sized type");
    static_assert(std::is_integral<T>::value, "[popCount_countBits] non-integral type");
    static_assert(!std::is_same<T, bool>::value, "[popCount_countBits] can not bool type");

    unsigned int count{0};
    if constexpr (CPJ::Traits::is_uint8_t<T>())
    {
        uint32_t value = static_cast<uint32_t>(v); // 将 uint8_t 扩展为 uint32_t
        return static_cast<unsigned int>(std::popcount(value));
    }
    else if constexpr (CPJ::Traits::is_u32_type<T>())
    {
        return static_cast<unsigned int>(std::popcount(v));
    }
    else if constexpr (CPJ::Traits::is_u64_type<T>())
    {
        return static_cast<unsigned int>(std::popcount(v));
    }
    else
    {
        assert_msg(false, "[popcount_std] get Unknown type");
        return 0;
    }
}

/* ************************************************************************
 * Func: 寻找 [v] 对应的二进制中右起第一个1的位置,第一个索引的位置为1，不是0
 *       类似于ffs,
 *       __builtin_ffs系列函数的参数为signed,但在测试中发现unsigned也没有出错
 * example：
 * <int> 2                [10]        return 2
 * <int> 40               [101000]    return 4
 * <int64_t> 0x100000000  [1,32-0]    return 33
 * ************************************************************************/
template <typename T>
inline constexpr unsigned int findFirstSet_folly(T const v)
{
    using S0 = int;
    using S1 = long int;
    using S2 = long long int;
    static_assert(sizeof(T) <= sizeof(S2), "over-sized type");
    static_assert(std::is_integral<T>::value, "non-integral type");
    static_assert(!std::is_same<T, bool>::value, "bool type");

    // clang-format off
    return static_cast<unsigned int>(
        sizeof(T) <= sizeof(S0) ? __builtin_ffs(CPJ::Traits::bits_to_signed<S0>(v)) :
        sizeof(T) <= sizeof(S1) ? __builtin_ffsl(CPJ::Traits::bits_to_signed<S1>(v)) :
        sizeof(T) <= sizeof(S2) ? __builtin_ffsll(CPJ::Traits::bits_to_signed<S2>(v)) :
        0);
    // clang-format on
}

template <typename T>
inline unsigned int findFirstSet_normal(T v)
{
    unsigned int count = 1;
    while (v != 0)
    {
        if (v & 1)
        {
            return count;
        }
        else
        {
            count++;
        }
        v >>= 1;
    }
    return 0;
}

} // namespace Bits

} // namespace CPJ

// void main_test_bitOp()
// {
//     uint8_t a = 5;                       // 二进制: 00000101
//     uint32_t b = 123456789;              // 二进制: 00000111010110111100110100010101
//     uint64_t c = 1234567890123456789ULL; // 二进制较长，省略

//     uint32_t count_a_countBits_U8 = CPJ::Bits::popcount_countBits(a);
//     uint32_t count_a_countBits_U32 = CPJ::Bits::popcount_countBits(b);
//     uint32_t count_a_countBits_U64 = CPJ::Bits::popcount_countBits(c);

//     uint32_t count_a_asm_U8 = CPJ::Bits::popcount_asm(a);
//     uint32_t count_a_asm_U32 = CPJ::Bits::popcount_asm(b);
//     uint32_t count_a_asm_U64 = CPJ::Bits::popcount_asm(c);

//     assert_msg(count_a_countBits_U8 == count_a_asm_U8, "count_a_countBits_U8 = %u, count_a_asm_U8 = %u", count_a_countBits_U8, count_a_asm_U8);
//     assert_msg(count_a_countBits_U32 == count_a_asm_U32, "count_a_countBits_U32 = %u, count_a_asm_U32 = %u", count_a_countBits_U32,
//     count_a_asm_U32); assert_msg(count_a_countBits_U64 == count_a_asm_U64, "count_a_countBits_U64 = %u, count_a_asm_U64 = %u",
//     count_a_countBits_U64, count_a_asm_U64);

//     uint32_t count_a_folly_U8 = CPJ::Bits::popcount_folly(a);
//     uint32_t count_a_folly_U32 = CPJ::Bits::popcount_folly(b);
//     uint32_t count_a_folly_U64 = CPJ::Bits::popcount_folly(c);

//     assert_msg(count_a_countBits_U8 == count_a_folly_U8, "count_a_countBits_U8 = %u, count_a_folly_U8 = %u", count_a_countBits_U8,
//     count_a_folly_U8); assert_msg(count_a_countBits_U32 == count_a_folly_U32, "count_a_countBits_U32 = %u, count_a_folly_U32 = %u",
//     count_a_countBits_U32,
//                count_a_folly_U32);
//     assert_msg(count_a_countBits_U64 == count_a_folly_U64, "count_a_countBits_U64 = %u, count_a_folly_U64 = %u", count_a_countBits_U64,
//                count_a_folly_U64);

//     uint32_t count_a_std_U8 = CPJ::Bits::popcount_std(a);
//     uint32_t count_a_std_U32 = CPJ::Bits::popcount_std(b);
//     uint32_t count_a_std_U64 = CPJ::Bits::popcount_std(c);

//     assert_msg(count_a_countBits_U8 == count_a_std_U8, "count_a_countBits_U8 = %u, count_a_std_U8 = %u", count_a_countBits_U8, count_a_std_U8);
//     assert_msg(count_a_countBits_U32 == count_a_std_U32, "count_a_countBits_U32 = %u, count_a_std_U32 = %u", count_a_countBits_U32,
//     count_a_std_U32); assert_msg(count_a_countBits_U64 == count_a_std_U64, "count_a_countBits_U64 = %u, count_a_std_U64 = %u",
//     count_a_countBits_U64, count_a_std_U64);

//     Msg_check("All finish");

//     CPJ::IOAdaptor ioAdapter("/home/omnisky/.vscode/project/2024-4-2/build/Sort_rawData_30000000_u32.bin");
//     ioAdapter.openFile();
//     std::unique_ptr<count_type[]> data(ioAdapter.readBinFileEntire_sync<count_type>(omp_get_max_threads()));
//     ioAdapter.closeFile();
//     count_type num = 30000000;
//     Msg_finish("Read finish");

//     CPJ::Timer time;

//     /* 预热 */
//     size_t total_heat = 0;
//     {
//         time.start();
//         omp_par_for_reductionAdd(total_heat)(count_type index = 0; index < num; index++) { total_heat +=
//         CPJ::Bits::popcount_countBits(data[index]); } Msg_info("Heat, Used time: %s", time.get_time_str().c_str());
//     }

//     /* 预热 */
//     size_t countBits = 0;
//     {
//         time.start();
//         omp_par_for_reductionAdd(countBits)(count_type index = 0; index < num; index++) { countBits += CPJ::Bits::popcount_countBits(data[index]);
//         } Msg_info("CountBits, Used time: %s", time.get_time_str().c_str());
//     }
//     assert_msg(total_heat == countBits, "total_heat = %zu, countBits = %zu ", total_heat, countBits);

//     /* ASM */
//     size_t countAsm = 0;
//     {
//         time.start();
//         omp_par_for_reductionAdd(countAsm)(count_type index = 0; index < num; index++) { countAsm += CPJ::Bits::popcount_asm(data[index]); }
//         Msg_info("CountAsm, Used time: %s", time.get_time_str().c_str());
//     }
//     assert_msg(total_heat == countAsm, "total_heat = %zu, countAsm = %zu ", total_heat, countAsm);

//     /* Folly */
//     size_t countFolly = 0;
//     {
//         time.start();
//         omp_par_for_reductionAdd(countFolly)(count_type index = 0; index < num; index++) { countFolly += CPJ::Bits::popcount_folly(data[index]); }
//         Msg_info("CountFolly, Used time: %s", time.get_time_str().c_str());
//     }
//     assert_msg(total_heat == countFolly, "total_heat = %zu, countFolly = %zu ", total_heat, countFolly);

//     /* STD */
//     size_t countStd = 0;
//     {
//         time.start();
//         omp_par_for_reductionAdd(countStd)(count_type index = 0; index < num; index++) { countStd += CPJ::Bits::popcount_std(data[index]); }
//         Msg_info("CountStd, Used time: %s", time.get_time_str().c_str());
//     }
//     assert_msg(total_heat == countStd, "total_heat = %zu, countStd = %zu ", total_heat, countStd);

//     /* Now */
//     size_t countNow = 0;
//     {
//         time.start();
//         omp_par_for_reductionAdd(countNow)(count_type index = 0; index < num; index++) { countNow += __builtin_popcount(data[index]); }
//         Msg_info("countNow, Used time: %s", time.get_time_str().c_str());
//     }
//     assert_msg(total_heat == countNow, "total_heat = %zu, countNow = %zu ", total_heat, countNow);
// }