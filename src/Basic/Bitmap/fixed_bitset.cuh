#pragma once

#include "Basic/CUDA/cuda_check.cuh"
#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Thread/omp_def.hpp"
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <numa.h>
#include <omp.h>
#include <stdint.h>

#include <cstdint>

namespace CPJ {

typedef uint64_t bit_type;

#define SMART_LENGTH 1024000

#define BITSIZE 64
#define BIT_OFFSET(i) ((i) >> 6)
#define BIT_MOD(i) ((i)&0x3f) // 0x3f = 63

class Fixed_Bitset
{

  public:
    bit_type* array{nullptr};
    bit_type len;
    bit_type arrlen;

    typedef bit_type array_type;

    Fixed_Bitset() : array(nullptr), len(0), arrlen(0) {}

    Fixed_Bitset(bit_type n) : array(nullptr), len(0), arrlen(0) { setSize(n); }

    Fixed_Bitset(const Fixed_Bitset& db)
    {
        array = NULL;
        len = 0;
        arrlen = 0;
        *this = db;
    }

    ~Fixed_Bitset()
    {
        if (array != nullptr)
        {
            CUDA_CHECK(cudaFreeHost(array));
        }
    }

    void fix_trailing_bits()
    {
        bit_type lastbits = BIT_MOD(len);
        if (lastbits == 0) return;
        array[arrlen - 1] &= ((bit_type(1) << lastbits) - 1);
    }

    void setSize(bit_type n)
    {
        if constexpr (sizeof(bit_type) != 8) assert_msg(false, "<bit_type> Only Support With 64 Bits");

        if (len != 0) assert_msg(false, "Fixed_Bitset Not Allow Set Size More Time");

        len = n;
        arrlen = BIT_OFFSET(n) + (BIT_MOD(n) > 0);
        CUDA_CHECK(cudaMallocHost((void**)&array, arrlen * sizeof(bit_type)));
        fix_trailing_bits();
        parallel_clear();
    }

    inline bit_type size() const { return len; }

    inline bool empty() const
    {
        for (bit_type i = 0; i < arrlen; ++i)
            if (array[i]) return false;
        return true;
    }

    inline bool parallel_empty() const
    {
        volatile bool flag = true;
#pragma omp parallel for shared(flag)
        for (bit_type i = 0; i < arrlen; ++i)
        {
            if (!flag) continue;
            if (array[i] == 0) flag = false;
        }
        return flag;
    }

    inline void clear()
    {
        for (bit_type i = 0; i < arrlen; ++i) array[i] = 0;
    }

    inline void clear_memset_() { memset((void*)array, 0, sizeof(bit_type) * arrlen); }

    inline void parallel_clear()
    {
        omp_parallel_for(bit_type i = 0; i < arrlen; ++i) { array[i] = 0; }
    }

    inline void fill()
    {
        for (bit_type i = 0; i < arrlen; ++i) array[i] = (bit_type)-1;
        fix_trailing_bits();
    }

    inline void parallel_fill()
    {
        omp_parallel_for(bit_type i = 0; i < arrlen; ++i) array[i] = (bit_type)-1;
        fix_trailing_bits();
    }

    /* ***************************************************************************************************************************
     *                                              Normal Function
     * ***************************************************************************************************************************/
    inline bool get(bit_type b) const
    {
        bit_type arrpos, bitpos;
        bit_to_pos(b, arrpos, bitpos);
        return array[arrpos] & (bit_type(1) << bit_type(bitpos));
    }

    inline bool set_bit(bit_type b)
    {
        bit_type arrpos, bitpos;
        bit_to_pos(b, arrpos, bitpos);
        const bit_type mask(bit_type(1) << bit_type(bitpos));
        return __sync_fetch_and_or(array + arrpos, mask) & mask;
    }

    inline bool set_bit_unsync(bit_type b)
    {
        bit_type arrpos, bitpos;
        bit_to_pos(b, arrpos, bitpos);
        const bit_type mask(bit_type(1) << bit_type(bitpos));
        bool ret = array[arrpos] & mask;
        array[arrpos] |= mask;
        return ret;
    }

    inline bool clear_bit(bit_type b)
    {
        bit_type arrpos, bitpos;
        bit_to_pos(b, arrpos, bitpos);
        const bit_type test_mask(bit_type(1) << bit_type(bitpos));
        const bit_type clear_mask(~test_mask);
        return __sync_fetch_and_and(array + arrpos, clear_mask) & test_mask;
    }

    inline bool clear_bit_unsync(bit_type b)
    {
        bit_type arrpos, bitpos;
        bit_to_pos(b, arrpos, bitpos);
        const bit_type test_mask(bit_type(1) << bit_type(bitpos));
        const bit_type clear_mask(~test_mask);
        bool ret = array[arrpos] & test_mask;
        array[arrpos] &= clear_mask;
        return ret;
    }

    inline bool set(bit_type b, bool value)
    {
        if (value) return set_bit(b);
        else return clear_bit(b);
    }

    inline bool set_unsync(bit_type b, bool value)
    {
        if (value) return set_bit_unsync(b);
        else return clear_bit_unsync(b);
    }

    bit_type popcount() const
    {
        bit_type ret = 0;
        for (bit_type i = 0; i < arrlen; ++i)
        {
            ret += __builtin_popcountl(array[i]);
        }
        return ret;
    }

    bit_type parallel_popcount() const
    {
        bit_type ret = 0;
#pragma omp parallel for reduction(+ : ret)
        for (bit_type i = 0; i < arrlen; ++i)
        {
            ret += __builtin_popcountl(array[i]);
        }
        return ret;
    }

    bit_type parallel_popcount(size_t end) const
    {
        assert(end <= len);
        size_t lasWord = (end + 63) / 64;
        bit_type ret = 0;
#pragma omp parallel for reduction(+ : ret)
        for (bit_type i = 0; i < (lasWord - 1); ++i)
        {
            ret += __builtin_popcountl(array[i]);
        }
        size_t word = array[lasWord - 1];
        bit_type lastbits = BIT_MOD(end);
        if (lastbits != 0)
        {
            word &= ((bit_type(1) << lastbits) - 1);
            ret += __builtin_popcountl(word);
        }
        return ret;
    }

    inline size_t containing_word(size_t b)
    {
        size_t arrpos, bitpos;
        bit_to_pos(b, arrpos, bitpos);
        return array[arrpos];
    }

    inline Fixed_Bitset& operator=(const Fixed_Bitset& db)
    {
        len = db.len;
        arrlen = db.arrlen;
        CUDA_CHECK(cudaMallocHost((void**)&array, arrlen * sizeof(bit_type)));
        memcpy(array, db.array, sizeof(bit_type) * arrlen);
        return *this;
    }

    inline Fixed_Bitset& operator()(const Fixed_Bitset& db)
    {
        len = db.len;
        arrlen = db.arrlen;
        array = db.array;
        return *this;
    }

    void resize(bit_type n) { setSize(n); }

    void clear_memset() { parallel_clear(); }

    void clear_smart()
    {
        if (len <= SMART_LENGTH)
        {
            clear();
        }
        else
        {
            parallel_clear();
        }
    }

    void fill_smart()
    {
        if (len <= SMART_LENGTH)
        {
            fill();
        }
        else
        {
            parallel_fill();
        }
    }

  private:
    inline static void bit_to_pos(bit_type b, bit_type& arrpos, bit_type& bitpos)
    {
        arrpos = BIT_OFFSET(b);
        bitpos = BIT_MOD(b);
    }

}; // end of class [Fixed_Bitset]

} // namespace CPJ