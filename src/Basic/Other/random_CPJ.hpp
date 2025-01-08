#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Other/traits_CPJ.hpp"
#include <cstddef>
#include <random>
#include <vector>
namespace CPJ {

class Random
{
  public:
    /* *********************************************************************************************************
     * @description: 随机生成randomNum个[left, right)范围内的随机数, 但是这些数可能有重复的
     * @param [T] left 范围的左界限
     * @param [T] right 范围的右界限
     * @param [size_t] randomNum 要生成的随机数的个数
     * @return [std::vector<T>] 生成的随机数会保存到vec中返回
     * *********************************************************************************************************/
    template <typename T, typename SIZE_T>
    static std::vector<T> generateRandomNumbers_uniform(T left, T right, SIZE_T randomNum)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::vector<T> random_numbers_vec;

        if constexpr (CPJ::Traits::is_integral_point<T>())
        {
            if constexpr (CPJ::Traits::is_signed<T>())
            {
                assert_msg(left < right, "Invalid [left(%ld), right(%ld))", SCI64(left), SCI64(right));
            }
            else
            {
                assert_msg(left < right, "Invalid [left(%zu), right(%zu))", SCU64(left), SCU64(right));
            }

            std::uniform_int_distribution<T> dis(left, right - 1);
            random_numbers_vec.resize(randomNum);
            for (SIZE_T i = 0; i < randomNum; ++i) random_numbers_vec[i] = dis(gen);
        }
        else if constexpr (CPJ::Traits::is_float_point<T>())
        {
            assert_msg(left < right, "Invalid [left(%f), right(%f))", left, right);
            std::uniform_real_distribution<T> dis(left, right);
            random_numbers_vec.resize(randomNum);
            for (SIZE_T i = 0; i < randomNum; ++i) random_numbers_vec[i] = dis(gen);
        }
        else
        {
            assert_msg(false, "Func[generateRandomNumver()] only support <integral_type> or <float_type>");
        }

        return random_numbers_vec;
    }

    /* *********************************************************************************************************
     * @description: 随机生成randomNum个[left, right)范围内的不重复的随机数, 但是只支持生成整型
     * @param [T] left 范围的左界限
     * @param [T] right 范围的右界限
     * @param [size_t] randomNum 要生成的随机数的个数
     * @return [std::vector<T>] 生成的随机数会保存到vec中返回
     * *********************************************************************************************************/
    template <typename T, typename SIZE_T>
    static std::vector<T> generateRandomNumbers_NoRepeat(T left, T right, SIZE_T randomNum)
    {
        assert_msg(CPJ::Traits::is_integral_point<T>(), "generateRandomNumbers_NoRepeat current only support generate integral");
        assert_msg_smart((right - left) >= randomNum, "Too small range[{}, {}) can not geberate {} no repeat random number", left, right, randomNum);

        // 生成0到vertexNum_-1的所有可能值
        std::vector<T> all_numbers(randomNum);
        std::iota(all_numbers.begin(), all_numbers.end(), 0);

        // 随机洗牌
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(all_numbers.begin(), all_numbers.end(), g);

        return all_numbers;
    }
};
} // namespace CPJ
