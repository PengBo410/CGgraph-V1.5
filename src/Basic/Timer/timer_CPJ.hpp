#pragma once

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <thread>

namespace CPJ {

class Timer
{
  public:
    Timer()
    {
        start();
        reset_period_point();
    }

    inline void start() { start_time_ = std::chrono::high_resolution_clock::now(); }
    inline void stop() { end_time_ = std::chrono::high_resolution_clock::now(); }
    inline double get_time_ns()
    {
        stop();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time_ - start_time_);
        return static_cast<double>(elapsed_time.count());
    }
    inline double get_time_us()
    {
        stop();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_);
        return static_cast<double>(elapsed_time.count());
    }
    inline double get_time_ms() { return static_cast<double>(get_time_us() / 1000); }
    inline double get_time_s() { return static_cast<double>(get_time_ms() / 1000); }
    inline std::string get_time_str()
    {
        double microseconds = get_time_us();
        std::ostringstream oss;

        if (microseconds >= MIN_THRESHOLD * S_THRESHOLD * MS_THRESHOLD * US_THRESHOLD)
        {
            // 超过60分钟，以小时为单位输出
            double hours = static_cast<double>(microseconds) / (MIN_THRESHOLD * S_THRESHOLD * MS_THRESHOLD * US_THRESHOLD);
            oss << std::fixed << std::setprecision(FIX_THRESHOLD) << hours << " (h)";
        }
        else if (microseconds >= S_THRESHOLD * MS_THRESHOLD * US_THRESHOLD)
        {
            // 超过1分钟，以分钟为单位输出
            double minutes = static_cast<double>(microseconds) / (S_THRESHOLD * MS_THRESHOLD * US_THRESHOLD);
            oss << std::fixed << std::setprecision(FIX_THRESHOLD) << minutes << " (min)";
        }
        else if (microseconds >= MS_THRESHOLD * US_THRESHOLD)
        {
            // 超过1000毫秒，以秒为单位输出
            double seconds = static_cast<double>(microseconds) / (MS_THRESHOLD * US_THRESHOLD);
            oss << std::fixed << std::setprecision(FIX_THRESHOLD) << seconds << " (s)";
        }
        else if (microseconds >= US_THRESHOLD)
        {
            // 超过1000微秒，以毫秒为单位输出
            double milliseconds = static_cast<double>(microseconds) / US_THRESHOLD;
            oss << std::fixed << std::setprecision(FIX_THRESHOLD) << milliseconds << " (ms)";
        }
        else
        {
            // 小于1000微秒，直接输出微秒
            oss << std::fixed << std::setprecision(FIX_THRESHOLD) << microseconds << " (us)";
        }

        return oss.str();
    }

    inline void sleep_for_s(uint32_t seconds) { std::this_thread::sleep_for(std::chrono::seconds(seconds)); }
    inline void sleep_for_ms(uint32_t milliseconds) { std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds)); }
    inline void sleep_for_us(uint32_t microseconds) { std::this_thread::sleep_for(std::chrono::microseconds(microseconds)); }
    inline void sleep_for_ns(uint32_t nanoseconds) { std::this_thread::sleep_for(std::chrono::nanoseconds(nanoseconds)); }

    inline std::string curFormatTime()
    {
        auto current_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        struct tm* timeinfo = std::localtime(&current_time); // 将时间点转换为本地时间

        std::ostringstream curTime;
        curTime << std::put_time(timeinfo, "%Y-%m-%d %H:%M:%S");
        return curTime.str();
    }

    inline void reset_period_point() { period_point_ = std::chrono::steady_clock::now(); } //! 这个函数应该在调用处被显示的调用
    inline bool period_ms(uint32_t milliseconds)
    {
        auto cur = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(cur - period_point_);
        if (elapsed_time >= std::chrono::milliseconds(milliseconds))
        {
            reset_period_point();
            return true;
        }
        return false;
    }
    inline bool period_us(uint32_t microseconds)
    {
        auto cur = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(cur - period_point_);
        if (elapsed_time.count() >= microseconds)
        {
            // reset_period_point(); //!加在此处处理结果就不对, 不是很理解
            return true;
        }
        return false;
    }

  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time_;
    std::chrono::time_point<std::chrono::steady_clock>
        period_point_; // someone say [std::chrono::steady_clock] can more precise control of time intervals than [std::chrono::high_resolution_clock]

    static constexpr double US_THRESHOLD = 1000;
    static constexpr double MS_THRESHOLD = 1000;
    static constexpr double S_THRESHOLD = 60;
    static constexpr double MIN_THRESHOLD = 60;
    static constexpr int FIX_THRESHOLD = 2;
};

/* *********************************************************************************************************
 * @description: 获取当前的时间
 * @return [*]
 * *********************************************************************************************************/
inline std::string getCurDate()
{
    // 获取当前时间
    std::time_t rawtime;
    std::tm timeinfo;
    char buffer[80];

    // 获取当前时间点
    std::time(&rawtime);

// 将时间转换为本地时间，并确保线程安全
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&timeinfo, &rawtime); // Windows 平台上的线程安全版本
#else
    localtime_r(&rawtime, &timeinfo); // POSIX 平台上的线程安全版本
#endif

    // 格式化时间
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d-%H-%M-%S", &timeinfo);
    return std::string(buffer);
}

} // namespace CPJ

/*

    CPJ::Timer time;
    timer t;

    int num = 100000000;
    int *a = new int[num];
    memset(a, 0, num * sizeof(int));

    time.sleep_for_s(1);

    Msg_info("Used time: %.2lf (ns)", time.get_time_ns());
    Msg_info("Used time: %.2lf (us)", time.get_time_us());
    Msg_info("Used time: %.2lf (ms)", time.get_time_ms());
    Msg_info("Used time: %.2lf (s)", time.get_time_s());
    Msg_info("Used time: %.2lf (ms)", t.get_time_ms());

    time.reset_period_point();
    for (int i = 0; i < 8000; i++)
    {
        if (time.period_us(70))
        {
            Msg_info("[%d] = %d", i, i);
            time.reset_period_point();
        }
    }
*/
