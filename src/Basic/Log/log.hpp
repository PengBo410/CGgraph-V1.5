#pragma once

/* 不要在此文件中调用console_V3_3.hpp */
#include "Basic/Console/console_var.hpp"
#include <assert.h>
#include <filesystem>
#include <fstream>
#include <iomanip> // std::setprecision
#include <ios>
#include <iostream> // std::cout, std::endl, std::flush, std::ostream
#include <libgen.h> // basename(...)
#include <mutex>    // std::mutex
#include <omp.h>
#include <regex>   //正则
#include <utility> // std::forward, std::move, std::swap

// 我们暂时放弃LOG_LEVEL
// #define LOG_EVERYTHING 0 // 打印任何输出
// #define LOG_DEBUG 1
// #define LOG_INFO 2
// #define LOG_EMPH 3
// #define LOG_WARNING 4
// #define LOG_ERROR 5
// #define LOG_FATAL 6
// #define LOG_NONE 7 // 什么都不打印

// #ifndef OUTPUTLEVEL
// #define OUTPUTLEVEL LOG_DEBUG
// #endif

#define LOG_SPECIAL 0 // 是否在LOG中输出宽度和精度

class [[nodiscard]] Log_File
{

  private:
    std::ofstream out_stream;
    std::string log_file = "";
    mutable std::mutex stream_mutex = {};

  public:
    // 构造函数
    Log_File() {}

    // 析构函数
    ~Log_File() { out_stream.close(); }

    void myClose() { out_stream.close(); }

    std::string get_log_file() { return log_file; }

    // 实例化
    bool set_log_file(std::string file = "default")
    {
        // 如果当前out_stream已经开启则关闭
        if (out_stream.good())
        {
            out_stream.flush();
            out_stream.close();
            log_file = "";
        }

        if (file == "default")
        {
            std::filesystem::path executablePath = std::filesystem::current_path();
            auto lastPath = executablePath.parent_path();
            const std::string log_folder = lastPath.string() + "/log";
            const std::string log_file_ = log_folder + "/z_" + Console_Val::serverName + ".log";
            std::filesystem::path filePath_fs(log_folder);
            bool is = std::filesystem::exists(filePath_fs);
            if (!is)
            {
                std::error_code ec;
                std::filesystem::create_directories(log_folder, ec);
            }
            log_file = log_file_;
        }
        else
        {
            log_file = file;
        }

        out_stream.open(log_file.c_str());
        if (!out_stream.is_open())
        {
            printf("LogFile [%s] open error", log_file.c_str());
            assert(false);
            return false;
        }

        return true;
    }

    //> 以下的代码我们只是为了统一printf的输出
    template <typename... Args>
    void log(int serverId, int threadId, const char* file, int line, const std::string& format, Args&&... args)
    {
        const std::scoped_lock lock(stream_mutex); // 加锁避免多线程打印冲突
        out_stream << "[N(" << std::setw(1) << serverId << ")-T(" << std::setw(2) << threadId << ")]: ";
        log_cpj(format, std::forward<Args>(args)...);
        out_stream << " -> [" << file << ":" << line << " 行]\n";
    }

    void myFlush()
    {
        out_stream << std::flush; // 立即刷新缓冲区，将数据写入文件
    }

  private:
    // 基础
    void log_cpj(const std::string& format) { out_stream << format.c_str(); }

    // 递归
    template <typename T, typename... Args>
    void log_cpj(const std::string& format, T&& arg, Args&&... args)
    {
        std::regex placeholderRegex("%[0-9]*\\.?[0-9]*[lh]?[a-zA-Z]*"); // 匹配C中的占位符
        std::smatch match;
        if (std::regex_search(format, match, placeholderRegex))
        {
            std::string placeholder = match[0].str();
            size_t pos = format.find(placeholder);
            // printf("%s\n", placeholder.c_str());
            out_stream << format.substr(0, pos); // 将当前占位符前面的字符串先输出
            if constexpr (LOG_SPECIAL == 1) special_cpj(placeholder);
            out_stream << std::forward<T>(arg);                                      // 将实际的数据输出
            std::string remainingFormat = format.substr(pos + placeholder.length()); // 截取出后面的字符串
            log_cpj(remainingFormat, std::forward<Args>(args)...);
        }
    }

    void special_cpj(const std::string& holder)
    {
        size_t found = holder.find("s");
        if (found != std::string::npos)
        {
            return;
        }
        found = holder.find(".");
        //^ 包含"."
        if (found != std::string::npos)
        {
            std::regex pattern(
                "%([0-9]*)\\.(\\d*)"); // 匹配%x.y, +有可能让td::regex_search为false, 但*总是让td::regex_search为true,但matches[1].str()可能为empty
            std::smatch matches;

            if (std::regex_search(holder, matches, pattern))
            {
                // 输出整数部分，如果为空，默认为0
                std::string integerPart = matches[1].str();
                int integer = (integerPart.empty() ? 0 : std::stoi(integerPart));

                // 输出小数部分，如果为空，默认为0
                std::string decimalPart = matches[2].str();
                int decimal = (decimalPart.empty() ? 0 : std::stoi(decimalPart));

                // std::cout << "整数部分: " << integer << std::endl; // 输出整数部分，如果为空，默认为0
                // std::cout << "小数部分: " << decimal << std::endl; // 输出小数部分，如果为空，默认为0

                out_stream << std::setw(integer) << std::setprecision(decimal);
            }
            else
            {
                printf("[special_cpj]: (%s)未找到匹配的模式. -> [%s:%d 行]\n", holder.c_str(), basename((char*)(__FILE__)), __LINE__);
            }
        }
        //^ 不包含"."
        else
        {
            std::regex pattern("%(\\d*)"); // 匹配%x; 其中, ([0-9]+) 和(\\d+)的效果是一样的, ()是必须的
            std::smatch matches;

            if (std::regex_search(holder, matches, pattern))
            {
                // 输出整数部分，如果为空，默认为0
                std::string integerPart = matches[1].str();
                int integer = (integerPart.empty() ? 0 : std::stoi(integerPart));

                // std::cout << "整数部分: " << integer << std::endl; // 输出整数部分，如果为空，默认为0
                out_stream << std::setw(integer);
            }
            else
            {
                printf("[special_cpj]: (%s)未找到匹配的模式. -> [%s:%d 行]\n", holder.c_str(), basename((char*)(__FILE__)), __LINE__);
            }
        }
    }

  private:
    inline static std::ostream& (&endl)(std::ostream&) = static_cast<std::ostream& (&)(std::ostream&)>(std::endl);
    inline static std::ostream& (&flush)(std::ostream&) = static_cast<std::ostream& (&)(std::ostream&)>(std::flush);

}; // end of class [Log_File]

inline Log_File& global_logFile()
{
    static Log_File log;
    return log;
}

#define Log_info(format, ...)                                                                                                                        \
    global_logFile().log(Console_Val::serverId, omp_get_thread_num(), basename((char*)__FILE__), __LINE__, format, ##__VA_ARGS__);
