#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Other/fileSystem_CPJ.hpp"
#include "Basic/Other/random_CPJ.hpp"
#include "Basic/Timer/timer_CPJ.hpp"
#include "Basic/Type/data_type.hpp"
#include "flag.hpp"
#include "single.hpp"
#include <libgen.h>
#include <sstream>
#include <string>
#include <vector>
namespace CPJ {

class ProcessorSpeed
{
  private:
    CSR_Result_type& csrResult_;

    static constexpr bool DEBUG_TIMER = true;

    const Algorithm_type algorithm_;
    const std::string& graphName_{""};
    const AUTO_GPUMEM_type autoGPUMemType_;
    const OrderMethod_type orderMethod_ = OrderMethod_type::CGgraphRV1_5;
    std::string speedBasePath_CPU_ = "";
    std::string speedBasePath_GPU_ = "";
    static constexpr int RUN_NUM = 5;
    static constexpr bool DEBUG_PRINTF = false;

    CPJ::CPU_BFS* cpu_bfs_{nullptr};
    CPJ::GPU_BFS* gpu_bfs_{nullptr};
    CPJ::CPU_SSSP* cpu_sssp_{nullptr};
    CPJ::GPU_SSSP* gpu_sssp_{nullptr};

  public:
    ProcessorSpeed(CSR_Result_type& csrResult, const Algorithm_type algorithm, const std::string& graphName, AUTO_GPUMEM_type autoGPUMemType)
        : csrResult_(csrResult), algorithm_(algorithm), graphName_(graphName), autoGPUMemType_(autoGPUMemType)
    {
        if (algorithm_ == Algorithm_type::SSSP)
        {
            assert_msg(csrResult.csr_weight != nullptr, "Algorithm_type::SSSP needs [csr_weight]");
        }

        if (autoGPUMemType == AUTO_GPUMEM_type::FULL_DEVICE_MEM)
        {
            speedBasePath_CPU_ =
                "FPS_CPU_" + graphName + "_" + OrderMethod_type_name[SCI32(orderMethod_)] + "_" + Algorithm_type_name[SCI32(algorithm_)] + ".bin";
            speedBasePath_GPU_ =
                "FPS_GPU_" + graphName + "_" + OrderMethod_type_name[SCI32(orderMethod_)] + "_" + Algorithm_type_name[SCI32(algorithm_)] + ".bin";
        }
        else if (autoGPUMemType == AUTO_GPUMEM_type::PARTIAL_DEVICE_MEM)
        {
            speedBasePath_CPU_ =
                "PPS_CPU_" + graphName + "_" + OrderMethod_type_name[SCI32(orderMethod_)] + "_" + Algorithm_type_name[SCI32(algorithm_)] + ".bin";
            speedBasePath_GPU_ =
                "PPS_GPU_" + graphName + "_" + OrderMethod_type_name[SCI32(orderMethod_)] + "_" + Algorithm_type_name[SCI32(algorithm_)] + ".bin";
        }
        else if ((autoGPUMemType_ == AUTO_GPUMEM_type::DISABLE_DEVICE_MEM) && (static_cast<GPU_memory_type>(FLAGS_gpuMemory) == GPU_memory_type::UVM))
        {
            speedBasePath_CPU_ =
                "UPS_CPU_" + graphName + "_" + OrderMethod_type_name[SCI32(orderMethod_)] + "_" + Algorithm_type_name[SCI32(algorithm_)] + ".bin";
            speedBasePath_GPU_ =
                "UPS_GPU_" + graphName + "_" + OrderMethod_type_name[SCI32(orderMethod_)] + "_" + Algorithm_type_name[SCI32(algorithm_)] + ".bin";
        }
        else if ((autoGPUMemType_ == AUTO_GPUMEM_type::DISABLE_DEVICE_MEM) &&
                 (static_cast<GPU_memory_type>(FLAGS_gpuMemory) == GPU_memory_type::ZERO_COPY))
        {
            speedBasePath_CPU_ =
                "ZPS_CPU_" + graphName + "_" + OrderMethod_type_name[SCI32(orderMethod_)] + "_" + Algorithm_type_name[SCI32(algorithm_)] + ".bin";
            speedBasePath_GPU_ =
                "ZPS_GPU_" + graphName + "_" + OrderMethod_type_name[SCI32(orderMethod_)] + "_" + Algorithm_type_name[SCI32(algorithm_)] + ".bin";
        }
        else
        {
            assert_msg(false, "Error, Switch GPU_Memory_type to [GPU_memory_type::ZERO_COPY] or GPU_memory_type::UVM");
        }
    }

    double getRatioApproximate()
    {
        assert_msg(RUN_NUM >= 3, "Too small RUN_NUM, which must >=3");
        double ratio = 0.0;

        std::string pre_CPU = "FPS_CPU";
        std::string pre_GPU = "FPS_GPU";
        auto vec_CPU = CPJ::FS::findFile_withPrefix("./", pre_CPU);
        auto vec_GPU = CPJ::FS::findFile_withPrefix("./", pre_GPU);
        if ((vec_CPU.size() != 0) && (vec_GPU.size() != 0))
        {
            for (int CPU_file_id = 0; CPU_file_id < vec_CPU.size(); CPU_file_id++)
            {
                std::string temp_CPU = vec_CPU[CPU_file_id].substr(0, pre_CPU.length());
                for (int GPU_file_id = 0; GPU_file_id < vec_CPU.size(); GPU_file_id++)
                {
                    if (vec_GPU[GPU_file_id].ends_with(temp_CPU))
                    {
                        std::vector<std::vector<SpeedRecord_type>> vec_CPU_new = readRecordFromFile(vec_CPU[CPU_file_id]);
                        std::vector<std::vector<SpeedRecord_type>> vec_GPU_new = readRecordFromFile(vec_GPU[GPU_file_id]);
                        double rat_cpu = 0.0;
                        double rat_gpu = 0.0;
                        for (int i = 0; i < vec_CPU_new.size(); i++)
                        {
                            rat_cpu += vec_CPU_new[i].back().total_time_ms;
                            rat_gpu += vec_GPU_new[i].back().total_time_ms;
                        }
                        ratio = (rat_cpu) / (rat_gpu);
                        Msg_info("[AUTO_GPUMEM_type::PARTIAL_DEVICE_MEM] used file[%s] and file[%s] to get SpeedRatio = %.2lf",
                                 vec_CPU[CPU_file_id].c_str(), vec_GPU[GPU_file_id].c_str(), ratio);
                        return ratio;
                    }
                }
            }

            assert_msg(false, "Error in Func[%s]", __FUNCTION__);
        }
        else
        {
            CPJ::Timer time;
            if (algorithm_ == Algorithm_type::BFS)
            {
                cpu_bfs_ = new CPJ::CPU_BFS(csrResult_);
                if (autoGPUMemType_ == AUTO_GPUMEM_type::FULL_DEVICE_MEM)
                {
                    gpu_bfs_ = new CPJ::GPU_BFS(csrResult_, GPU_memory_type::GPU_MEM);
                }
                else
                {
                    assert_msg(false, "You must first run a graph data that can be fully stored by the current GPU, SpeedRatio needed");
                }
            }
            else if (algorithm_ == Algorithm_type::SSSP)
            {
                cpu_sssp_ = new CPJ::CPU_SSSP(csrResult_, true);
                if (autoGPUMemType_ == AUTO_GPUMEM_type::FULL_DEVICE_MEM)
                {
                    gpu_sssp_ = new CPJ::GPU_SSSP(csrResult_, GPU_memory_type::GPU_MEM, true);
                }
                else
                {
                    assert_msg(false, "You must first run a graph data that can be fully stored by the current GPU, SpeedRatio needed");
                }
            }
            else
            {
                assert_msg(false, "WCC and PageRank wait...");
            }

            std::vector<std::vector<SpeedRecord_type>> vec_CPU;
            std::vector<std::vector<SpeedRecord_type>> vec_GPU;
            vec_CPU.reserve(RUN_NUM);
            vec_GPU.reserve(RUN_NUM);
            vec_CPU.clear();
            vec_GPU.clear();
            while (vec_CPU.size() < RUN_NUM)
            {
                std::vector<count_type> root_random_vec =
                    CPJ::Random::generateRandomNumbers_uniform(static_cast<count_type>(0), csrResult_.vertexNum, 1);
                int64_t root_random = static_cast<int64_t>(root_random_vec[0]);

                time.start();
                std::vector<SpeedRecord_type> measure_CPU;
                if (algorithm_ == Algorithm_type::BFS) measure_CPU = CPU_BFS(cpu_bfs_, root_random);
                else if (algorithm_ == Algorithm_type::SSSP) measure_CPU = CPU_SSSP(cpu_sssp_, root_random);
                else
                {
                    assert_msg(false, "WCC and PageRank wait...");
                }

                if (measure_CPU.size() != 0)
                {
                    vec_CPU.push_back(measure_CPU);
                    Msg_info("The (%zu) measurement of CPU, Used time: %s", vec_CPU.size(), time.get_time_str().c_str());

                    time.start();
                    std::vector<SpeedRecord_type> measure_GPU;
                    if (algorithm_ == Algorithm_type::BFS) measure_GPU = GPU(gpu_bfs_, root_random);
                    else if (algorithm_ == Algorithm_type::SSSP) measure_GPU = GPU(gpu_sssp_, root_random);
                    else
                    {
                        assert_msg(false, "WCC and PageRank wait...");
                    }
                    vec_GPU.push_back(measure_GPU);
                    Msg_info("The (%zu) measurement of GPU, Used time: %s", vec_GPU.size(), time.get_time_str().c_str());
                }
            }
            if (algorithm_ == Algorithm_type::BFS)
            {
                cpu_bfs_->freeAssistCPUMemory();
                gpu_bfs_->freeGPUMemory();
            }
            else if (algorithm_ == Algorithm_type::SSSP)
            {
                cpu_sssp_->freeAssistCPUMemory();
                cpu_sssp_->freeSSSP_destWeight();
                gpu_sssp_->freeGPUMemory();
            }
            else
            {
                assert_msg(false, "WCC and PageRank wait...");
            }

            // clang-format off
            std::sort(vec_CPU.begin(), vec_CPU.end(),
                [&](std::vector<SpeedRecord_type>& a, std::vector<SpeedRecord_type>& b)
                {
                    return a.back().total_time_ms < b.back().total_time_ms;
                }
            );

            std::sort(vec_GPU.begin(), vec_GPU.end(),
                [&](std::vector<SpeedRecord_type>& a, std::vector<SpeedRecord_type>& b)
                {
                    return a.back().total_time_ms < b.back().total_time_ms;
                }
            );
            // clang-format on

            std::vector<std::vector<SpeedRecord_type>> vec_CPU_new(vec_CPU.begin() + 1, vec_CPU.end() - 1);
            assert_msg(vec_CPU_new.size() == (RUN_NUM - 2), "Error vec_CPU_new size, size = %zu", vec_CPU_new.size());

            std::vector<std::vector<SpeedRecord_type>> vec_GPU_new(vec_GPU.begin() + 1, vec_GPU.end() - 1);
            assert_msg(vec_GPU_new.size() == (RUN_NUM - 2), "Error vec_GPU_new size, size = %zu", vec_GPU_new.size());

            writeRecordToFile(vec_CPU_new, speedBasePath_CPU_);
            writeRecordToFile(vec_GPU_new, speedBasePath_GPU_);

            if constexpr (DEBUG_PRINTF)
            {
                // 输出数据
                std::stringstream ss;
                ss << "Write To File: " << std::endl;
                for (const auto& outer : vec_CPU_new)
                {
                    for (const auto& record : outer)
                    {
                        ss << "[" << record.iteId << "]: "
                           << "activeVertexNum(" << record.activeVertexNum << "), activeEdgeNum(" << record.activeEdgeNum
                           << "), time: " << record.time_ms << " (ms), totalTime: " << record.total_time_ms << " (ms)" << std::endl;
                    }
                    ss << std::endl;
                }
                ss << "-------------------------------------" << std::endl;
                for (const auto& outer : vec_GPU_new)
                {
                    for (const auto& record : outer)
                    {
                        ss << "[" << record.iteId << "]: "
                           << "activeVertexNum(" << record.activeVertexNum << "), activeEdgeNum(" << record.activeEdgeNum
                           << "), time: " << record.time_ms << " (ms), totalTime: " << record.total_time_ms << " (ms)" << std::endl;
                    }
                    ss << std::endl;
                }

                Msg_info("%s", ss.str().c_str());
            }

            double rat_cpu = 0.0;
            double rat_gpu = 0.0;
            for (int i = 0; i < vec_CPU_new.size(); i++)
            {
                rat_cpu += vec_CPU_new[i].back().total_time_ms;
                rat_gpu += vec_GPU_new[i].back().total_time_ms;
            }
            ratio = (rat_cpu) / (rat_gpu);
        }
        return ratio;
    }

    double getRatio()
    {
        assert_msg(RUN_NUM >= 3, "Too small RUN_NUM, which must >=3");
        double ratio = 0.0;

        if (CPJ::FS::isExist(speedBasePath_CPU_) && CPJ::FS::isExist(speedBasePath_GPU_))
        {
            std::vector<std::vector<SpeedRecord_type>> vec_CPU_new = readRecordFromFile(speedBasePath_CPU_);
            std::vector<std::vector<SpeedRecord_type>> vec_GPU_new = readRecordFromFile(speedBasePath_GPU_);

            if constexpr (DEBUG_PRINTF)
            {
                // 输出数据
                std::stringstream ss;
                ss << "Write To File: " << std::endl;
                for (const auto& outer : vec_CPU_new)
                {
                    for (const auto& record : outer)
                    {
                        ss << "[" << record.iteId << "]: "
                           << "activeVertexNum(" << record.activeVertexNum << "), activeEdgeNum(" << record.activeEdgeNum
                           << "), time: " << record.time_ms << " (ms), totalTime: " << record.total_time_ms << " (ms)" << std::endl;
                    }
                    ss << std::endl;
                }
                ss << "-------------------------------------" << std::endl;
                for (const auto& outer : vec_GPU_new)
                {
                    for (const auto& record : outer)
                    {
                        ss << "[" << record.iteId << "]: "
                           << "activeVertexNum(" << record.activeVertexNum << "), activeEdgeNum(" << record.activeEdgeNum
                           << "), time: " << record.time_ms << " (ms), totalTime: " << record.total_time_ms << " (ms)" << std::endl;
                    }
                    ss << std::endl;
                }

                Msg_info("%s", ss.str().c_str());
            }
            double rat_cpu = 0.0;
            double rat_gpu = 0.0;
            for (int i = 0; i < vec_CPU_new.size(); i++)
            {
                rat_cpu += vec_CPU_new[i].back().total_time_ms;
                rat_gpu += vec_GPU_new[i].back().total_time_ms;
            }
            ratio = (rat_cpu) / (rat_gpu);
        }
        else
        {
            CPJ::Timer time;

            if (algorithm_ == Algorithm_type::BFS) cpu_bfs_ = new CPJ::CPU_BFS(csrResult_);
            else if (algorithm_ == Algorithm_type::SSSP) cpu_sssp_ = new CPJ::CPU_SSSP(csrResult_, true);
            else
            {
                assert_msg(false, "WCC and PageRank wait...");
            }

            if (autoGPUMemType_ == AUTO_GPUMEM_type::FULL_DEVICE_MEM)
            {
                if (algorithm_ == Algorithm_type::BFS) gpu_bfs_ = new CPJ::GPU_BFS(csrResult_, GPU_memory_type::GPU_MEM);
                else if (algorithm_ == Algorithm_type::SSSP) gpu_sssp_ = new CPJ::GPU_SSSP(csrResult_, GPU_memory_type::GPU_MEM, true);
                else assert_msg(false, "WCC and PageRank wait...");
            }
            else if (autoGPUMemType_ == AUTO_GPUMEM_type::PARTIAL_DEVICE_MEM)
            {
                std::string pre_CPU = "FPS_CPU";
                std::string pre_GPU = "FPS_GPU";
                auto vec_CPU = CPJ::FS::findFile_withPrefix("./", pre_CPU);
                auto vec_GPU = CPJ::FS::findFile_withPrefix("./", pre_GPU);

                // 进一步过滤OrderMethod
                for (int CPU_file_id = 0; CPU_file_id < vec_CPU.size(); CPU_file_id++)
                {
                    if (vec_CPU[CPU_file_id].find(OrderMethod_type_name[SCI32(orderMethod_)]) == std::string::npos)
                    {
                        vec_CPU.erase(vec_CPU.begin() + CPU_file_id);
                    }
                }
                for (int GPU_file_id = 0; GPU_file_id < vec_GPU.size(); GPU_file_id++)
                {
                    if (vec_GPU[GPU_file_id].find(OrderMethod_type_name[SCI32(orderMethod_)]) == std::string::npos)
                    {
                        vec_GPU.erase(vec_GPU.begin() + GPU_file_id);
                    }
                }

                // 进一步过滤Algorithm
                for (int CPU_file_id = 0; CPU_file_id < vec_CPU.size(); CPU_file_id++)
                {
                    if (vec_CPU[CPU_file_id].find(Algorithm_type_name[SCI32(algorithm_)]) == std::string::npos)
                    {
                        vec_CPU.erase(vec_CPU.begin() + CPU_file_id);
                    }
                }
                for (int GPU_file_id = 0; GPU_file_id < vec_GPU.size(); GPU_file_id++)
                {
                    if (vec_GPU[GPU_file_id].find(Algorithm_type_name[SCI32(algorithm_)]) == std::string::npos)
                    {
                        vec_GPU.erase(vec_GPU.begin() + GPU_file_id);
                    }
                }

                if ((vec_CPU.size() != 0) && (vec_GPU.size() != 0))
                {
                    for (int CPU_file_id = 0; CPU_file_id < vec_CPU.size(); CPU_file_id++)
                    {
                        std::string temp_CPU = vec_CPU[CPU_file_id].substr(pre_CPU.length());
                        for (int GPU_file_id = 0; GPU_file_id < vec_CPU.size(); GPU_file_id++)
                        {
                            if (vec_GPU[GPU_file_id].ends_with(temp_CPU))
                            {
                                Msg_info("Used file[%s] and [%s] to get speedRatio",
                                         static_cast<const char*>(basename((char*)vec_CPU[CPU_file_id].c_str())),
                                         static_cast<const char*>(basename((char*)vec_GPU[GPU_file_id].c_str())));
                                std::vector<std::vector<SpeedRecord_type>> vec_CPU_new = readRecordFromFile(vec_CPU[CPU_file_id]);
                                std::vector<std::vector<SpeedRecord_type>> vec_GPU_new = readRecordFromFile(vec_GPU[GPU_file_id]);
                                double rat_cpu = 0.0;
                                double rat_gpu = 0.0;
                                for (int i = 0; i < vec_CPU_new.size(); i++)
                                {
                                    rat_cpu += vec_CPU_new[i].back().total_time_ms;
                                    rat_gpu += vec_GPU_new[i].back().total_time_ms;
                                }
                                ratio = (rat_cpu) / (rat_gpu);
                                Msg_info("[AUTO_GPUMEM_type::PARTIAL_DEVICE_MEM] used file[%s] and file[%s] to get SpeedRatio = %.2lf",
                                         vec_CPU[CPU_file_id].c_str(), vec_GPU[GPU_file_id].c_str(), ratio);
                                return ratio;
                            }
                        }
                    }
                    assert_msg(false, "Can not find any satisfied file in Func[%s]", __FUNCTION__);
                }
                else
                {
                    assert_msg(false, "You must first run a graph data that can be fully stored by the current GPU, SpeedRatio needed");
                }
            }
            else if (autoGPUMemType_ == AUTO_GPUMEM_type::DISABLE_DEVICE_MEM)
            {
                if (algorithm_ == Algorithm_type::BFS) gpu_bfs_ = new CPJ::GPU_BFS(csrResult_, static_cast<GPU_memory_type>(FLAGS_gpuMemory));
                else if (algorithm_ == Algorithm_type::SSSP)
                    gpu_sssp_ = new CPJ::GPU_SSSP(csrResult_, static_cast<GPU_memory_type>(FLAGS_gpuMemory), true);
                else assert_msg(false, "WCC and PageRank wait...");
            }
            else if (autoGPUMemType_ == AUTO_GPUMEM_type::DISABLE_GPU)
            {
                Msg_warn("Disable GPU ?");
                return 0.0;
            }

            std::vector<std::vector<SpeedRecord_type>> vec_CPU;
            std::vector<std::vector<SpeedRecord_type>> vec_GPU;
            vec_CPU.reserve(RUN_NUM);
            vec_GPU.reserve(RUN_NUM);
            vec_CPU.clear();
            vec_GPU.clear();
            while (vec_CPU.size() < RUN_NUM)
            {
                time.start();
                std::vector<count_type> root_random_vec =
                    CPJ::Random::generateRandomNumbers_uniform(static_cast<count_type>(0), csrResult_.vertexNum, 1);
                int64_t root_random = static_cast<int64_t>(root_random_vec[0]);

                std::vector<SpeedRecord_type> measure_CPU;
                if (algorithm_ == Algorithm_type::BFS) measure_CPU = CPU_BFS(cpu_bfs_, root_random);
                else if (algorithm_ == Algorithm_type::SSSP) measure_CPU = CPU_SSSP(cpu_sssp_, root_random);
                else
                {
                    assert_msg(false, "WCC and PageRank wait...");
                }

                if (measure_CPU.size() != 0)
                {
                    vec_CPU.push_back(measure_CPU);
                    std::vector<SpeedRecord_type> measure_GPU;
                    if (algorithm_ == Algorithm_type::BFS) measure_GPU = GPU(gpu_bfs_, root_random);
                    else if (algorithm_ == Algorithm_type::SSSP) measure_GPU = GPU(gpu_sssp_, root_random);
                    else
                    {
                        assert_msg(false, "WCC and PageRank wait...");
                    }
                    vec_GPU.push_back(measure_GPU);
                }
                Msg_info("The (%zu) measurement finished, Used time: %s", vec_CPU.size(), time.get_time_str().c_str());
            }
            if (algorithm_ == Algorithm_type::BFS)
            {
                cpu_bfs_->freeAssistCPUMemory();
                gpu_bfs_->freeGPUMemory();
            }
            else if (algorithm_ == Algorithm_type::SSSP)
            {
                cpu_sssp_->freeAssistCPUMemory();
                cpu_sssp_->freeSSSP_destWeight();
                gpu_sssp_->freeGPUMemory();
            }
            else
            {
                assert_msg(false, "WCC and PageRank wait...");
            }

            // clang-format off
            std::sort(vec_CPU.begin(), vec_CPU.end(),
                [&](std::vector<SpeedRecord_type>& a, std::vector<SpeedRecord_type>& b)
                {
                    return a.back().total_time_ms < b.back().total_time_ms;
                }
            );

            std::sort(vec_GPU.begin(), vec_GPU.end(),
                [&](std::vector<SpeedRecord_type>& a, std::vector<SpeedRecord_type>& b)
                {
                    return a.back().total_time_ms < b.back().total_time_ms;
                }
            );
            // clang-format on

            std::vector<std::vector<SpeedRecord_type>> vec_CPU_new(vec_CPU.begin() + 1, vec_CPU.end() - 1);
            assert_msg(vec_CPU_new.size() == (RUN_NUM - 2), "Error vec_CPU_new size, size = %zu", vec_CPU_new.size());

            std::vector<std::vector<SpeedRecord_type>> vec_GPU_new(vec_GPU.begin() + 1, vec_GPU.end() - 1);
            assert_msg(vec_GPU_new.size() == (RUN_NUM - 2), "Error vec_GPU_new size, size = %zu", vec_GPU_new.size());

            writeRecordToFile(vec_CPU_new, speedBasePath_CPU_);
            writeRecordToFile(vec_GPU_new, speedBasePath_GPU_);

            if constexpr (DEBUG_PRINTF)
            {
                // 输出数据
                std::stringstream ss;
                ss << "Write To File: " << std::endl;
                for (const auto& outer : vec_CPU_new)
                {
                    for (const auto& record : outer)
                    {
                        ss << "[" << record.iteId << "]: "
                           << "activeVertexNum(" << record.activeVertexNum << "), activeEdgeNum(" << record.activeEdgeNum
                           << "), time: " << record.time_ms << " (ms), totalTime: " << record.total_time_ms << " (ms)" << std::endl;
                    }
                    ss << std::endl;
                }
                ss << "-------------------------------------" << std::endl;
                for (const auto& outer : vec_GPU_new)
                {
                    for (const auto& record : outer)
                    {
                        ss << "[" << record.iteId << "]: "
                           << "activeVertexNum(" << record.activeVertexNum << "), activeEdgeNum(" << record.activeEdgeNum
                           << "), time: " << record.time_ms << " (ms), totalTime: " << record.total_time_ms << " (ms)" << std::endl;
                    }
                    ss << std::endl;
                }

                Msg_info("%s", ss.str().c_str());
            }

            double rat_cpu = 0.0;
            double rat_gpu = 0.0;
            for (int i = 0; i < vec_CPU_new.size(); i++)
            {
                rat_cpu += vec_CPU_new[i].back().total_time_ms;
                rat_gpu += vec_GPU_new[i].back().total_time_ms;
            }
            ratio = (rat_cpu) / (rat_gpu);
        }
        return ratio;
    }

  private:
    std::vector<SpeedRecord_type> CPU_BFS(CPJ::CPU_BFS* do_CPU, int64_t root)
    {
        assert_msg(algorithm_ == Algorithm_type::BFS, "Must be  Algorithm_type::BFS");
        return do_CPU->measureBFS(root);
    }

    std::vector<SpeedRecord_type> CPU_SSSP(CPJ::CPU_SSSP* do_CPU, int64_t root)
    {
        assert_msg(algorithm_ == Algorithm_type::SSSP, "Must be  Algorithm_type::SSSP");
        return do_CPU->measureSSSP(root);
    }
    std::vector<SpeedRecord_type> GPU(CPJ::GPU_BFS* do_GPU, int64_t root)
    {
        assert_msg(algorithm_ == Algorithm_type::BFS, "Must be  Algorithm_type::BFS");
        return do_GPU->measureBFS(root);
    }
    std::vector<SpeedRecord_type> GPU(CPJ::GPU_SSSP* do_GPU, int64_t root)
    {
        assert_msg(algorithm_ == Algorithm_type::SSSP, "Must be  Algorithm_type::SSSP");
        return do_GPU->measureSSSP(root);
    }

    void writeRecordToFile(const std::vector<std::vector<SpeedRecord_type>>& data, const std::string& filename)
    {
        CPJ::Timer timer;
        std::ofstream outFile(filename, std::ios::binary);
        if (!outFile)
        {
            std::cerr << "Failed to open the file for writing." << std::endl;
            return;
        }

        size_t outerSize = data.size();
        outFile.write(reinterpret_cast<const char*>(&outerSize), sizeof(outerSize));

        for (const auto& innerVector : data)
        {
            size_t innerSize = innerVector.size();
            outFile.write(reinterpret_cast<const char*>(&innerSize), sizeof(innerSize));
            outFile.write(reinterpret_cast<const char*>(innerVector.data()), sizeof(SpeedRecord_type) * innerSize);
        }

        outFile.close();
        Msg_info("The record write to file[%s] finish, Used time: %s", filename.c_str(), timer.get_time_str().c_str());
    }

    std::vector<std::vector<SpeedRecord_type>> readRecordFromFile(const std::string& filename)
    {
        std::ifstream inFile(filename, std::ios::binary);
        if (!inFile)
        {
            std::cerr << "Failed to open the file for reading." << std::endl;
            return {};
        }

        size_t outerSize;
        inFile.read(reinterpret_cast<char*>(&outerSize), sizeof(outerSize));

        std::vector<std::vector<SpeedRecord_type>> data(outerSize);

        for (auto& innerVector : data)
        {
            size_t innerSize;
            inFile.read(reinterpret_cast<char*>(&innerSize), sizeof(innerSize));
            innerVector.resize(innerSize);

            inFile.read(reinterpret_cast<char*>(innerVector.data()), sizeof(SpeedRecord_type) * innerSize);
        }

        inFile.close();

        return data;
    }
};
} // namespace CPJ