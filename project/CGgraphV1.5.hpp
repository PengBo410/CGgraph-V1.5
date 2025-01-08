#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Graph/checkAlgResult.hpp"
#include "Basic/Graph/graphFileList.hpp"
#include "CG_BFS.hpp"
#include "CG_SSSP.hpp"
#include "flag.hpp"
#include "processorSpeed.hpp"
#include "single.hpp"
#include "subgraphExtraction.hpp"
#include <string>

void Run_CGgraph()
{
    Algorithm_type algorithm = static_cast<Algorithm_type>(FLAGS_algorithm);

    CSR_Result_type csrResult;
    vertex_id_type* old2new = CPJ::getGraphData(csrResult, FLAGS_graphName, true, false, false);
    int64_t root = old2new[FLAGS_root];

    CPJ::SubgraphExt subgraphExt(csrResult, static_cast<Algorithm_type>(FLAGS_algorithm), FLAGS_useDeviceId);
    auto gpuMemType = subgraphExt.getGPUMemType();
    CPJ::ProcessorSpeed speed(csrResult, algorithm, FLAGS_graphName, gpuMemType);
    double ratio = speed.getRatio();

    if (ratio <= 0.5)
    {
        Msg_warn("GPU is too slow, CPU-ONLY RUN");
        if (algorithm == Algorithm_type::BFS)
        {
            CPJ::CPU_BFS cpu_bfs(csrResult);
            cpu_bfs.measureBFS(root);
        }
        else if (algorithm == Algorithm_type::SSSP)
        {
            CPJ::CPU_SSSP cpu_sssp(csrResult, true);
            cpu_sssp.measureSSSP(root);
        }
        else
        {
            assert_msg(false, "Unknow algorithm");
        }
    }
    else
    {
        Msg_info("Begin: usedGPUMem = %.2lf (GB)",
                 BYTES_TO_GB(CPJ::MemoryInfo::getMemoryTotal_Device(FLAGS_useDeviceId) - CPJ::MemoryInfo::getMemoryFree_Device(FLAGS_useDeviceId)));
        Host_dataPointer_type hostData;
        Device_dataPointer_type deviceData;
        subgraphExt.extractSubgraph(hostData, deviceData, gpuMemType);
        if (algorithm == Algorithm_type::BFS)
        {
            double total_time{0.0};
            CPJ::CG_BFS cg_bfs(hostData, deviceData, gpuMemType, ratio);
            for (int run_id = 0; run_id < FLAGS_runs; run_id++)
            {
                total_time += cg_bfs.doBFS(root);
            }
            Msg_info("Total avg time: %.3lf (ms)", total_time / FLAGS_runs);
        }
        else if (algorithm == Algorithm_type::SSSP)
        {
            double total_time{0.0};
            CPJ::CG_SSSP cg_sssp(hostData, deviceData, gpuMemType, ratio);
            for (int run_id = 0; run_id < FLAGS_runs; run_id++)
            {
                total_time += cg_sssp.doSSSP(root);
            }
            Msg_finish("[Complete]: CGgraph total avg time: %.3lf (ms)", total_time / FLAGS_runs);
        }
        else
        {
            assert_msg(false, "Unknow algorithm");
        }
    }
}