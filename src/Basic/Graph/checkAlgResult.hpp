#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/IO/io_common.hpp"
#include "Basic/Thread/omp_def.hpp"
#include "Basic/Type/data_type.hpp"

#include <cstdlib>
#include <filesystem>
#include <limits>
#include <numaif.h> // move_page
#include <string>
#include <type_traits>

inline std::string getBaseGraphFilePath() { return "/data/webgraph/bin/"; }

std::string getAlgName(Algorithm_type alg)
{
    std::string algorithm = "";

    if (alg == Algorithm_type::BFS) return algorithm = "BFS";
    else if (alg == Algorithm_type::SSSP) return algorithm = "SSSP";
    else
    {
        assert_msg(false, "getAlgName ERROR");
        return "";
    }
}

std::string getResultFilePathV2(const CheckInfo_type& checkInfo)
{
    std::string rightFile = "";
    std::string algorithmName = Algorithm_type_name[SCI32(checkInfo.algorithm)];

    switch (checkInfo.algorithm)
    {
    case Algorithm_type::BFS:
        rightFile = "/data/webgraph/checkResult/" + algorithmName + "/" + checkInfo.graphName + "_" + algorithmName + "_" +
                    std::to_string(checkInfo.root) + ".bin";
        if (!std::filesystem::exists(rightFile))
        {
            Msg_error("When Check Result, Can Not Find Compared File[%s], Skip Result Check...", rightFile.c_str());
            return "";
        }
        return rightFile;
        break;
    case Algorithm_type::SSSP:
        rightFile = "/data/webgraph/checkResult/" + algorithmName + "/" + checkInfo.graphName + "_" + algorithmName + "_" +
                    std::to_string(checkInfo.root) + ".bin";
        if (!std::filesystem::exists(rightFile))
        {
            Msg_error("When Check Result, Can Not Find Compared File[%s], Skip Result Check...", rightFile.c_str());
            return "";
        }
        return rightFile;
        break;
    case Algorithm_type::MAX_VALUE:
        Msg_error("When Check Result, Get invalid Algorithm, Skip Result Check...");
        return "";
        break;
    }
    return "";
}

std::string getResultFilePath(CheckInfo_type& checkInfo)
{
    std::string rightFile = "";
    if (checkInfo.algorithm == Algorithm_type::BFS)
    {
        rightFile = "/data/webgraph/checkResult/BFS/" + checkInfo.graphName + "_" + getAlgName(checkInfo.algorithm) + "_" +
                    std::to_string(checkInfo.root) + ".bin";

        bool isExist = (access(rightFile.c_str(), F_OK) >= 0);
        if (!isExist)
        {
            Msg_error("getResultFilePath时, 未发现对应的[%s]文件, 跳过检查...", rightFile.c_str());
            exit(1);
        }

        return rightFile;
    }
    else if (checkInfo.algorithm == Algorithm_type::SSSP)
    {
        rightFile = "/data/webgraph/checkResult/SSSP/" + checkInfo.graphName + "_" + getAlgName(checkInfo.algorithm) + "_" +
                    std::to_string(checkInfo.root) + ".bin";

        bool isExist = (access(rightFile.c_str(), F_OK) >= 0);
        if (!isExist)
        {
            assert_msg(false, "getResultFilePath时, 未发现对应的[%s]文件", rightFile.c_str());
        }

        return rightFile;
    }
    else
    {
        assert_msg(false, "getResultFilePath时, 未知算法");
        return "";
    }
}