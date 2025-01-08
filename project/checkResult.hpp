#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Graph/checkAlgResult.hpp"
#include "Basic/IO/io_adapter_V1.hpp"
#include "Basic/Timer/timer_CPJ.hpp"
#include "Basic/Type/data_type.hpp"
#include <limits>
#include <omp.h>
#include <string>

namespace CPJ {

template <typename Result_type>
class CheckResult
{
  public:
    CheckResult(const CheckInfo_type& checkInfo_type, const count_type& vertexNum, Result_type* result, vertex_id_type* old2new)
    {
        std::string comparedResultFile = getResultFilePathV2(checkInfo_type);
        {
            CPJ::IOAdaptor ioAdapter(comparedResultFile);
            ioAdapter.openFile();
            std::unique_ptr<vertex_data_type[]> result_right(ioAdapter.readBinFileEntire_sync<vertex_data_type>(omp_get_max_threads()));
#pragma omp parallel for
            for (vertex_id_type vertex_id = 0; vertex_id < vertexNum; vertex_id++)
            {
                if (checkInfo_type.orderMethod == OrderMethod_type::NATIVE)
                {
                    if (result_right[vertex_id] == 999999999)
                    {
                        assert_msg(result[vertex_id] == std::numeric_limits<vertex_data_type>::max(), "result_right[%zu] = %zu, result[%zu] = %zu",
                                   SCU64(vertex_id), SCU64(result_right[vertex_id]), SCU64(vertex_id), SCU64(result[vertex_id]));
                    }
                    else
                    {
                        assert_msg(result_right[vertex_id] == result[vertex_id], "result_right[%zu] = %zu, result[%zu] = %zu", SCU64(vertex_id),
                                   SCU64(result_right[vertex_id]), SCU64(vertex_id), SCU64(result[vertex_id]));
                    }
                }
                else
                {
                    if (result_right[vertex_id] == 999999999)
                    {
                        vertex_id_type transformVertexId = old2new[vertex_id];
                        assert_msg(result[transformVertexId] == std::numeric_limits<vertex_data_type>::max(),
                                   "result_right[%zu] = %zu, result[%zu] = %zu", SCU64(vertex_id), SCU64(result_right[vertex_id]),
                                   SCU64(transformVertexId), SCU64(result[transformVertexId]));
                    }
                    else
                    {
                        vertex_id_type transformVertexId = old2new[vertex_id];
                        assert_msg(result_right[vertex_id] == result[transformVertexId], "result_right[%zu] = %zu, result[%zu] = %zu",
                                   SCU64(vertex_id), SCU64(result_right[vertex_id]), SCU64(transformVertexId), SCU64(result[transformVertexId]));
                    }
                }
            }
            Msg_finish("Function [%s] finish check", __FUNCTION__);
        }
    }
};

template <typename Result_type>
class SaveResult
{
  public:
    SaveResult(const Result_type* result, const count_type& vertexNum, std::string graphName, Algorithm_type algorithm, int64_t root)
    {
        std::string path = "/data/webgraph/checkResult/" + std::string(Algorithm_type_name[SCI32(algorithm)]) + "/" + graphName + "_" +
                           std::string(Algorithm_type_name[SCI32(algorithm)]) + "_" + std::to_string(root) + ".bin";

        CPJ::Timer timer;
        CPJ::IOAdaptor ioAdapter(path);
        ioAdapter.openFile("w");
        ioAdapter.writeBinFile_sync(result, vertexNum * sizeof(Result_type), omp_get_max_threads());
        ioAdapter.closeFile();
        Msg_info("Wrire the result to file[%s] finished, Used time: %s", path.c_str(), timer.get_time_str().c_str());
    }
};
} // namespace CPJ