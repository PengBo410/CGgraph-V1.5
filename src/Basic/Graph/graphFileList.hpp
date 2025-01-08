#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Graph/getGraphCSR.hpp"
#include "Basic/Other/fileSystem_CPJ.hpp"
#include "Basic/Type/data_type.hpp"
#include "flag.hpp"
#include "reorderGraph.hpp"
#include <assert.h>
#include <string>

namespace CPJ {

/* *********************************************************************************************************
 * @description: 按需修改图数据路径
 * @param [string&] graphName
 * @return [*]
 * *********************************************************************************************************/
vertex_id_type* getGraphData(CSR_Result_type& csrResult, std::string graphName, bool isSortNbr, bool getOutDegree, bool getInDegree)
{
    vertex_id_type* old2new{nullptr};
    GraphFile_type graphFile;

    if (graphName == "twitter2010")
    {
        graphFile.vertices = 61578415;
        graphFile.edges = 1468364884;

        graphFile.csrOffsetFile = "/data/webgraph/bin/twitter2010/native_csrOffset_u32.bin";
        graphFile.csrDestFile = "/data/webgraph/bin/twitter2010/native_csrDest_u32.bin";
        graphFile.csrWeightFile = "/data/webgraph/bin/twitter2010/native_csrWeight_u32.bin";

    } // end of [twitter2010]

    else if (graphName == "friendster")
    {
        graphFile.vertices = 124836180;
        graphFile.edges = 1806067135;

        graphFile.csrOffsetFile = "/data/webgraph/bin/friendster/native_csrOffset_u32.bin";
        graphFile.csrDestFile = "/data/webgraph/bin/friendster/native_csrDest_u32.bin";
        graphFile.csrWeightFile = "/data/webgraph/bin/friendster/native_csrWeight_u32.bin";
    }
    else if (graphName == "uk-union")
    {
        graphFile.vertices = 133633040;
        graphFile.edges = 5475109924;

        graphFile.csrOffsetFile = "/data/webgraph/bin/uk-union/native_csrOffset_u64.bin";
        graphFile.csrDestFile = "/data/webgraph/bin/uk-union/native_csrDest_u32.bin";
        graphFile.csrWeightFile = "/data/webgraph/bin/uk-union/native_csrWeight_u32.bin";
    }
    else
    {
        assert_msg(false, "Unknow graphName [%s]", graphName.c_str());
    }

    assert_msg(graphFile.vertices < std::numeric_limits<count_type>::max(),
               "Total vertices need set the <count_type> and <vertex_id_type> to uint64_t");
    assert_msg(graphFile.edges < std::numeric_limits<countl_type>::max(), "Total edges need set the <countl_type> to uint64_t");
    if (graphFile.edges < std::numeric_limits<uint32_t>::max())
    {
        bool isSame = std::is_same_v<countl_type, uint32_t>;
        assert_msg(isSame, "Total edges can be stored by uint32_t, So set the <countl_type> to uint32_t");
    }

    if (CPJ::FS::isExist(getCGgraph_reorder_rankFile(graphName)))
    {
        graphFile.rankFile = getCGgraph_reorder_rankFile(graphName);
        graphFile.old2newFile = getCGgraph_reorder_old2newFile(graphName);
        graphFile.csrOffsetFile = getCGgraph_reorder_csrOffset(graphName);
        graphFile.csrDestFile = getCGgraph_reorder_csrDest(graphName);
        graphFile.csrWeightFile = getCGgraph_reorder_csrWeight(graphName);

        old2new = CPJ::getGraphCSR(csrResult, graphFile, static_cast<Algorithm_type>(SCI32(FLAGS_algorithm)), OrderMethod_type::CGgraphRV1_5,
                                   isSortNbr, getOutDegree, getInDegree);
    }
    else
    {
        CSR_Result_type csrResult_native;
        CPJ::getGraphCSR(csrResult_native, graphFile, static_cast<Algorithm_type>(SCI32(FLAGS_algorithm)), OrderMethod_type::NATIVE, isSortNbr, true,
                         getInDegree);
        CPJ::ReorderGraph reorderGraph(csrResult_native);
        csrResult = reorderGraph.doReorder(graphName);
        reorderGraph.freeOldCSR();
        old2new = reorderGraph.getOld2New();
    }

    return old2new;
}

} // namespace CPJ