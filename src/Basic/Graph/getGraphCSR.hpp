#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Graph/compute_degree.hpp"
#include "Basic/Graph/sortNbr.hpp"
#include "Basic/IO/io_common.hpp"
#include "flag.hpp"
#include <cstddef>

namespace CPJ {

vertex_id_type* getGraphCSR(CSR_Result_type& csrResult, GraphFile_type& graphFile, Algorithm_type algorithm, OrderMethod_type orderMethod,
                            bool isSortNbr = true, bool getOutDegree = true, bool getInDegree = true,
                            GraphRepresentation graphRep = GraphRepresentation::CSR)
{
    size_t vertices = graphFile.vertices;
    size_t edges = graphFile.edges;
    std::string rank_file = graphFile.rankFile;
    std::string old2new_file = graphFile.old2newFile;
    std::string csrOffsetFile = graphFile.csrOffsetFile;
    std::string csrDestFile = graphFile.csrDestFile;
    std::string csrWeightFile = graphFile.csrWeightFile;
    Msg_info("GraphName:[%s], |V| = %zu, |E| = %zu", fLS::FLAGS_graphName.c_str(), static_cast<uint64_t>(vertices), static_cast<uint64_t>(edges));
    assert_msg(vertices < std::numeric_limits<count_type>::max(), "vertexNum need set the <count_type> to uint64_t");
    assert_msg(edges < std::numeric_limits<countl_type>::max(), "edgeNum need set the <countl_type> to uint64_t");

    csrResult.csr_offset = CPJ::load_binFile_cStyle<countl_type>(csrOffsetFile, static_cast<uint64_t>(vertices + 1));
    csrResult.csr_dest = CPJ::load_binFile_cStyle<vertex_id_type>(csrDestFile, static_cast<uint64_t>(edges));
    if (algorithm == Algorithm_type::SSSP || orderMethod == OrderMethod_type::NATIVE)
    {
        csrResult.csr_weight = load_binFile_cStyle<edge_data_type>(csrWeightFile, static_cast<uint64_t>(edges));
    }

    csrResult.vertexNum = vertices;
    csrResult.edgeNum = edges;
    Msg_info("Using CSRFile Construct csrResult complete");

    vertex_id_type* old2new{nullptr};
    if (orderMethod == OrderMethod_type::CGgraphRV1_5)
    {
        old2new = load_binFile_cStyle<vertex_id_type>(old2new_file, vertices);
    }

    if (isSortNbr)
    {
        if (algorithm == Algorithm_type::SSSP || orderMethod == OrderMethod_type::NATIVE)
        {
            CPJ::sortNbr(csrResult, true, false);
        }
        else
        {
            CPJ::sortNbr_noWeight(csrResult, true, false);
        }
    }

    if (getOutDegree || getInDegree)
    {
        Compute_degree* computeDegree = new Compute_degree(csrResult);
        if (getOutDegree) csrResult.outDegree = computeDegree->getOutdegree();
        if (getInDegree) csrResult.inDegree = computeDegree->getIndegree();
    }

    return old2new;
}
} // namespace CPJ