#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include <gflags/gflags.h>

//> gflags
DEFINE_string(graphName, "friendster", "The Graph Name");
DEFINE_int64(root, 25689, "The Root For BFS/SSSP Or MaxIte For PageRank");
DEFINE_int32(algorithm, SCI32(Algorithm_type::BFS), Algorithm_type_help);
DEFINE_int32(gpuMemory, SCI32(GPU_memory_type::GPU_MEM), GPU_memory_type_help);
DEFINE_int32(runs, 5, "The Number Of Times That The Algorithm Needs To Run");
DEFINE_int32(useDeviceId, 0, "The GPU ID To Be Used");