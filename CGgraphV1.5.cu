#include "Basic/Console/console_V3_3.hpp"
#include "Basic/IO/io_adapter_V1.hpp"
#include "Basic/Thread/omp_def.hpp"
#include "CGgraphV1.5.hpp"
#include "Comm/MPI/mpi_env.hpp"

#include <gflags/gflags.h>
#include <iostream>
#include <omp.h>

int main(int argc, char* argv[])
{
    //> gflags
    gflags::SetUsageMessage("Welcom To Access CGgraph-V1.5:");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    //> MPI - Begin
    MPIEnv mpiEnv;
    mpiEnv.mpi_env_init(&argc, &argv);

    //> LOG
    global_logFile().set_log_file();

    Run_CGgraph();

    //> MPI - End
    mpiEnv.mpi_env_finalize();
    Msg_info("Hello, from 2023-10-8-CPJ!");

    return 0;
}