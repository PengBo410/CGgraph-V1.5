#pragma once

#include "Basic/Console/console_var.hpp"
#include <mpi.h>
#include <ostream>

class MPIEnv
{
  private:
    int machineId_{0};
    int machineNum_{0};
    int namelen_{0};
    char serverName_[MPI_MAX_PROCESSOR_NAME];

    static constexpr bool USED_MPI_THREAD = false;

  public:
    MPIEnv() : machineId_(0), machineNum_(0), namelen_(0) {}

    void mpi_env_init(int* argc, char** argv[])
    {
        if constexpr (USED_MPI_THREAD)
        {
            int provided{0};
            MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
        }
        else
        {
            MPI_Init(argc, argv);
        }

        MPI_Comm_rank(MPI_COMM_WORLD, &machineId_);
        MPI_Comm_size(MPI_COMM_WORLD, &machineNum_);
        MPI_Get_processor_name(serverName_, &namelen_);

        Console_Val::serverId = machineId_;
        Console_Val::serverNum = machineNum_;
        Console_Val::serverName = std::string(serverName_, namelen_);
    }

    void mpi_env_finalize()
    {
        int is_finalized = 0;
        MPI_Finalized(&is_finalized);
        if (is_finalized) return;

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
    }

    ~MPIEnv()
    {
        int is_finalized = 0;
        MPI_Finalized(&is_finalized);
        if (is_finalized) return;

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
    }
};

inline int serverId() { return Console_Val::serverId; }
inline int serverNum() { return Console_Val::serverNum; }
inline std::string serverName() { return Console_Val::serverName; }
inline void mpi_barrier() { MPI_Barrier(MPI_COMM_WORLD); }
inline void mpi_barrier_flush()
{
    std::fflush(stderr);
    MPI_Barrier(MPI_COMM_WORLD);
}
inline MPI_Comm get_mpi_comm() { return MPI_COMM_WORLD; }

const constexpr bool showMPIDataType = false;
template <typename T>
inline MPI_Datatype MPIDataType()
{
    if constexpr (std::is_same<T, char>::value)
    {
        if constexpr (showMPIDataType) printf("MPI_CHAR\n");
        return MPI_CHAR;
    }
    else if constexpr (std::is_same<T, unsigned char>::value)
    {
        if constexpr (showMPIDataType) printf("MPI_UNSIGNED_CHAR\n");
        return MPI_UNSIGNED_CHAR;
    }
    else if constexpr (std::is_same<T, int>::value)
    {
        if constexpr (showMPIDataType) printf("MPI_INT\n");
        return MPI_INT;
    }
    else if constexpr (std::is_same<T, unsigned>::value)
    {
        if constexpr (showMPIDataType) printf("MPI_UNSIGNED\n");
        return MPI_UNSIGNED;
    }
    else if constexpr (std::is_same<T, long>::value)
    {
        if constexpr (showMPIDataType) printf("MPI_LONG\n");
        return MPI_LONG;
    }
    else if constexpr (std::is_same<T, unsigned long>::value)
    {
        if constexpr (showMPIDataType) printf("MPI_UNSIGNED_LONG\n");
        return MPI_UNSIGNED_LONG;
    }
    else if constexpr (std::is_same<T, float>::value)
    {
        if constexpr (showMPIDataType) printf("MPI_FLOAT\n");
        return MPI_FLOAT;
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        if constexpr (showMPIDataType) printf("MPI_DOUBLE\n");
        return MPI_DOUBLE;
    }
    else
    {
        printf("type not supported\n");
        exit(-1);
    }
}