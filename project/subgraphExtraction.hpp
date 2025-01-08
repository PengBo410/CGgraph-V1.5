#pragma once

#include "Basic/CUDA/cuda_check.cuh"
#include "Basic/CUDA/gpu_util.cuh"
#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Graph/graphFileList.hpp"
#include "Basic/Memory/memInfo_CPJ.hpp"
#include "Basic/Other/scan_CPJ.hpp"
#include "Basic/Thread/omp_def.hpp"
#include "Basic/Timer/timer_CPJ.hpp"
#include "Basic/Type/data_type.hpp"
#include "flag.hpp"
#include <algorithm>
#include <cstddef>
#include <type_traits>

namespace CPJ {

class SubgraphExt
{
  private:
    count_type vertexNum_{0};
    countl_type edgeNum_{0};
    countl_type* csr_offset_{nullptr};
    vertex_id_type* csr_dest_{nullptr};
    edge_data_type* csr_weight_{nullptr};
    degree_type* outDegree_{nullptr};

    static constexpr bool DEBUG_TIMER = true;

    const Algorithm_type algorithm_{Algorithm_type::MAX_VALUE};
    const int useDeviceId_{0};

    size_t dataMem_device_{0};
    size_t assistMem_device_{0};

  public:
    size_t availableMem_device_{0};

  private:
    const size_t bitset_size_{0};
    static constexpr size_t RESERVED_MEM = MB(128);
    static constexpr double RADIO = 0.56;

    static constexpr bool SSSP_DEST_WEIGHT = true;

  public:
    SubgraphExt(CSR_Result_type& csrResult, const Algorithm_type algorithm, const int useDeviceId = 0)
        : vertexNum_(csrResult.vertexNum), edgeNum_(csrResult.edgeNum), csr_offset_(csrResult.csr_offset), csr_dest_(csrResult.csr_dest),
          algorithm_(algorithm), useDeviceId_(useDeviceId), bitset_size_(sizeof(int) * (vertexNum_ + 32 - 1) / 32)
    {
        CUDA_CHECK(cudaSetDevice(useDeviceId));
        availableMem_device_ = CPJ::MemoryInfo::getMemoryAvailable_Device(useDeviceId_);
        dataMem_device_ = getDataMem();
        assistMem_device_ = getAssistMem(vertexNum_);
        Msg_info("The GPU[%d]:  availableMem = %.2lf (GB), dataMem = %.2lf (GB), assistMem = %.2lf (GB)", useDeviceId_,
                 BYTES_TO_GB(availableMem_device_), BYTES_TO_GB(dataMem_device_), BYTES_TO_GB(assistMem_device_));
        if (algorithm_ == Algorithm_type::SSSP)
        {
            assert_msg(csrResult.csr_weight != nullptr, "Algorithm_type::SSSP needs [csr_weight]");
            csr_weight_ = csrResult.csr_weight;
        }
    }

    AUTO_GPUMEM_type getGPUMemType()
    {
        int64_t remainFor_data = availableMem_device_ - assistMem_device_;
        assert_msg(remainFor_data > 0, "The remaining GPU memory for data too small, remainFor_data = %zd", remainFor_data);
        AUTO_GPUMEM_type auto_GPUMem;

        double capacityRadio = SCD(remainFor_data) / SCD(dataMem_device_);
        if (capacityRadio >= 1.0)
        {
            auto_GPUMem = AUTO_GPUMEM_type::FULL_DEVICE_MEM;
            Msg_info("The remaining GPU memory for data is sufficient, finally there is still %.2lf (GB) GPU memory, used [%s]",
                     BYTES_TO_GB(remainFor_data - dataMem_device_), AUTO_GPUMEM_type_name[SCI32(auto_GPUMem)]);
        }
        else if (capacityRadio < 1.0 && capacityRadio >= RADIO)
        {
            auto_GPUMem = AUTO_GPUMEM_type::PARTIAL_DEVICE_MEM;
            Msg_info("The remaining GPU memory for data is not sufficient,  %.2lf%% of the data is stored in GPU memory, used [%s]",
                     capacityRadio * 100, AUTO_GPUMEM_type_name[SCI32(auto_GPUMem)]);
        }
        else if (capacityRadio < RADIO)
        {
            auto_GPUMem = AUTO_GPUMEM_type::DISABLE_DEVICE_MEM;
            Msg_info("The remaining GPU memory for data is too small, only %.2lf%% of the data can be stored in GPU memory, used [%s]",
                     capacityRadio * 100, AUTO_GPUMEM_type_name[SCI32(auto_GPUMem)]);
        }
        else
        {
            auto_GPUMem = AUTO_GPUMEM_type::MAX_VALUE;
            assert_msg(false, "Error AUTO_GPUMEM_type in Func[%s]", __FUNCTION__);
        }

        return auto_GPUMem;
    }

    AUTO_GPUMEM_type extractSubgraph(Host_dataPointer_type& hostData, Device_dataPointer_type& deviceData, AUTO_GPUMEM_type gpyMemType)
    {
        CPJ::Timer timer;

        if (gpyMemType == AUTO_GPUMEM_type::FULL_DEVICE_MEM)
        {
            hostData.vertexNum_host_ = vertexNum_;
            hostData.edgeNum_host_ = edgeNum_;
            hostData.csr_offset_host_ = csr_offset_;
            hostData.csr_dest_host_ = csr_dest_;
            if (algorithm_ == Algorithm_type::SSSP) hostData.csr_weight_host_ = csr_weight_;

            deviceData.disableGPU = false;
            deviceData.isEntireGraph = true;
            deviceData.vertexNum_device_ = vertexNum_;
            deviceData.edgeNum_device_ = edgeNum_;
            deviceData.cutVertexId_device_ = vertexNum_ - 1;
            timer.start();
            CUDA_CHECK(MALLOC_DEVICE(&deviceData.csr_offset_device_, deviceData.vertexNum_device_ + 1));
            CUDA_CHECK(H2D(deviceData.csr_offset_device_, hostData.csr_offset_host_, deviceData.vertexNum_device_ + 1));
            if (algorithm_ == Algorithm_type::SSSP)
            {
                if constexpr (SSSP_DEST_WEIGHT && std::is_same_v<vertex_id_type, edge_data_type>)
                {
                    CUDA_CHECK(MALLOC_HOST(&hostData.csr_destWeight_host_, hostData.edgeNum_host_ * 2));
                    memset(hostData.csr_destWeight_host_, 0, sizeof(vertex_id_type) * hostData.edgeNum_host_ * 2);
                    omp_par_for(countl_type edge_id = 0; edge_id < hostData.edgeNum_host_; edge_id++)
                    {
                        hostData.csr_destWeight_host_[edge_id * 2] = hostData.csr_dest_host_[edge_id];
                        hostData.csr_destWeight_host_[edge_id * 2 + 1] = hostData.csr_weight_host_[edge_id];
                    }
                    delete[] hostData.csr_dest_host_;
                    hostData.csr_dest_host_ = nullptr;
                    delete[] hostData.csr_weight_host_;
                    hostData.csr_weight_host_ = nullptr;

                    CUDA_CHECK(MALLOC_DEVICE(&deviceData.csr_destWeight_device_, deviceData.edgeNum_device_ * 2));
                    CUDA_CHECK(H2D(deviceData.csr_destWeight_device_, hostData.csr_destWeight_host_, deviceData.edgeNum_device_ * 2));

                    assert_msg(deviceData.csr_destWeight_device_ != nullptr,
                               "When [SSSP_DEST_WEIGHT] on, deviceData.csr_destWeight_device_ need set to not nullptr");
                    assert_msg(hostData.csr_dest_host_ == nullptr, "When [SSSP_DEST_WEIGHT] on, hostData.csr_dest_host_ need set to nullptr");
                    assert_msg(hostData.csr_weight_host_ == nullptr, "When [SSSP_DEST_WEIGHT] on, hostData.csr_weight_host_ need set to nullptr");

                    hostData.SSSP_dest_weight = true;
                    deviceData.SSSP_dest_weight = true;
                }
                else
                {
                    CUDA_CHECK(MALLOC_DEVICE(&deviceData.csr_dest_device_, deviceData.edgeNum_device_));
                    CUDA_CHECK(H2D(deviceData.csr_dest_device_, hostData.csr_dest_host_, deviceData.edgeNum_device_));
                    CUDA_CHECK(MALLOC_DEVICE(&deviceData.csr_weight_device_, deviceData.edgeNum_device_));
                    CUDA_CHECK(H2D(deviceData.csr_weight_device_, hostData.csr_weight_host_, deviceData.edgeNum_device_));
                    assert_msg(deviceData.csr_destWeight_device_ == nullptr,
                               "When [SSSP_DEST_WEIGHT] off, deviceData.csr_destWeight_device_ need set to nullptr");
                    assert_msg(deviceData.csr_dest_device_ != nullptr,
                               "When [SSSP_DEST_WEIGHT] off, deviceData.csr_dest_device_ need set to not nullptr");
                    assert_msg(deviceData.csr_weight_device_ != nullptr,
                               "When [SSSP_DEST_WEIGHT] off, deviceData.csr_weight_device_ need set to not nullptr");
                }
            }
            else
            {
                CUDA_CHECK(MALLOC_DEVICE(&deviceData.csr_dest_device_, deviceData.edgeNum_device_));
                CUDA_CHECK(H2D(deviceData.csr_dest_device_, hostData.csr_dest_host_, deviceData.edgeNum_device_));
            }
            Msg_info("In [%s], GPU raw data ready, Used time: %s", AUTO_GPUMEM_type_name[SCI32(gpyMemType)], timer.get_time_str().c_str());
        }

        else if (gpyMemType == AUTO_GPUMEM_type::PARTIAL_DEVICE_MEM)
        {
            hostData.vertexNum_host_ = vertexNum_;
            hostData.edgeNum_host_ = edgeNum_;
            hostData.csr_offset_host_ = csr_offset_;
            hostData.csr_dest_host_ = csr_dest_;
            if (algorithm_ == Algorithm_type::SSSP) hostData.csr_weight_host_ = csr_weight_;

            deviceData.disableGPU = false;
            size_t factDataSize = 0;
            do
            {
                int64_t remainFor_data = availableMem_device_ - getAssistMem(vertexNum_);
                remainFor_data -= (deviceData.vertexNum_device_ * sizeof(countl_type));
                size_t edgeSize = (algorithm_ == Algorithm_type::SSSP) ? (sizeof(vertex_id_type) + sizeof(edge_data_type)) : sizeof(vertex_id_type);
                size_t edgeNum_GPU = remainFor_data / edgeSize;
                Msg_info("Current GPU maybe can store (%zu) edges for [%s], GPUMemorySize = %.2lf (GB), edgeSize = %zu", edgeNum_GPU,
                         Algorithm_type_name[SCI32(algorithm_)], BYTES_TO_GB(remainFor_data), edgeSize);
                int64_t upperBound =
                    std::distance(hostData.csr_offset_host_,
                                  std::upper_bound(hostData.csr_offset_host_, hostData.csr_offset_host_ + hostData.vertexNum_host_ + 1, edgeNum_GPU));
                assert_msg(upperBound >= 2, "a....");
                deviceData.isEntireGraph = false;
                deviceData.vertexNum_device_ = upperBound - 1;
                deviceData.cutVertexId_device_ = deviceData.vertexNum_device_ - 1;
                deviceData.edgeNum_device_ = hostData.csr_offset_host_[deviceData.vertexNum_device_];

                factDataSize = edgeSize * deviceData.edgeNum_device_ + sizeof(countl_type) * (deviceData.vertexNum_device_ + 1);
                factDataSize += getAssistMem(vertexNum_);
            }
            while (factDataSize >= availableMem_device_);
            Msg_info("In fact, Current GPU  store (%zu) vertices, (%zu) edges for [%s], occupy (%.2lf)%% GPU memory",
                     SCU64(deviceData.vertexNum_device_), SCU64(deviceData.edgeNum_device_), AUTO_GPUMEM_type_name[SCI32(gpyMemType)],
                     SCD(factDataSize) / SCD(availableMem_device_) * 100);

            timer.start();
            CUDA_CHECK(MALLOC_DEVICE(&deviceData.csr_offset_device_, deviceData.vertexNum_device_ + 1));
            CUDA_CHECK(H2D(deviceData.csr_offset_device_, hostData.csr_offset_host_, deviceData.vertexNum_device_ + 1));
            if (algorithm_ == Algorithm_type::SSSP)
            {
                if constexpr (SSSP_DEST_WEIGHT && std::is_same_v<vertex_id_type, edge_data_type>)
                {
                    assert_msg((hostData.edgeNum_host_ * 2) < std::numeric_limits<countl_type>::max(),
                               "(edgeNum_ * 2) large than [countl_type] max()");
                    CUDA_CHECK(MALLOC_HOST(&hostData.csr_destWeight_host_, hostData.edgeNum_host_ * 2));
                    memset(hostData.csr_destWeight_host_, 0, sizeof(vertex_id_type) * hostData.edgeNum_host_ * 2);
                    omp_par_for(countl_type edge_id = 0; edge_id < hostData.edgeNum_host_; edge_id++)
                    {
                        hostData.csr_destWeight_host_[edge_id * 2] = hostData.csr_dest_host_[edge_id];
                        hostData.csr_destWeight_host_[edge_id * 2 + 1] = hostData.csr_weight_host_[edge_id];
                    }
                    delete[] hostData.csr_dest_host_;
                    hostData.csr_dest_host_ = nullptr;
                    delete[] hostData.csr_weight_host_;
                    hostData.csr_weight_host_ = nullptr;

                    CUDA_CHECK(MALLOC_DEVICE(&deviceData.csr_destWeight_device_, deviceData.edgeNum_device_ * 2));
                    CUDA_CHECK(H2D(deviceData.csr_destWeight_device_, hostData.csr_destWeight_host_, deviceData.edgeNum_device_ * 2));

                    assert_msg(deviceData.csr_destWeight_device_ != nullptr,
                               "When [SSSP_DEST_WEIGHT] on, deviceData.csr_destWeight_device_ need set to not nullptr");
                    assert_msg(hostData.csr_dest_host_ == nullptr, "When [SSSP_DEST_WEIGHT] on, hostData.csr_dest_host_ need set to nullptr");
                    assert_msg(hostData.csr_weight_host_ == nullptr, "When [SSSP_DEST_WEIGHT] on, hostData.csr_weight_host_ need set to nullptr");

                    hostData.SSSP_dest_weight = true;
                    deviceData.SSSP_dest_weight = true;
                }
                else
                {
                    CUDA_CHECK(MALLOC_DEVICE(&deviceData.csr_dest_device_, deviceData.edgeNum_device_));
                    CUDA_CHECK(H2D(deviceData.csr_dest_device_, hostData.csr_dest_host_, deviceData.edgeNum_device_));
                    CUDA_CHECK(MALLOC_DEVICE(&deviceData.csr_weight_device_, deviceData.edgeNum_device_));
                    CUDA_CHECK(H2D(deviceData.csr_weight_device_, hostData.csr_weight_host_, deviceData.edgeNum_device_));
                    assert_msg(deviceData.csr_destWeight_device_ == nullptr,
                               "When [SSSP_DEST_WEIGHT] off, deviceData.csr_destWeight_device_ need set to nullptr");
                    assert_msg(deviceData.csr_dest_device_ != nullptr,
                               "When [SSSP_DEST_WEIGHT] off, deviceData.csr_dest_device_ need set to not nullptr");
                    assert_msg(deviceData.csr_weight_device_ != nullptr,
                               "When [SSSP_DEST_WEIGHT] off, deviceData.csr_weight_device_ need set to not nullptr");
                }
            }
            else
            {
                CUDA_CHECK(MALLOC_DEVICE(&deviceData.csr_dest_device_, deviceData.edgeNum_device_));
                CUDA_CHECK(H2D(deviceData.csr_dest_device_, hostData.csr_dest_host_, deviceData.edgeNum_device_));
            }
            Msg_info("In [%s], GPU raw data ready, Used time: %s", AUTO_GPUMEM_type_name[SCI32(gpyMemType)], timer.get_time_str().c_str());
        }

        else if ((gpyMemType == AUTO_GPUMEM_type::DISABLE_DEVICE_MEM) &&
                 (static_cast<GPU_memory_type>(SCI32(FLAGS_gpuMemory)) == GPU_memory_type::UVM))
        {
            hostData.vertexNum_host_ = vertexNum_;
            hostData.edgeNum_host_ = edgeNum_;
            hostData.csr_offset_host_ = csr_offset_;
            hostData.csr_dest_host_ = csr_dest_;
            if (algorithm_ == Algorithm_type::SSSP) hostData.csr_weight_host_ = csr_weight_;

            deviceData.disableGPU = false;
            deviceData.isEntireGraph = true;
            deviceData.vertexNum_device_ = vertexNum_;
            deviceData.edgeNum_device_ = edgeNum_;
            deviceData.cutVertexId_device_ = vertexNum_ - 1;
            timer.start();
            CUDA_CHECK(MALLOC_DEVICE(&deviceData.csr_offset_device_, deviceData.vertexNum_device_ + 1));
            CUDA_CHECK(H2D(deviceData.csr_offset_device_, hostData.csr_offset_host_, deviceData.vertexNum_device_ + 1));
            if (algorithm_ == Algorithm_type::SSSP)
            {
                if constexpr (SSSP_DEST_WEIGHT && std::is_same_v<vertex_id_type, edge_data_type>)
                {
                    CUDA_CHECK(cudaMallocManaged((void**)&hostData.csr_destWeight_host_, (hostData.edgeNum_host_ * 2) * sizeof(vertex_id_type)));
                    CUDA_CHECK(cudaMemAdvise(hostData.csr_destWeight_host_, (hostData.edgeNum_host_ * 2) * sizeof(vertex_id_type),
                                             cudaMemAdviseSetReadMostly, useDeviceId_));
                    omp_par_for(countl_type edge_id = 0; edge_id < hostData.edgeNum_host_; edge_id++)
                    {
                        hostData.csr_destWeight_host_[edge_id * 2] = hostData.csr_dest_host_[edge_id];
                        hostData.csr_destWeight_host_[edge_id * 2 + 1] = hostData.csr_weight_host_[edge_id];
                    }
                    delete[] hostData.csr_dest_host_;
                    hostData.csr_dest_host_ = nullptr;
                    delete[] hostData.csr_weight_host_;
                    hostData.csr_weight_host_ = nullptr;

                    deviceData.csr_destWeight_device_ = hostData.csr_destWeight_host_; // Host和Device一起使用
                    assert_msg(deviceData.csr_destWeight_device_ != nullptr,
                               "When [SSSP_DEST_WEIGHT] on, deviceData.csr_destWeight_device_ need set to not nullptr");
                    assert_msg(hostData.csr_dest_host_ == nullptr, "When [SSSP_DEST_WEIGHT] on, hostData.csr_dest_host_ need set to nullptr");
                    assert_msg(hostData.csr_weight_host_ == nullptr, "When [SSSP_DEST_WEIGHT] on, hostData.csr_weight_host_ need set to nullptr");

                    hostData.SSSP_dest_weight = true;
                    deviceData.SSSP_dest_weight = true;
                }
                else
                {
                    CUDA_CHECK(cudaMallocManaged((void**)&deviceData.csr_dest_device_, deviceData.edgeNum_device_ * sizeof(vertex_id_type)));
                    CUDA_CHECK(cudaMemAdvise(deviceData.csr_dest_device_, deviceData.edgeNum_device_ * sizeof(vertex_id_type),
                                             cudaMemAdviseSetReadMostly, useDeviceId_));
                    std::memcpy(deviceData.csr_dest_device_, hostData.csr_dest_host_, sizeof(vertex_id_type) * deviceData.edgeNum_device_);
                    delete[] hostData.csr_dest_host_;
                    hostData.csr_dest_host_ = deviceData.csr_dest_device_;

                    CUDA_CHECK(cudaMallocManaged((void**)&deviceData.csr_weight_device_, deviceData.edgeNum_device_ * sizeof(edge_data_type)));
                    CUDA_CHECK(cudaMemAdvise(deviceData.csr_weight_device_, deviceData.edgeNum_device_ * sizeof(edge_data_type),
                                             cudaMemAdviseSetReadMostly, useDeviceId_));
                    std::memcpy(deviceData.csr_weight_device_, hostData.csr_weight_host_, sizeof(edge_data_type) * deviceData.edgeNum_device_);
                    delete[] hostData.csr_weight_host_;
                    hostData.csr_weight_host_ = deviceData.csr_weight_device_;

                    assert_msg(deviceData.csr_destWeight_device_ == nullptr,
                               "When [SSSP_DEST_WEIGHT] off, deviceData.csr_destWeight_device_ need set to nullptr");
                    assert_msg(deviceData.csr_dest_device_ != nullptr,
                               "When [SSSP_DEST_WEIGHT] off, deviceData.csr_dest_device_ need set to not nullptr");
                    assert_msg(deviceData.csr_weight_device_ != nullptr,
                               "When [SSSP_DEST_WEIGHT] off, deviceData.csr_weight_device_ need set to not nullptr");
                }
            }
            else
            {
                CUDA_CHECK(cudaMallocManaged((void**)&deviceData.csr_dest_device_, deviceData.edgeNum_device_ * sizeof(vertex_id_type)));
                CUDA_CHECK(cudaMemAdvise(deviceData.csr_dest_device_, deviceData.edgeNum_device_ * sizeof(vertex_id_type), cudaMemAdviseSetReadMostly,
                                         useDeviceId_));
                std::memcpy(deviceData.csr_dest_device_, hostData.csr_dest_host_, sizeof(vertex_id_type) * deviceData.edgeNum_device_);
                delete[] hostData.csr_dest_host_;
                hostData.csr_dest_host_ = deviceData.csr_dest_device_;
            }
            Msg_info("In [%s], GPU raw data ready, Used time: %s", AUTO_GPUMEM_type_name[SCI32(gpyMemType)], timer.get_time_str().c_str());
        }

        else if ((gpyMemType == AUTO_GPUMEM_type::DISABLE_DEVICE_MEM) &&
                 ((static_cast<GPU_memory_type>(SCI32(FLAGS_gpuMemory)) == GPU_memory_type::ZERO_COPY) ||
                  (static_cast<GPU_memory_type>(SCI32(FLAGS_gpuMemory)) == GPU_memory_type::GPU_MEM)))
        {
            hostData.vertexNum_host_ = vertexNum_;
            hostData.edgeNum_host_ = edgeNum_;
            hostData.csr_offset_host_ = csr_offset_;
            hostData.csr_dest_host_ = csr_dest_;
            if (algorithm_ == Algorithm_type::SSSP) hostData.csr_weight_host_ = csr_weight_;

            deviceData.disableGPU = false;
            deviceData.isEntireGraph = true;
            deviceData.vertexNum_device_ = vertexNum_;
            deviceData.edgeNum_device_ = edgeNum_;
            deviceData.cutVertexId_device_ = vertexNum_ - 1;
            timer.start();
            CUDA_CHECK(MALLOC_DEVICE(&deviceData.csr_offset_device_, deviceData.vertexNum_device_ + 1));
            CUDA_CHECK(H2D(deviceData.csr_offset_device_, hostData.csr_offset_host_, deviceData.vertexNum_device_ + 1));
            if (algorithm_ == Algorithm_type::SSSP)
            {
                if constexpr (SSSP_DEST_WEIGHT && std::is_same_v<vertex_id_type, edge_data_type>)
                {
                    CUDA_CHECK(cudaMallocManaged((void**)&hostData.csr_destWeight_host_, (hostData.edgeNum_host_ * 2) * sizeof(vertex_id_type)));
                    CUDA_CHECK(cudaMemAdvise(hostData.csr_destWeight_host_, (hostData.edgeNum_host_ * 2) * sizeof(vertex_id_type),
                                             cudaMemAdviseSetAccessedBy, useDeviceId_));
                    omp_par_for(countl_type edge_id = 0; edge_id < hostData.edgeNum_host_; edge_id++)
                    {
                        hostData.csr_destWeight_host_[edge_id * 2] = hostData.csr_dest_host_[edge_id];
                        hostData.csr_destWeight_host_[edge_id * 2 + 1] = hostData.csr_weight_host_[edge_id];
                    }
                    delete[] hostData.csr_dest_host_;
                    hostData.csr_dest_host_ = nullptr;
                    delete[] hostData.csr_weight_host_;
                    hostData.csr_weight_host_ = nullptr;

                    deviceData.csr_destWeight_device_ = hostData.csr_destWeight_host_; // Host和Device一起使用
                    assert_msg(deviceData.csr_destWeight_device_ != nullptr,
                               "When [SSSP_DEST_WEIGHT] on, deviceData.csr_destWeight_device_ need set to not nullptr");
                    assert_msg(hostData.csr_dest_host_ == nullptr, "When [SSSP_DEST_WEIGHT] on, hostData.csr_dest_host_ need set to nullptr");
                    assert_msg(hostData.csr_weight_host_ == nullptr, "When [SSSP_DEST_WEIGHT] on, hostData.csr_weight_host_ need set to nullptr");

                    hostData.SSSP_dest_weight = true;
                    deviceData.SSSP_dest_weight = true;
                }
                else
                {
                    CUDA_CHECK(cudaMallocManaged((void**)&deviceData.csr_dest_device_, deviceData.edgeNum_device_ * sizeof(vertex_id_type)));
                    CUDA_CHECK(cudaMemAdvise(deviceData.csr_dest_device_, deviceData.edgeNum_device_ * sizeof(vertex_id_type),
                                             cudaMemAdviseSetAccessedBy, useDeviceId_));
                    std::memcpy(deviceData.csr_dest_device_, hostData.csr_dest_host_, sizeof(vertex_id_type) * deviceData.edgeNum_device_);
                    delete[] hostData.csr_dest_host_;
                    hostData.csr_dest_host_ = deviceData.csr_dest_device_;

                    CUDA_CHECK(cudaMallocManaged((void**)&deviceData.csr_weight_device_, deviceData.edgeNum_device_ * sizeof(edge_data_type)));
                    CUDA_CHECK(cudaMemAdvise(deviceData.csr_weight_device_, deviceData.edgeNum_device_ * sizeof(edge_data_type),
                                             cudaMemAdviseSetAccessedBy, useDeviceId_));
                    std::memcpy(deviceData.csr_weight_device_, hostData.csr_weight_host_, sizeof(edge_data_type) * deviceData.edgeNum_device_);
                    delete[] hostData.csr_weight_host_;
                    hostData.csr_weight_host_ = deviceData.csr_weight_device_;

                    assert_msg(deviceData.csr_destWeight_device_ == nullptr,
                               "When [SSSP_DEST_WEIGHT] off, deviceData.csr_destWeight_device_ need set to nullptr");
                    assert_msg(deviceData.csr_dest_device_ != nullptr,
                               "When [SSSP_DEST_WEIGHT] off, deviceData.csr_dest_device_ need set to not nullptr");
                    assert_msg(deviceData.csr_weight_device_ != nullptr,
                               "When [SSSP_DEST_WEIGHT] off, deviceData.csr_weight_device_ need set to not nullptr");
                }
            }
            else
            {
                CUDA_CHECK(cudaMallocManaged((void**)&deviceData.csr_dest_device_, deviceData.edgeNum_device_ * sizeof(vertex_id_type)));
                CUDA_CHECK(cudaMemAdvise(deviceData.csr_dest_device_, deviceData.edgeNum_device_ * sizeof(vertex_id_type), cudaMemAdviseSetAccessedBy,
                                         useDeviceId_));
                std::memcpy(deviceData.csr_dest_device_, hostData.csr_dest_host_, sizeof(vertex_id_type) * deviceData.edgeNum_device_);
                delete[] hostData.csr_dest_host_;
                hostData.csr_dest_host_ = deviceData.csr_dest_device_;
            }
            Msg_info("In [%s], GPU raw data ready, Used time: %s", AUTO_GPUMEM_type_name[SCI32(gpyMemType)], timer.get_time_str().c_str());
        }

        else if (gpyMemType == AUTO_GPUMEM_type::DISABLE_GPU)
        {
            hostData.vertexNum_host_ = vertexNum_;
            hostData.edgeNum_host_ = edgeNum_;
            hostData.csr_offset_host_ = csr_offset_;
            hostData.csr_dest_host_ = csr_dest_;
            if (algorithm_ == Algorithm_type::SSSP) hostData.csr_weight_host_ = csr_weight_;

            deviceData.disableGPU = true;
            Msg_info("GPU is disabled");
        }

        else
        {
            assert_msg(false, "Error AUTO_GPUMEM_type[%s] in Func[%s]", AUTO_GPUMEM_type_name[SCI32(gpyMemType)], __FUNCTION__);
        }

        return gpyMemType;
    }

  private:
    size_t getDataMem()
    {
        size_t dataMem = 0;

        switch (algorithm_)
        {
        case Algorithm_type::BFS:
            dataMem += (sizeof(countl_type) * (vertexNum_ + 1));
            dataMem += (sizeof(vertex_id_type) * edgeNum_);
            break;
        case Algorithm_type::SSSP:
            dataMem += (sizeof(countl_type) * (vertexNum_ + 1));
            dataMem += (sizeof(vertex_id_type) * edgeNum_);
            dataMem += (sizeof(edge_data_type) * edgeNum_);
            break;
        case Algorithm_type::MAX_VALUE:
            assert_msg(false, "Unknow algorithm in func [%s]", __FUNCTION__);
            break;
        }

        return dataMem;
    }

  public:
    size_t getAssistMem(const count_type vertexNum_device)
    {
        size_t assistMem = 0;

        switch (algorithm_)
        {
        case Algorithm_type::BFS:
            assistMem += (sizeof(vertex_id_type) * vertexNum_device * 2); // frontier_in_device_, frontier_out_device_
            assistMem += (sizeof(countl_type) * vertexNum_device);        // frontier_degExSum_device_
            assistMem += (sizeof(countl_type) * vertexNum_device);        // frontier_balance_device_
            assistMem += (sizeof(vertex_data_type) * vertexNum_);         // vertexValue_device_
            assistMem += (bitset_size_ * 2);                              // sinkBitset_device_, visitedBitset_device_
            break;
        case Algorithm_type::SSSP:
            assistMem += (sizeof(vertex_id_type) * vertexNum_device * 2); // frontier_in_device_, frontier_out_device_
            assistMem += (sizeof(countl_type) * vertexNum_device);        // frontier_degExSum_device_
            assistMem += (sizeof(countl_type) * vertexNum_device);        // frontier_balance_device_
            assistMem += (sizeof(vertex_data_type) * vertexNum_ * 2);     // vertexValue_device_ vertexValue_device2_
            assistMem += (bitset_size_ * 2);                              // sinkBitset_device_, visitedBitset_device_
            break;
        case Algorithm_type::MAX_VALUE:
            assert_msg(false, "Unknow algorithm in func [%s]", __FUNCTION__);
            break;
        }

        return assistMem + RESERVED_MEM;
    }
};

} // namespace CPJ