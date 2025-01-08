#pragma once

#include "Basic/Bitmap/bitsOp_CPJ.hpp"
#include "Basic/CUDA/cuda_check.cuh"
#include "Basic/CUDA/gpu_util.cuh"
#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Memory/alloc_CPJ.hpp"
#include "Basic/Memory/memInfo_CPJ.hpp"
#include "Basic/Other/scan_CPJ.hpp"
#include "Basic/Thread/atomic_linux.hpp"
#include "Basic/Thread/omp_def.hpp"
#include "Basic/Timer/timer_CPJ.hpp"
#include "Basic/Type/data_type.hpp"
#include "CG_help.hpp"
#include "basic_def.hpp"
#include "favor.hpp"
#include "simple_threadPool.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iterator>
#include <limits>
#include <math.h>

#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <cub/device/device_radix_sort.cuh>
#include <type_traits>

namespace CPJ {

class CG_SSSP
{
  private:
    /* DATA */
    const count_type vertexNum_host_{0};
    const countl_type edgeNum_host_{0};
    const countl_type* csr_offset_host_{nullptr};
    const vertex_id_type* csr_dest_host_{nullptr};
    const edge_data_type* csr_weight_host_{nullptr};
    const vertex_id_type* csr_destWeight_host_{nullptr};

    const count_type vertexNum_device_{0};
    const countl_type edgeNum_device_{0};
    const vertex_id_type cutVertexId_device_{0};
    const countl_type* csr_offset_device_{nullptr};
    const vertex_id_type* csr_dest_device_{nullptr};
    const edge_data_type* csr_weight_device_{nullptr};
    const vertex_id_type* csr_destWeight_device_{nullptr};
    const AUTO_GPUMEM_type gpuMemType_;

    const bool isEntireGraph_device_{true};
    const bool disable_device_{false};
    const bool SSSP_destWeight_{false};

    /* Result */
    vertex_data_type* vertexValue_host_{nullptr};
    vertex_data_type* vertexValue_temp_host_{nullptr};
    vertex_data_type* vertexValue_device_{nullptr};
    vertex_data_type* vertexValue_host2_{nullptr};
    vertex_data_type* vertexValue_device2_{nullptr};

    /* Assist */
    count_type frontierNum_in_host_{0};
    std::atomic<count_type> frontierNum_out_host_{0};
    vertex_id_type* frontier_in_host_{nullptr};
    vertex_id_type* frontier_out_host_{nullptr};
    countl_type* frontier_degExSum_host_{nullptr};
    countl_type* frontier_degExSum_forPar_host_{nullptr};
    countl_type* frontier_balance_host_{nullptr};
    countl_type* thread_edges_host_{nullptr};
    vertex_id_type** frontierOut_thread_host_{nullptr};

    vertex_id_type* frontier_in_device_{nullptr};
    vertex_id_type* frontier_out_device_{nullptr};
    countl_type* frontier_degExSum_device_{nullptr};
    countl_type* frontier_balance_device_{nullptr};
    count_type* frontierNum_out_device_{nullptr};
    const bool is_sortFrontier_forDevice_{false};
    const int num_vertex_bits_forDevice_{0};
    void* cub_temp_device_{nullptr};
    size_t cub_alloc_size_{0};

    /* Thrust */
    thrust::device_ptr<count_type> frontierNum_out_thrust_{nullptr};
    thrust::device_ptr<countl_type> frontier_degExSum_thrust_{nullptr};
    thrust::device_ptr<countl_type> frontier_balance_thrust_{nullptr};

    /* Bitset */
    const count_type visitedBitsetNum_host_{0};
    std::atomic<uint32_t>* visitedBitset_host_{nullptr};
    uint32_t* visitedBitset_storeTempDevice_host_{nullptr};
    uint32_t* visitedBitset_curIte_host_{nullptr};
    uint32_t* visitedBitset_curIte_device_{nullptr};
    uint32_t* notSinkBitset_host_{nullptr};
    uint32_t* visitedBitset_device_{nullptr};
    uint32_t* visitedBitset_temp_device_{nullptr};
    uint32_t* notSinkBitset_device_{nullptr};

    /* Host Balance */
    const int ThreadNum_host_{0};
    Host::Balance_CG::ThreadState_type* threadState_host_{nullptr};
    Host::Balance_CG::ThreadState_forIncModel_type* threadState_forIncModel_host_{nullptr};
    count_type* frontierAppend_thread_host_{nullptr};
    const count_type frontierAppendNum_thread_host_{0};
    static constexpr count_type CUSTOM_CHUNK_NUM_FRONTIER_APPEND_EACH_THREAD = 6;
    const count_type chunkNum_frontierAppend_eachThread_host_{0};

    /* Debug or Constexpr */
    static constexpr bool ISSORT = true;
    static constexpr bool TEMP_TIME_DEBUG = false;
    static constexpr bool USED_STD_PAR = true;
    static constexpr int socketId_ = 0;
    static constexpr std::memory_order ATOMIC_ORDER = std::memory_order_relaxed;
    static constexpr bool TEMP_DETAIL_ACTIVES_DEBUG = true;
    static constexpr bool INC_ITE_DEBUG = false;
    static constexpr bool ONLY_CGTIME_DEBUG = false;
    bool USE_DEVICE_FOR_GET_FRONTIER = false;

    /* Processor Process Model */
    enum class CGModel { CPU_ONLY, GPU_ONLY, CPU_GPU, MAX_VALUE };
    const char* CGModel_name[SCU32(CGModel::MAX_VALUE)] = {"CPU_ONLY", "GPU_ONLY", "CPU_GPU"};
    CGModel CGModel_;
    CGModel CGModel_lastIte_{CGModel::CPU_ONLY};
    const double CG_ratio_{0.0};
    double final_ratio_{0.0};
    bool LOCK_GPU_ONLY_{false};
    CPJ::ThreadPool* agent_device{nullptr};
    count_type alreadyEdgeNum_BFS_{0};
    static constexpr countl_type TH_CPU_ONLY_EDGENUM_v1 = 1000000;
    static constexpr countl_type TH_CPU_GPU_EDGENUM_v1 = 200000000;
    degree_type noSink_avgDegree_{0};
    std::mutex co_mutex_;
    std::condition_variable co_cv_;
    bool co_canCPUrun_ = false;

    /* Algorithm */
    int ite_{0};
    size_t nBlock_{0};

    /* Timer */
    CPJ::Timer globalTimer_;

    /* Stream */
    cudaStream_t stream;

  public:
    CG_SSSP(Host_dataPointer_type& dataPtr_host, Device_dataPointer_type& dataPtr_device, AUTO_GPUMEM_type gpuMemType, double CG_ratio)
        : vertexNum_host_(dataPtr_host.vertexNum_host_), edgeNum_host_(dataPtr_host.edgeNum_host_), csr_offset_host_(dataPtr_host.csr_offset_host_),
          csr_dest_host_(dataPtr_host.csr_dest_host_), csr_weight_host_(dataPtr_host.csr_weight_host_),
          csr_destWeight_host_(dataPtr_host.csr_destWeight_host_), vertexNum_device_(dataPtr_device.vertexNum_device_),
          edgeNum_device_(dataPtr_device.edgeNum_device_), csr_offset_device_(dataPtr_device.csr_offset_device_),
          csr_dest_device_(dataPtr_device.csr_dest_device_), csr_weight_device_(dataPtr_device.csr_weight_device_),
          csr_destWeight_device_(dataPtr_device.csr_destWeight_device_), cutVertexId_device_(dataPtr_device.cutVertexId_device_),
          isEntireGraph_device_(dataPtr_device.isEntireGraph), disable_device_(dataPtr_device.disableGPU),
          SSSP_destWeight_(dataPtr_host.SSSP_dest_weight && dataPtr_device.SSSP_dest_weight), gpuMemType_(gpuMemType),
          is_sortFrontier_forDevice_((vertexNum_device_ <= std::numeric_limits<int>::max()) && ISSORT),
          num_vertex_bits_forDevice_((int)log2((float)vertexNum_device_) + 1), visitedBitsetNum_host_((vertexNum_host_ + INT_SIZE - 1) / INT_SIZE),
          ThreadNum_host_(omp_get_max_threads()),
          frontierAppendNum_thread_host_((ThreadNum_host_ * CUSTOM_CHUNK_NUM_FRONTIER_APPEND_EACH_THREAD) > visitedBitsetNum_host_
                                             ? ThreadNum_host_
                                             : (ThreadNum_host_ * CUSTOM_CHUNK_NUM_FRONTIER_APPEND_EACH_THREAD)),
          chunkNum_frontierAppend_eachThread_host_(frontierAppendNum_thread_host_ / ThreadNum_host_), CG_ratio_(CG_ratio)
    {
        /* Check DATA */
        globalTimer_.start();
        checkGPUMem(Algorithm_type::SSSP);
        assert_msg(isEntireGraph_device_ ? (cutVertexId_device_ == (vertexNum_host_ - 1)) : true, "Error...");
        Msg_info("Check GPU memory type finish, Used time: %s", globalTimer_.get_time_str().c_str());

        /* Result */
        globalTimer_.start();
        CUDA_CHECK(MALLOC_HOST(&vertexValue_host_, vertexNum_host_));
        CUDA_CHECK(MALLOC_HOST(&vertexValue_temp_host_, vertexNum_host_));
        CUDA_CHECK(MALLOC_DEVICE(&vertexValue_device_, vertexNum_host_));
        CUDA_CHECK(MALLOC_DEVICE(&vertexValue_device2_, vertexNum_host_));
        Msg_info("Graph result vertexValue alloc ready, Used time: %s", globalTimer_.get_time_str().c_str());

        /* Assist */
        globalTimer_.start();
        CUDA_CHECK(MALLOC_HOST(&frontier_in_host_, vertexNum_host_));
        CUDA_CHECK(MALLOC_HOST(&frontier_out_host_, vertexNum_host_));
        CUDA_CHECK(MALLOC_HOST(&frontier_degExSum_host_, vertexNum_host_)); // frontier_degExSum_forPar_hsot_
        CUDA_CHECK(MALLOC_HOST(&frontier_degExSum_forPar_host_, vertexNum_host_));
        CUDA_CHECK(MALLOC_HOST(&frontier_balance_host_, ThreadNum_host_));
        CUDA_CHECK(MALLOC_DEVICE(&frontier_in_device_, vertexNum_device_));
        CUDA_CHECK(MALLOC_DEVICE(&frontier_out_device_, vertexNum_device_));
        CUDA_CHECK(MALLOC_DEVICE(&frontier_degExSum_device_, vertexNum_device_));
        CUDA_CHECK(MALLOC_DEVICE(&frontier_balance_device_, vertexNum_host_)); //! 是否会超出范围
        CUDA_CHECK(MALLOC_DEVICE(&frontierNum_out_device_, 1));
        frontierNum_out_thrust_ = thrust::device_pointer_cast(frontierNum_out_device_);
        frontier_degExSum_thrust_ = thrust::device_pointer_cast(frontier_degExSum_device_);
        frontier_balance_thrust_ = thrust::device_pointer_cast(frontier_balance_device_);
        memset(frontier_in_host_, 0, vertexNum_host_ * sizeof(vertex_id_type));
        memset(frontier_out_host_, 0, vertexNum_host_ * sizeof(vertex_id_type));
        memset(frontier_degExSum_host_, 0, vertexNum_host_ * sizeof(countl_type));
        memset(frontier_degExSum_forPar_host_, 0, vertexNum_host_ * sizeof(countl_type));
        memset(frontier_balance_host_, 0, ThreadNum_host_ * sizeof(countl_type));
        CUDA_CHECK(MEMSET_DEVICE(frontier_in_device_, vertexNum_device_));
        CUDA_CHECK(MEMSET_DEVICE(frontier_out_device_, vertexNum_device_));
        CUDA_CHECK(MEMSET_DEVICE(frontier_degExSum_device_, vertexNum_device_));
        CUDA_CHECK(MEMSET_DEVICE(frontier_balance_device_, vertexNum_host_));
        CUDA_CHECK(MEMSET_DEVICE(frontierNum_out_device_, 1));
        thread_edges_host_ = (countl_type*)CPJ::AllocMem::allocMem<countl_type>((ThreadNum_host_ + 1), socketId_);
        memset(thread_edges_host_, 0, (ThreadNum_host_ + 1) * sizeof(countl_type));
        frontierOut_thread_host_ = (vertex_id_type**)std::aligned_alloc(ALLOC_ALIGNMENT, ThreadNum_host_ * sizeof(vertex_id_type*));
        assert_msg(frontierOut_thread_host_ != nullptr, "std::aligned_alloc error");
        for (int thread_id = 0; thread_id < ThreadNum_host_; thread_id++)
        {
            frontierOut_thread_host_[thread_id] =
                (vertex_id_type*)std::aligned_alloc(ALLOC_ALIGNMENT, HOST_THREAD_FLUSH_VALUE * sizeof(vertex_id_type));
            assert_msg(frontierOut_thread_host_[thread_id] != nullptr, "std::aligned_alloc error");
            memset(frontierOut_thread_host_[thread_id], 0, HOST_THREAD_FLUSH_VALUE * sizeof(vertex_id_type));
        }
        Msg_info("Graph Assist data alloc ready, Used time:  %s", globalTimer_.get_time_str().c_str());

        /* Bitset */
        globalTimer_.start();
        CUDA_CHECK(MALLOC_HOST(&visitedBitset_host_, visitedBitsetNum_host_));
        CUDA_CHECK(MALLOC_HOST(&visitedBitset_storeTempDevice_host_, visitedBitsetNum_host_));
        CUDA_CHECK(MALLOC_HOST(&visitedBitset_curIte_host_, visitedBitsetNum_host_));
        CUDA_CHECK(MALLOC_HOST(&notSinkBitset_host_, visitedBitsetNum_host_));
        CUDA_CHECK(MALLOC_DEVICE(&visitedBitset_device_, visitedBitsetNum_host_));
        CUDA_CHECK(MALLOC_DEVICE(&notSinkBitset_device_, visitedBitsetNum_host_));
        CUDA_CHECK(MALLOC_DEVICE(&visitedBitset_curIte_device_, visitedBitsetNum_host_));
        CUDA_CHECK(MALLOC_DEVICE(&visitedBitset_temp_device_, visitedBitsetNum_host_));
        memset(visitedBitset_host_, 0, visitedBitsetNum_host_ * sizeof(uint32_t));
        memset(visitedBitset_storeTempDevice_host_, 0, visitedBitsetNum_host_ * sizeof(uint32_t));
        memset(visitedBitset_curIte_host_, 0, visitedBitsetNum_host_ * sizeof(uint32_t));
        memset(notSinkBitset_host_, 0, visitedBitsetNum_host_ * sizeof(uint32_t));
        CUDA_CHECK(MEMSET_DEVICE(visitedBitset_device_, visitedBitsetNum_host_));
        CUDA_CHECK(MEMSET_DEVICE(notSinkBitset_device_, visitedBitsetNum_host_));
        CUDA_CHECK(MEMSET_DEVICE(visitedBitset_curIte_device_, visitedBitsetNum_host_));
        CUDA_CHECK(MEMSET_DEVICE(visitedBitset_temp_device_, visitedBitsetNum_host_));
        setSinkBitset();
        CUDA_CHECK(H2D(notSinkBitset_device_, notSinkBitset_host_, visitedBitsetNum_host_));
        Msg_info("Graph bitset alloc and set ready, Used time: %s", globalTimer_.get_time_str().c_str());

        /* Stream */
        globalTimer_.start();
        CUDA_CHECK(cudaStreamCreate(&stream));
        Msg_info("Create stream ready, Used time: %s", globalTimer_.get_time_str().c_str());

        /* Host Balance */
        globalTimer_.start();
        threadState_host_ =
            (Host::Balance_CG::ThreadState_type*)CPJ::AllocMem::allocMem<Host::Balance_CG::ThreadState_type>(ThreadNum_host_, socketId_);
        memset(threadState_host_, 0, sizeof(Host::Balance_CG::ThreadState_type) * ThreadNum_host_);
        threadState_forIncModel_host_ =
            (Host::Balance_CG::ThreadState_forIncModel_type*)CPJ::AllocMem::allocMem<Host::Balance_CG::ThreadState_forIncModel_type>(ThreadNum_host_,
                                                                                                                                     socketId_);
        memset(threadState_forIncModel_host_, 0, sizeof(Host::Balance_CG::ThreadState_forIncModel_type) * ThreadNum_host_);

        frontierAppend_thread_host_ = (count_type*)CPJ::AllocMem::allocMem<count_type>(frontierAppendNum_thread_host_, socketId_);
        memset(frontierAppend_thread_host_, 0, sizeof(count_type) * frontierAppendNum_thread_host_);
        Msg_info("Host Balance alloc and set ready, Used time: %s", globalTimer_.get_time_str().c_str());

        final_ratio_ = CG_ratio_ / (1 + CG_ratio_);
        final_ratio_ = final_ratio_ / 1.05;
        Msg_info("GPU memType[%s], GPU is full graph (%s), cutVertexId_device = %zu / %zu, CG_ratio_ = %.2lf, final_ratio_ = %.2lf",
                 AUTO_GPUMEM_type_name[SCI32(gpuMemType)], isEntireGraph_device_ ? "true" : "false", SCU64(cutVertexId_device_),
                 SCU64(vertexNum_host_), CG_ratio_, final_ratio_);
        assert_msg(ISSORT, "ISSORT must be true");
        USE_DEVICE_FOR_GET_FRONTIER = (USE_DEVICE_FOR_GET_FRONTIER && isEntireGraph_device_ && (gpuMemType == AUTO_GPUMEM_type::FULL_DEVICE_MEM));

        if (SSSP_destWeight_)
        {
            assert_msg((2 * edgeNum_host_) < std::numeric_limits<countl_type>::max(), "SSSP_destWeight_ can lead to error countl_type");
        }

        if (is_sortFrontier_forDevice_)
        {
            cub::DeviceRadixSort::SortKeys(cub_temp_device_, cub_alloc_size_, static_cast<const vertex_id_type*>(frontier_in_device_),
                                           static_cast<vertex_id_type*>(frontier_out_device_), (int)vertexNum_host_);
            CUDA_CHECK(cudaMalloc(&cub_temp_device_, cub_alloc_size_));
        }
    }

    double doSSSP(int64_t root)
    {
        assert_msg(root >= 0 && root < vertexNum_host_, "Error root, root = %ld", root);
        if ((csr_offset_host_[root + 1] - csr_offset_host_[root]) == 0) return 0.0;
        std::fill(vertexValue_host_, vertexValue_host_ + vertexNum_host_, std::numeric_limits<vertex_data_type>::max());
        vertexValue_host_[root] = 0;
        CUDA_CHECK(H2D(vertexValue_device_, vertexValue_host_, vertexNum_host_));

        frontier_in_host_[0] = root;
        frontierNum_in_host_ = 1;
        visitedBitset_host_[root / INT_SIZE] = 1U << (root % INT_SIZE);
        visitedBitset_curIte_host_[root / INT_SIZE] = 1U << (root % INT_SIZE);
        if (root < cutVertexId_device_)
        {
            CUDA_CHECK(H2D(frontier_in_device_, frontier_in_host_, frontierNum_in_host_));
            thrust::device_ptr<uint32_t> visitedBitset_thrust = thrust::device_pointer_cast(visitedBitset_device_);
            visitedBitset_thrust[root / INT_SIZE] = 1U << (root % INT_SIZE);
        }

        std::vector<int> bindCore_vec(1, 0);
        if (agent_device == nullptr) agent_device = new CPJ::ThreadPool(SCU64(1), bindCore_vec);

        countl_type threads_req{0};
        countl_type threads_max_avg{0};
        count_type activeVertices_curIte{0};

        double processTime{0.0};
        ite_ = 0;
        CPJ::Timer totalTime;
        CPJ::Timer singleTime;
        CPJ::Timer tempTime;
        CPJ::Timer t_host;
        CPJ::Timer t_device;
        bool visitedBitset_increment_model = true;
        CGModel_ = CGModel::CPU_ONLY;
        alreadyEdgeNum_BFS_ = 0;

        totalTime.start();
        do
        {
            singleTime.start();
            ite_++;
            if constexpr (TEMP_DETAIL_ACTIVES_DEBUG) activeVertices_curIte = frontierNum_in_host_;

            CGModel_lastIte_ = CGModel_;

            countl_type degreeTotal = 0;
            if ((CGModel_lastIte_ == CGModel::GPU_ONLY) && (isEntireGraph_device_)) degreeTotal = buildDegreeExSum_GPU();
            else degreeTotal = buildDegreeExSum_CPU();
            alreadyEdgeNum_BFS_ += degreeTotal;

            if (!LOCK_GPU_ONLY_) CGModel_ = whichCGModel(frontierNum_in_host_, degreeTotal);
            else CGModel_ = CGModel::GPU_ONLY;

            if (CGModel_ == CGModel::GPU_ONLY)
            {
                CUDA_CHECK(MEMSET_DEVICE(visitedBitset_device_, visitedBitsetNum_host_));
            }
            else if (CGModel_ == CGModel::CPU_ONLY)
            {
                CPJ::ManagedMem::memset_smart(visitedBitset_host_, visitedBitsetNum_host_);
            }
            else
            {
                CUDA_CHECK(MEMSET_DEVICE(visitedBitset_device_, visitedBitsetNum_host_));
                CPJ::ManagedMem::memset_smart(visitedBitset_host_, visitedBitsetNum_host_);
            }

            visitedBitset_increment_model = false;
            if ((degreeTotal > 1000000) && (CGModel_ != CGModel::GPU_ONLY))
            {
                visitedBitset_increment_model = true;
            }

            if (CGModel_ == CGModel::CPU_ONLY)
            {
                threads_req = (degreeTotal + HOST_PER_THREAD_MIN_EDGES_TASKS - 1) / HOST_PER_THREAD_MIN_EDGES_TASKS;
                threads_req = std::min(threads_req, static_cast<countl_type>(ThreadNum_host_));
                threads_max_avg = (degreeTotal + threads_req - 1) / threads_req;
                thread_edges_host_[0] = 0;
                for (int thread_id = 1; thread_id < threads_req; thread_id++)
                    thread_edges_host_[thread_id] = thread_edges_host_[thread_id - 1] + threads_max_avg;
                thread_edges_host_[threads_req] = degreeTotal;
                assert_msg(CGModel_lastIte_ != CGModel::GPU_ONLY, "After CGModel::GPU_ONLY, can not appear CPU_ONLY!");

                thrust::upper_bound(thrust::host, frontier_degExSum_host_, frontier_degExSum_host_ + frontierNum_in_host_, thread_edges_host_,
                                    thread_edges_host_ + threads_req, frontier_balance_host_);

                balance_model(threads_req, frontierNum_in_host_, degreeTotal, visitedBitset_increment_model, frontier_degExSum_host_,
                              frontier_in_host_);

                if (visitedBitset_increment_model)
                {
                    frontierNum_out_host_ = getVisitedIncAndSortedFrontierOut_Opt();
                }

            } // end of [CGModel::CPU_ONLY]

            else if (CGModel_ == CGModel::GPU_ONLY)
            {
                assert_msg(isEntireGraph_device_, "Must be entire graph");
                if (CGModel_lastIte_ != CGModel::GPU_ONLY)
                {
                    if (!USE_DEVICE_FOR_GET_FRONTIER)
                    {
                        CUDA_CHECK(H2D(vertexValue_device2_, vertexValue_host_, vertexNum_device_));
                        int numBlocks = (vertexNum_host_ + BLOCKSIZE_2 - 1) / BLOCKSIZE_2;
                        SSSP_DEVICE_SPACE::merge_vertexValue_kernel<<<numBlocks, BLOCKSIZE_2, 0, stream>>>(vertexValue_device_, vertexValue_device2_,
                                                                                                           vertexNum_host_, notSinkBitset_device_);
                        CUDA_CHECK(H2D(frontier_in_device_, frontier_in_host_, frontierNum_in_host_));
                    }
                    CUDA_CHECK(H2D(frontier_degExSum_device_, frontier_degExSum_host_, frontierNum_in_host_));
                }

                if (CGModel_lastIte_ != CGModel::GPU_ONLY)
                {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    std::swap(vertexValue_device_, vertexValue_device2_);
                }

                nBlock_ = (degreeTotal + TASK_PER_BLOCK - 1) / TASK_PER_BLOCK;
                thrust::counting_iterator<countl_type> cnt_iter(0);
                auto query_iter_first = thrust::make_transform_iterator(cnt_iter, Device::Set_blockTask_type());
                auto query_iter_last = thrust::make_transform_iterator(cnt_iter + nBlock_, Device::Set_blockTask_type());
                assert_msg(vertexNum_device_ > (query_iter_last - query_iter_first),
                           "frontier_balance_thrust_ too large, vertexNum_ = %zu, (query_iter_last - query_iter_first) = %zu",
                           SCU64(vertexNum_device_),
                           SCU64((query_iter_last - query_iter_first))); // TODO 尝试取消(这是在检查frontier_balance_device_的预申请容量)
                thrust::upper_bound(thrust::device, frontier_degExSum_device_, frontier_degExSum_device_ + frontierNum_in_host_, query_iter_first,
                                    query_iter_last, frontier_balance_device_);

                nBlock_ = std::min(nBlock_, SCU64(MAX_BLOCKS));
                SSSP_DEVICE_SPACE::balance_model(nBlock_, vertexValue_device_, csr_offset_device_, csr_dest_device_, csr_weight_device_,
                                                 csr_destWeight_device_, frontier_in_device_, frontier_out_device_, frontierNum_in_host_, degreeTotal,
                                                 frontierNum_out_device_, frontier_degExSum_device_, frontier_balance_device_, notSinkBitset_device_,
                                                 reinterpret_cast<int*>(visitedBitset_device_), SSSP_destWeight_);

                frontierNum_out_host_.store(frontierNum_out_thrust_[0]);
                frontierNum_out_host_ = frontierNum_out_thrust_[0];

            } // end of [CGModel::GPU_ONLY]

            else if (CGModel_ == CGModel::CPU_GPU)
            {
                auto result = balance_CPUGPU(frontierNum_in_host_, degreeTotal);
                const count_type cutVertexIndex = std::get<0>(result);
                const count_type activeVertexNum_host_final = std::get<1>(result);
                const countl_type activeEdgeNum_host_final = std::get<2>(result);
                const count_type activeVertexNum_device_final = std::get<3>(result);
                const countl_type activeEdgeNum_device_final = std::get<4>(result);

                if (CGModel_lastIte_ == CGModel::CPU_GPU) CUDA_CHECK(H2D(vertexValue_device2_, vertexValue_host_, vertexNum_host_));
                else CUDA_CHECK(H2D(vertexValue_device_, vertexValue_host_, vertexNum_host_));

                auto agent_device_return =
                    agent_device->enqueue(&CG_SSSP::balance_model_device_inCPUGPU, this, activeVertexNum_device_final, activeEdgeNum_device_final);

                threads_req = (activeEdgeNum_host_final + HOST_PER_THREAD_MIN_EDGES_TASKS - 1) / HOST_PER_THREAD_MIN_EDGES_TASKS;
                threads_req = std::min(threads_req, static_cast<countl_type>(ThreadNum_host_));
                threads_max_avg = (activeEdgeNum_host_final + threads_req - 1) / threads_req;
                thread_edges_host_[0] = 0;
                for (int thread_id = 1; thread_id < threads_req; thread_id++)
                    thread_edges_host_[thread_id] = thread_edges_host_[thread_id - 1] + threads_max_avg;
                thread_edges_host_[threads_req] = activeEdgeNum_host_final;
                countl_type temp = frontier_degExSum_host_[cutVertexIndex];
                omp_par_for_threads(threads_req)(count_type cut_index = cutVertexIndex; cut_index < frontierNum_in_host_; cut_index++)
                {
                    frontier_degExSum_host_[cut_index] -= temp;
                }
                thrust::upper_bound(thrust::host, frontier_degExSum_host_ + cutVertexIndex, frontier_degExSum_host_ + frontierNum_in_host_,
                                    thread_edges_host_, thread_edges_host_ + threads_req, frontier_balance_host_);
                balance_model_inc_host(threads_req, activeVertexNum_host_final, activeEdgeNum_host_final, frontier_degExSum_host_ + cutVertexIndex,
                                       frontier_in_host_ + cutVertexIndex);

                assert_msg(agent_device_return.valid(), "agent_device_return.valid() = false");
                agent_device_return.wait();

                if (!USE_DEVICE_FOR_GET_FRONTIER)
                {
                    omp_par_for(count_type index = 0; index < visitedBitsetNum_host_; index++)
                    {
                        visitedBitset_host_[index] |= visitedBitset_storeTempDevice_host_[index];
                    }
                }
                frontierNum_out_host_ = getVisitedIncAndSortedFrontierOut_Opt_CPUGPU();
            }

            else
            {
                assert_msg(false, "Unknow CGModel");
            }

            if (frontierNum_out_host_ == 0)
            {
                processTime = totalTime.get_time_ms();
                if constexpr (TEMP_DETAIL_ACTIVES_DEBUG)
                {
                    /* 打印的是当前迭代处理过的活跃顶点和活跃边 */
                    Msg_info("\t[CGPU_SSSP]: The (%2d) iteration, activeNum[C/C] = %zu (%zu), Used time: = %4.2lf (ms)", ite_,
                             SCU64(activeVertices_curIte), SCU64(degreeTotal), singleTime.get_time_ms());
                }
                else
                {
                    /* 打印的是当前迭代处理过的活跃边和下一轮的活跃定带你 */
                    Msg_info("\t[CGPU_SSSP]: The (%2d) iteration, activeNum[N/C] = %zu (%zu), Used time: = %4.2lf (ms)", ite_,
                             SCU64(frontierNum_in_host_), SCU64(degreeTotal), singleTime.get_time_ms());
                }
                Msg_node("[Complete]: CGPU_SSSP -> iteration: %2d", ite_);
                break;
            }

            if (CGModel_ == CGModel::CPU_ONLY)
            {
                if (ISSORT && !visitedBitset_increment_model)
                {
                    if constexpr (USED_STD_PAR)
                    {
                        std::sort(std::execution::par, frontier_out_host_, frontier_out_host_ + frontierNum_out_host_);
                    }
                    else
                    {
                        std::sort(frontier_out_host_, frontier_out_host_ + frontierNum_out_host_);
                    }
                }

                frontierNum_in_host_ = frontierNum_out_host_;
                frontierNum_out_host_ = 0;
                std::swap(frontier_in_host_, frontier_out_host_);

            } // end of [CGModel::CPU_ONLY]

            else if (CGModel_ == CGModel::GPU_ONLY)
            {
                if (is_sortFrontier_forDevice_)
                {
                    int start_bit = static_cast<int>(0.35 * num_vertex_bits_forDevice_);
                    cub::DeviceRadixSort::SortKeys(cub_temp_device_, cub_alloc_size_, static_cast<const vertex_id_type*>(frontier_out_device_),
                                                   static_cast<vertex_id_type*>(frontier_in_device_), (int)frontierNum_out_host_, start_bit,
                                                   num_vertex_bits_forDevice_);
                }

                frontierNum_in_host_ = frontierNum_out_host_;
                frontierNum_out_host_ = 0;
                CUDA_CHECK(MEMSET_DEVICE(frontierNum_out_device_, 1));

            } // end of [CGModel::GPU_ONLY]

            else
            {
                frontierNum_in_host_ = frontierNum_out_host_;
                frontierNum_out_host_ = 0;
                if (!USE_DEVICE_FOR_GET_FRONTIER)
                {
                    co_canCPUrun_ = false;
                }
                else
                {
                    std::swap(frontier_in_device_, frontier_out_device_);
                }
                CUDA_CHECK(MEMSET_DEVICE(frontierNum_out_device_, 1));
                std::swap(frontier_in_host_, frontier_out_host_);
            }

            if constexpr (TEMP_DETAIL_ACTIVES_DEBUG)
            {
                /* 打印的是当前迭代处理过的活跃顶点和活跃边 */
                Msg_info("\t[CGPU_SSSP]: The (%2d) iteration, activeNum[C/C] = %zu (%zu), Used time: = %4.2lf (ms)", ite_,
                         SCU64(activeVertices_curIte), SCU64(degreeTotal), singleTime.get_time_ms());
                // Msg_check("\t result_right[275] = %u", vertexValue_host_[275]);
            }
            else
            {
                /* 打印的是当前迭代处理过的活跃边和下一轮的活跃定带你 */
                Msg_info("\t[CGPU_SSSP]: The (%2d) iteration, activeNum[N/C] = %zu (%zu), Used time: = %4.2lf (ms)", ite_,
                         SCU64(frontierNum_in_host_), SCU64(degreeTotal), singleTime.get_time_ms());
            }
        }
        while (true);

        agent_device->clearPool();
        assert_msg(agent_device->isEmpty(), "Device Agent not free, please check");
        agent_device = nullptr;

        return processTime;

    } // end of Func [doSSSP]

    vertex_data_type* getResult()
    {
        CUDA_CHECK(D2H(vertexValue_temp_host_, vertexValue_device_, vertexNum_host_));
        omp_par_for(count_type vertex_id = 0; vertex_id < vertexNum_host_; vertex_id++)
        {
            vertexValue_host_[vertex_id] = std::min(vertexValue_temp_host_[vertex_id], vertexValue_host_[vertex_id]);
        }
        return vertexValue_host_;
    }

  private:
    void checkGPUMem(Algorithm_type algorithm)
    {
        cudaPointerAttributes attributes;
        switch (gpuMemType_)
        {
        case AUTO_GPUMEM_type::FULL_DEVICE_MEM:
        case AUTO_GPUMEM_type::PARTIAL_DEVICE_MEM:
            CUDA_CHECK(cudaPointerGetAttributes(&attributes, (const void*)csr_offset_device_));
            assert_msg(attributes.type == cudaMemoryType::cudaMemoryTypeDevice, "csr_offset_device_ is not devicePtr, usedGPUType = %s",
                       GPU_memory_type_name[SCI32(gpuMemType_)]);
            if (SSSP_destWeight_)
            {
                CUDA_CHECK(cudaPointerGetAttributes(&attributes, (const void*)csr_destWeight_device_));
                assert_msg(attributes.type == cudaMemoryType::cudaMemoryTypeDevice,
                           "[SSSP_destWeight_] csr_destWeight_device_ is not cudaMemoryTypeDevice, usedGPUType = %s",
                           GPU_memory_type_name[SCI32(gpuMemType_)]);

                CUDA_CHECK(cudaPointerGetAttributes(&attributes, (const void*)csr_dest_device_));
                assert_msg(attributes.type == cudaMemoryType::cudaMemoryTypeUnregistered,
                           "[SSSP_destWeight_] csr_dest_device_ is not cudaMemoryTypeUnregistered, usedGPUType = %s",
                           GPU_memory_type_name[SCI32(gpuMemType_)]);
                if (algorithm == Algorithm_type::SSSP)
                {
                    CUDA_CHECK(cudaPointerGetAttributes(&attributes, (const void*)csr_weight_device_));
                    assert_msg(attributes.type == cudaMemoryType::cudaMemoryTypeUnregistered,
                               "[SSSP_destWeight_] csr_weight_device_ is not cudaMemoryTypeUnregistered, usedGPUType = %s",
                               GPU_memory_type_name[SCI32(gpuMemType_)]);
                }
            }
            else
            {
                CUDA_CHECK(cudaPointerGetAttributes(&attributes, (const void*)csr_destWeight_device_));
                assert_msg(attributes.type == cudaMemoryType::cudaMemoryTypeUnregistered,
                           "[SSSP_destWeight_] csr_destWeight_device_ is not cudaMemoryTypeUnregistered, usedGPUType = %s",
                           GPU_memory_type_name[SCI32(gpuMemType_)]);

                CUDA_CHECK(cudaPointerGetAttributes(&attributes, (const void*)csr_dest_device_));
                assert_msg(attributes.type == cudaMemoryType::cudaMemoryTypeDevice,
                           "[SSSP_destWeight_] csr_dest_device_ is not devicePtr, usedGPUType = %s", GPU_memory_type_name[SCI32(gpuMemType_)]);
                if (algorithm == Algorithm_type::SSSP)
                {
                    CUDA_CHECK(cudaPointerGetAttributes(&attributes, (const void*)csr_weight_device_));
                    assert_msg(attributes.type == cudaMemoryType::cudaMemoryTypeDevice,
                               "[SSSP_destWeight_] csr_weight_device_ is not devicePtr, usedGPUType = %s", GPU_memory_type_name[SCI32(gpuMemType_)]);
                }
            }

            break;

        case AUTO_GPUMEM_type::DISABLE_DEVICE_MEM:
            CUDA_CHECK(cudaPointerGetAttributes(&attributes, (const void*)csr_offset_device_));
            assert_msg(attributes.type == cudaMemoryType::cudaMemoryTypeDevice, "csr_offset_device_ is not devicePtr, usedGPUType = %s",
                       GPU_memory_type_name[SCI32(gpuMemType_)]);

            if (SSSP_destWeight_)
            {
                CUDA_CHECK(cudaPointerGetAttributes(&attributes, (const void*)csr_destWeight_device_));
                assert_msg(attributes.type == cudaMemoryType::cudaMemoryTypeManaged,
                           "[SSSP_destWeight_] csr_destWeight_device_ is not cudaMemoryTypeManaged, usedGPUType = %s",
                           GPU_memory_type_name[SCI32(gpuMemType_)]);

                CUDA_CHECK(cudaPointerGetAttributes(&attributes, (const void*)csr_dest_device_));
                assert_msg(attributes.type == cudaMemoryType::cudaMemoryTypeUnregistered,
                           "[SSSP_destWeight_] csr_dest_device_ is not cudaMemoryTypeUnregistered, usedGPUType = %s",
                           GPU_memory_type_name[SCI32(gpuMemType_)]);
                if (algorithm == Algorithm_type::SSSP)
                {
                    CUDA_CHECK(cudaPointerGetAttributes(&attributes, (const void*)csr_weight_device_));
                    assert_msg(attributes.type == cudaMemoryType::cudaMemoryTypeUnregistered,
                               "[SSSP_destWeight_] csr_weight_device_ is not cudaMemoryTypeUnregistered, usedGPUType = %s",
                               GPU_memory_type_name[SCI32(gpuMemType_)]);
                }
            }
            else
            {
                CUDA_CHECK(cudaPointerGetAttributes(&attributes, (const void*)csr_destWeight_device_));
                assert_msg(attributes.type == cudaMemoryType::cudaMemoryTypeUnregistered,
                           "[SSSP_destWeight_] csr_destWeight_device_ is not cudaMemoryTypeUnregistered, usedGPUType = %s",
                           GPU_memory_type_name[SCI32(gpuMemType_)]);

                CUDA_CHECK(cudaPointerGetAttributes(&attributes, (const void*)csr_dest_device_));
                assert_msg(attributes.type == cudaMemoryType::cudaMemoryTypeManaged,
                           "[SSSP_destWeight_] csr_dest_device_ is not cudaMemoryTypeManaged, usedGPUType = %s",
                           GPU_memory_type_name[SCI32(gpuMemType_)]);
                if (algorithm == Algorithm_type::SSSP)
                {
                    CUDA_CHECK(cudaPointerGetAttributes(&attributes, (const void*)csr_weight_device_));
                    assert_msg(attributes.type == cudaMemoryType::cudaMemoryTypeManaged,
                               "[SSSP_destWeight_] csr_weight_device_ is not cudaMemoryTypeManaged, usedGPUType = %s",
                               GPU_memory_type_name[SCI32(gpuMemType_)]);
                }
            }
            break;

        case AUTO_GPUMEM_type::DISABLE_GPU:
            break;
        case AUTO_GPUMEM_type::MAX_VALUE:
            assert_msg(false, "Error ...");
            break;
        }
    }

    void setSinkBitset()
    {
        count_type sinkNum = 0;
        omp_par_for_reductionAdd(sinkNum)(count_type bitSet_id = 0; bitSet_id < visitedBitsetNum_host_; bitSet_id++)
        {
            count_type vertexId = bitSet_id * INT_SIZE;
            uint32_t mask = 0;
#pragma unroll
            for (uint32_t bit_id = 0; bit_id < INT_SIZE; bit_id++)
            {
                uint32_t temp = (vertexId < vertexNum_host_) ? ((!(csr_offset_host_[vertexId + 1] - csr_offset_host_[vertexId])) ? 1 : 0) : 0;
                mask |= (temp << bit_id);
                vertexId++;

                if ((csr_offset_host_[vertexId + 1] - csr_offset_host_[vertexId]) == 0) sinkNum++;
            }
            notSinkBitset_host_[bitSet_id] = (~mask); // 这里取反
        }
        noSink_avgDegree_ = std::round(SCD(edgeNum_host_) / SCD(vertexNum_host_ - sinkNum));
        Msg_info("SinkNum = %zu, noSink_avgDegree = %zu", SCU64(sinkNum), SCU64(noSink_avgDegree_));
    }

    void setActiveDegree_host(const count_type PAR_TH = 1024)
    {
        if (frontierNum_in_host_ <= PAR_TH)
        {
            for (count_type frontier_in_id = 0; frontier_in_id < frontierNum_in_host_; frontier_in_id++)
            {
                vertex_id_type vertexId = frontier_in_host_[frontier_in_id];
                countl_type degree = csr_offset_host_[vertexId + 1] - csr_offset_host_[vertexId];
                frontier_degExSum_host_[frontier_in_id] = degree;
            }
        }
        else
        {
            omp_par_for(count_type frontier_in_id = 0; frontier_in_id < frontierNum_in_host_; frontier_in_id++)
            {
                vertex_id_type vertexId = frontier_in_host_[frontier_in_id];
                countl_type degree = csr_offset_host_[vertexId + 1] - csr_offset_host_[vertexId];
                frontier_degExSum_host_[frontier_in_id] = degree;
            }
        }
    }

    inline countl_type buildDegreeExSum_CPU()
    {
        countl_type degreeTotal{0};
        setActiveDegree_host();

        if (USED_STD_PAR && (frontierNum_in_host_ > 10000))
        {
            degreeTotal = CPJ::SCAN::HOST::exclusive_scan_std_par(frontier_degExSum_host_, frontierNum_in_host_, frontier_degExSum_forPar_host_);
            std::swap(frontier_degExSum_forPar_host_, frontier_degExSum_host_);
        }
        else
        {
            degreeTotal = CPJ::SCAN::HOST::exclusive_scan_std(frontier_degExSum_host_, frontierNum_in_host_, frontier_degExSum_host_);
        }
        return degreeTotal;
    }

    void setActiveDegree_device()
    {
        thrust::counting_iterator<count_type> iter_first(0);
        thrust::counting_iterator<count_type> iter_last = iter_first + frontierNum_in_host_;
        thrust::for_each(thrust::device, iter_first, iter_last,
                         Device::Set_activeDegree_type(frontier_in_device_, csr_offset_device_, frontier_degExSum_device_));
    }

    inline countl_type buildDegreeExSum_GPU()
    {
        countl_type degreeTotal{0};
        setActiveDegree_device();
        degreeTotal = CPJ::SCAN::Device::exclusive_scan_thrust(frontier_degExSum_device_, frontierNum_in_host_, frontier_degExSum_device_);
        return degreeTotal;
    }

    CGModel whichCGModel(const count_type activeVertexNum, const countl_type activeEdgeNum)
    {
        if (activeVertexNum + activeEdgeNum <= TH_CPU_ONLY_EDGENUM_v1)
        {
            return CGModel::CPU_ONLY;
        }

        if (activeVertexNum + activeEdgeNum >= TH_CPU_GPU_EDGENUM_v1)
        {
            return CGModel::CPU_GPU;
        }

        if ((alreadyEdgeNum_BFS_ / SCD(edgeNum_host_) > 0.9) && isEntireGraph_device_)
        {
            LOCK_GPU_ONLY_ = true;
            return CGModel::GPU_ONLY;
        }

        return CGModel::CPU_ONLY;
    }

    void balance_model(const int threads_req, const count_type activeVertexNum, const countl_type degreeTotal,
                       const bool visitedBitset_increment_model, const countl_type* frontier_degExSum_host, const vertex_id_type* frontier_in_host)
    {
        if (visitedBitset_increment_model)
        {
            balance_model_inc_host(threads_req, activeVertexNum, degreeTotal, frontier_degExSum_host, frontier_in_host);
        }
        else
        {
            balance_model_noInc_host(threads_req, activeVertexNum, degreeTotal, frontier_degExSum_host, frontier_in_host);
        }
    }

    void balance_model_inc_host(const int threads_req, const count_type activeVertexNum, const countl_type degreeTotal,
                                const countl_type* frontier_degExSum_host, const vertex_id_type* frontier_in_host)
    {

        omp_par_threads(threads_req)
        {
            CPJ::Timer time;
            const int threadId = omp_get_thread_num();
            const countl_type firstEdgeIndex_thread = thread_edges_host_[threadId];
            const countl_type lastEdgeIndex_thread = thread_edges_host_[threadId + 1] - 1;

            const count_type firstVertexIndex_thread = frontier_balance_host_[threadId] - 1;
            count_type lastVertexIndex_thread;
            if (lastEdgeIndex_thread < (degreeTotal - 1))
            {
                lastVertexIndex_thread = frontier_balance_host_[threadId + 1] - 1;
                if (frontier_degExSum_host_[lastVertexIndex_thread] == lastEdgeIndex_thread + 1) lastVertexIndex_thread--;
            }
            else
            {
                lastVertexIndex_thread = activeVertexNum - 1;
            }

            const count_type vertexNum_thread = lastVertexIndex_thread - firstVertexIndex_thread + 1;
            const degree_type firstNbrIndex_InFirstVertex_thread = firstEdgeIndex_thread - frontier_degExSum_host[firstVertexIndex_thread];
            const degree_type lastNbrIndex_InLastVertex_thread = lastEdgeIndex_thread - frontier_degExSum_host[lastVertexIndex_thread];

            threadState_host_[threadId].start = firstVertexIndex_thread;
            threadState_host_[threadId].cur = firstVertexIndex_thread;
            threadState_host_[threadId].end = firstVertexIndex_thread + vertexNum_thread;
            threadState_host_[threadId].firstNbrIndex_InFirstVertex_thread = firstNbrIndex_InFirstVertex_thread;
            threadState_host_[threadId].lastNbrIndex_InLastVertex_thread = lastNbrIndex_InLastVertex_thread;
            threadState_host_[threadId].status = Host::Balance_CG::ThreadStatus::VERTEX_WORKING;

            while (true)
            {
                const count_type curAtomic_fetch_forVertexIndex =
                    threadState_host_[threadId].cur.fetch_add(HOST_PER_THREAD_WORKING_VERTEX, ATOMIC_ORDER);
                if (curAtomic_fetch_forVertexIndex >= threadState_host_[threadId].end) break;
                const count_type curAtomic_fetch_forLastVertexIndex =
                    std::min(threadState_host_[threadId].end, (curAtomic_fetch_forVertexIndex + HOST_PER_THREAD_WORKING_VERTEX));
                for (count_type iter_id_forVertex = curAtomic_fetch_forVertexIndex; iter_id_forVertex < curAtomic_fetch_forLastVertexIndex;
                     iter_id_forVertex++)
                {
                    const vertex_id_type vid = frontier_in_host[iter_id_forVertex];
                    countl_type nbrStartIndex = csr_offset_host_[vid];
                    const degree_type nbrSize = csr_offset_host_[vid + 1] - nbrStartIndex;
                    degree_type nbrSize_forCurVertex_thread = nbrSize;

                    if (iter_id_forVertex == threadState_host_[threadId].start)
                    {
                        nbrSize_forCurVertex_thread -= firstNbrIndex_InFirstVertex_thread;
                        nbrStartIndex += firstNbrIndex_InFirstVertex_thread;
                    }

                    if (iter_id_forVertex == (threadState_host_[threadId].end - 1))
                    {
                        nbrSize_forCurVertex_thread -= (nbrSize - lastNbrIndex_InLastVertex_thread - 1);
                    }

                    for (degree_type nbr_id = 0; nbr_id < nbrSize_forCurVertex_thread; nbr_id++)
                    {
                        countl_type nbr_index = (nbrStartIndex + nbr_id);
                        vertex_id_type dest;
                        edge_data_type weight;
                        if (SSSP_destWeight_)
                        {
                            nbr_index = nbr_index * 2;
                            dest = csr_destWeight_host_[nbr_index];
                            weight = csr_destWeight_host_[nbr_index + 1];
                        }
                        else
                        {
                            dest = csr_dest_host_[nbr_index];
                            weight = csr_weight_host_[nbr_index];
                        }

                        edge_data_type msg = vertexValue_host_[vid] + weight;
                        if (msg < vertexValue_host_[dest])
                        {
                            if (LinuxAtomic::write_min(&vertexValue_host_[dest], msg))
                            {
                                bool is_notSink = (notSinkBitset_host_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                if (is_notSink)
                                {
                                    bool is_visited = (visitedBitset_host_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                    if (!is_visited)
                                    {
                                        int new_val = 1 << (dest % INT_SIZE);
                                        visitedBitset_host_[dest / INT_SIZE].fetch_or(new_val, ATOMIC_ORDER);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            threadState_host_[threadId].status = Host::Balance_CG::ThreadStatus::VERTEX_STEALING;
            for (int steal_offset = 1; steal_offset < threads_req; steal_offset++)
            {
                const int threadId_help = (threadId + steal_offset) % threads_req;
                while (threadState_host_[threadId_help].status != Host::Balance_CG::ThreadStatus::VERTEX_STEALING)
                {
                    const count_type curAtomic_fetch_forVertexIndex =
                        threadState_host_[threadId_help].cur.fetch_add(HOST_PER_THREAD_STEALING_VERTEX, ATOMIC_ORDER);
                    if (curAtomic_fetch_forVertexIndex >= threadState_host_[threadId_help].end) break;
                    const count_type curAtomic_fetch_forLastVertexIndex =
                        std::min(threadState_host_[threadId_help].end, (curAtomic_fetch_forVertexIndex + HOST_PER_THREAD_STEALING_VERTEX));
                    for (count_type iter_id_forVertex = curAtomic_fetch_forVertexIndex; iter_id_forVertex < curAtomic_fetch_forLastVertexIndex;
                         iter_id_forVertex++)
                    {
                        const vertex_id_type vid = frontier_in_host[iter_id_forVertex];
                        countl_type nbrStartIndex = csr_offset_host_[vid];
                        const degree_type nbrSize = csr_offset_host_[vid + 1] - nbrStartIndex;
                        degree_type nbrSize_forCurVertex_thread = nbrSize;

                        if (iter_id_forVertex == threadState_host_[threadId_help].start)
                        {
                            nbrSize_forCurVertex_thread -= threadState_host_[threadId_help].firstNbrIndex_InFirstVertex_thread;
                            nbrStartIndex += threadState_host_[threadId_help].firstNbrIndex_InFirstVertex_thread;
                        }

                        if (iter_id_forVertex == (threadState_host_[threadId_help].end - 1))
                        {
                            nbrSize_forCurVertex_thread -= (nbrSize - threadState_host_[threadId_help].lastNbrIndex_InLastVertex_thread - 1);
                        }

                        for (degree_type nbr_id = 0; nbr_id < nbrSize_forCurVertex_thread; nbr_id++)
                        {
                            countl_type nbr_index = (nbrStartIndex + nbr_id);
                            vertex_id_type dest;
                            edge_data_type weight;
                            if (SSSP_destWeight_)
                            {
                                nbr_index = nbr_index * 2;
                                dest = csr_destWeight_host_[nbr_index];
                                weight = csr_destWeight_host_[nbr_index + 1];
                            }
                            else
                            {
                                dest = csr_dest_host_[nbr_index];
                                weight = csr_weight_host_[nbr_index];
                            }

                            edge_data_type msg = vertexValue_host_[vid] + weight;
                            if (msg < vertexValue_host_[dest])
                            {
                                if (LinuxAtomic::write_min(&vertexValue_host_[dest], msg))
                                {
                                    bool is_notSink = (notSinkBitset_host_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                    if (is_notSink)
                                    {
                                        bool is_visited = (visitedBitset_host_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                        if (!is_visited)
                                        {
                                            int new_val = 1 << (dest % INT_SIZE);
                                            visitedBitset_host_[dest / INT_SIZE].fetch_or(new_val, ATOMIC_ORDER);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void balance_model_noInc_host(const int threads_req, const count_type activeVertexNum, const countl_type degreeTotal,
                                  const countl_type* frontier_degExSum_host, const vertex_id_type* frontier_in_host)
    {
        omp_par_threads(threads_req)
        {
            const int threadId = omp_get_thread_num();
            const countl_type firstEdgeIndex_thread = thread_edges_host_[threadId];
            const countl_type lastEdgeIndex_thread = thread_edges_host_[threadId + 1] - 1;

            const count_type firstVertexIndex_thread = frontier_balance_host_[threadId] - 1;
            count_type lastVertexIndex_thread;
            if (lastEdgeIndex_thread < (degreeTotal - 1))
            {
                lastVertexIndex_thread = frontier_balance_host_[threadId + 1] - 1;
                if (frontier_degExSum_host[lastVertexIndex_thread] == lastEdgeIndex_thread + 1) lastVertexIndex_thread--;
            }
            else
            {
                lastVertexIndex_thread = activeVertexNum - 1;
            }

            const count_type vertexNum_thread = lastVertexIndex_thread - firstVertexIndex_thread + 1;
            const degree_type firstNbrIndex_InFirstVertex_thread = firstEdgeIndex_thread - frontier_degExSum_host[firstVertexIndex_thread];
            const degree_type lastNbrIndex_InLastVertex_thread = lastEdgeIndex_thread - frontier_degExSum_host[lastVertexIndex_thread];

            threadState_host_[threadId].start = firstVertexIndex_thread;
            threadState_host_[threadId].cur = firstVertexIndex_thread;
            threadState_host_[threadId].end = firstVertexIndex_thread + vertexNum_thread;
            threadState_host_[threadId].firstNbrIndex_InFirstVertex_thread = firstNbrIndex_InFirstVertex_thread;
            threadState_host_[threadId].lastNbrIndex_InLastVertex_thread = lastNbrIndex_InLastVertex_thread;
            threadState_host_[threadId].status = Host::Balance_CG::ThreadStatus::VERTEX_WORKING;

            count_type frontierCounter_thread = 0;
            while (true)
            {
                const count_type curAtomic_fetch_forVertexIndex =
                    threadState_host_[threadId].cur.fetch_add(HOST_PER_THREAD_WORKING_VERTEX, ATOMIC_ORDER);
                if (curAtomic_fetch_forVertexIndex >= threadState_host_[threadId].end) break;
                const count_type curAtomic_fetch_forLastVertexIndex =
                    std::min(threadState_host_[threadId].end, (curAtomic_fetch_forVertexIndex + HOST_PER_THREAD_WORKING_VERTEX));
                for (count_type iter_id_forVertex = curAtomic_fetch_forVertexIndex; iter_id_forVertex < curAtomic_fetch_forLastVertexIndex;
                     iter_id_forVertex++)
                {
                    const vertex_id_type vid = frontier_in_host[iter_id_forVertex];
                    countl_type nbrStartIndex = csr_offset_host_[vid];
                    const degree_type nbrSize = csr_offset_host_[vid + 1] - nbrStartIndex;
                    degree_type nbrSize_forCurVertex_thread = nbrSize;

                    if (iter_id_forVertex == threadState_host_[threadId].start)
                    {
                        nbrSize_forCurVertex_thread -= firstNbrIndex_InFirstVertex_thread;
                        nbrStartIndex += firstNbrIndex_InFirstVertex_thread;
                    }

                    if (iter_id_forVertex == (threadState_host_[threadId].end - 1))
                    {
                        nbrSize_forCurVertex_thread -= (nbrSize - lastNbrIndex_InLastVertex_thread - 1);
                    }

                    for (degree_type nbr_id = 0; nbr_id < nbrSize_forCurVertex_thread; nbr_id++)
                    {

                        countl_type nbr_index = (nbrStartIndex + nbr_id);
                        vertex_id_type dest;
                        edge_data_type weight;
                        if (SSSP_destWeight_)
                        {
                            nbr_index = nbr_index * 2;
                            dest = csr_destWeight_host_[nbr_index];
                            weight = csr_destWeight_host_[nbr_index + 1];
                        }
                        else
                        {
                            dest = csr_dest_host_[nbr_index];
                            weight = csr_weight_host_[nbr_index];
                        }
                        edge_data_type msg = vertexValue_host_[vid] + weight;
                        if (msg < vertexValue_host_[dest])
                        {
                            if (LinuxAtomic::write_min(&vertexValue_host_[dest], msg))

                            {
                                bool is_visited = (visitedBitset_host_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                if (!is_visited)
                                {
                                    int new_val = 1 << (dest % INT_SIZE);
                                    uint32_t old_val = visitedBitset_host_[dest / INT_SIZE].fetch_or(new_val, ATOMIC_ORDER);
                                    bool is_notSink = (notSinkBitset_host_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                    if (is_notSink)
                                    {
                                        bool atomic_was_visited = (old_val >> (dest % INT_SIZE)) & 1;
                                        if (!atomic_was_visited)
                                        {
                                            frontierOut_thread_host_[threadId][frontierCounter_thread] = dest;
                                            frontierCounter_thread++;

                                            if (frontierCounter_thread == HOST_THREAD_FLUSH_VALUE)
                                            {
                                                count_type old_index = frontierNum_out_host_.fetch_add(frontierCounter_thread, ATOMIC_ORDER);
                                                std::memcpy(frontier_out_host_ + old_index, frontierOut_thread_host_[threadId],
                                                            frontierCounter_thread * sizeof(vertex_id_type));
                                                frontierCounter_thread = 0;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            threadState_host_[threadId].status = Host::Balance_CG::ThreadStatus::VERTEX_STEALING;
            for (int steal_offset = 1; steal_offset < threads_req; steal_offset++)
            {
                const int threadId_help = (threadId + steal_offset) % threads_req;
                while (threadState_host_[threadId_help].status != Host::Balance_CG::ThreadStatus::VERTEX_STEALING)
                {
                    const count_type curAtomic_fetch_forVertexIndex =
                        threadState_host_[threadId_help].cur.fetch_add(HOST_PER_THREAD_STEALING_VERTEX, ATOMIC_ORDER);
                    if (curAtomic_fetch_forVertexIndex >= threadState_host_[threadId_help].end) break;
                    const count_type curAtomic_fetch_forLastVertexIndex =
                        std::min(threadState_host_[threadId_help].end, (curAtomic_fetch_forVertexIndex + HOST_PER_THREAD_STEALING_VERTEX));
                    for (count_type iter_id_forVertex = curAtomic_fetch_forVertexIndex; iter_id_forVertex < curAtomic_fetch_forLastVertexIndex;
                         iter_id_forVertex++)
                    {
                        const vertex_id_type vid = frontier_in_host[iter_id_forVertex];
                        countl_type nbrStartIndex = csr_offset_host_[vid];
                        const degree_type nbrSize = csr_offset_host_[vid + 1] - nbrStartIndex;
                        degree_type nbrSize_forCurVertex_thread = nbrSize;

                        if (iter_id_forVertex == threadState_host_[threadId_help].start)
                        {
                            nbrSize_forCurVertex_thread -= threadState_host_[threadId_help].firstNbrIndex_InFirstVertex_thread;
                            nbrStartIndex += threadState_host_[threadId_help].firstNbrIndex_InFirstVertex_thread;
                        }

                        if (iter_id_forVertex == (threadState_host_[threadId_help].end - 1))
                        {
                            nbrSize_forCurVertex_thread -= (nbrSize - threadState_host_[threadId_help].lastNbrIndex_InLastVertex_thread - 1);
                        }

                        for (degree_type nbr_id = 0; nbr_id < nbrSize_forCurVertex_thread; nbr_id++)
                        {
                            countl_type nbr_index = (nbrStartIndex + nbr_id);
                            vertex_id_type dest;
                            edge_data_type weight;
                            if (SSSP_destWeight_)
                            {
                                nbr_index = nbr_index * 2;
                                dest = csr_destWeight_host_[nbr_index];
                                weight = csr_destWeight_host_[nbr_index + 1];
                            }
                            else
                            {
                                dest = csr_dest_host_[nbr_index];
                                weight = csr_weight_host_[nbr_index];
                            }
                            edge_data_type msg = vertexValue_host_[vid] + weight;
                            if (msg < vertexValue_host_[dest])
                            {
                                if (LinuxAtomic::write_min(&vertexValue_host_[dest], msg))

                                {
                                    bool is_visited = (visitedBitset_host_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                    if (!is_visited)
                                    {
                                        int new_val = 1 << (dest % INT_SIZE);
                                        uint32_t old_val = visitedBitset_host_[dest / INT_SIZE].fetch_or(new_val, ATOMIC_ORDER);
                                        bool is_notSink = (notSinkBitset_host_[dest / INT_SIZE] >> (dest % INT_SIZE)) & 1;
                                        if (is_notSink)
                                        {
                                            bool atomic_was_visited = (old_val >> (dest % INT_SIZE)) & 1;
                                            if (!atomic_was_visited)
                                            {
                                                frontierOut_thread_host_[threadId][frontierCounter_thread] = dest;
                                                frontierCounter_thread++;

                                                if (frontierCounter_thread == HOST_THREAD_FLUSH_VALUE)
                                                {
                                                    count_type old_index = frontierNum_out_host_.fetch_add(frontierCounter_thread, ATOMIC_ORDER);
                                                    std::memcpy(frontier_out_host_ + old_index, frontierOut_thread_host_[threadId],
                                                                frontierCounter_thread * sizeof(vertex_id_type));
                                                    frontierCounter_thread = 0;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (frontierCounter_thread > 0)
            {
                count_type old_index = frontierNum_out_host_.fetch_add(frontierCounter_thread, ATOMIC_ORDER);
                std::memcpy(frontier_out_host_ + old_index, frontierOut_thread_host_[threadId], frontierCounter_thread * sizeof(vertex_id_type));
            }
        } // end of [parallel for]

    } // namespace CPJ

    count_type getVisitedIncAndSortedFrontierOut_Opt_CPUGPU()
    {
        count_type activeVertexNum = 0;
        memset(frontierAppend_thread_host_, 0, sizeof(count_type) * frontierAppendNum_thread_host_);
        const count_type chunkNum_avg = visitedBitsetNum_host_ / frontierAppendNum_thread_host_;

        omp_par
        {
            const int threadId = omp_get_thread_num();
            const count_type stratIndex_thread = threadId * chunkNum_frontierAppend_eachThread_host_;

            for (count_type index = 0; index < chunkNum_frontierAppend_eachThread_host_; index++)
            {
                const count_type chunId_first = chunkNum_avg * (stratIndex_thread + index);
                const count_type chunkId_last = ((stratIndex_thread + index) == (frontierAppendNum_thread_host_ - 1))
                                                    ? visitedBitsetNum_host_
                                                    : (((stratIndex_thread + index) + 1) * chunkNum_avg);
                for (count_type chunk_id = chunId_first; chunk_id < chunkId_last; chunk_id++)
                {
                    if (visitedBitset_host_[chunk_id] != 0)
                    {
                        if (visitedBitset_host_[chunk_id] == std::numeric_limits<uint32_t>::max())
                        {
                            frontierAppend_thread_host_[stratIndex_thread + index] += INT_SIZE;
                        }
                        else if (visitedBitset_host_[chunk_id] != 0)
                        {
                            uint32_t temp = visitedBitset_host_[chunk_id];
                            frontierAppend_thread_host_[stratIndex_thread + index] += CPJ::Bits::popcount_folly(temp);
                        }
                    }
                }
            }

#pragma omp barrier

#pragma omp single
            {
                activeVertexNum =
                    CPJ::SCAN::HOST::exclusive_scan_std(frontierAppend_thread_host_, frontierAppendNum_thread_host_, frontierAppend_thread_host_);
            }

            count_type startIndex_thread = 0;
            if (activeVertexNum != 0)
            {
                const count_type avg_thread = activeVertexNum / ThreadNum;
                const count_type vertexStartIndex_expect_thread = threadId * avg_thread;
                const count_type vertexEndIndex_expect_thread = ((threadId == (ThreadNum - 1)) ? activeVertexNum : (threadId + 1) * avg_thread) - 1;

                const count_type start_balance_thread =
                    std::distance(frontierAppend_thread_host_,
                                  std::upper_bound(frontierAppend_thread_host_, frontierAppend_thread_host_ + frontierAppendNum_thread_host_,
                                                   vertexStartIndex_expect_thread));
                const count_type end_balance_thread =
                    std::distance(frontierAppend_thread_host_,
                                  std::upper_bound(frontierAppend_thread_host_, frontierAppend_thread_host_ + frontierAppendNum_thread_host_,
                                                   vertexEndIndex_expect_thread + 1));

                count_type firstVertexIndex_thread = start_balance_thread - 1;
                count_type lastVertexIndex_thread;
                if (vertexEndIndex_expect_thread < (activeVertexNum - 1))
                {
                    lastVertexIndex_thread = end_balance_thread - 1;
                    if (frontierAppend_thread_host_[lastVertexIndex_thread] == (vertexEndIndex_expect_thread + 1)) lastVertexIndex_thread--;
                }
                else
                {
                    lastVertexIndex_thread = frontierAppendNum_thread_host_ - 1;
                }

                const degree_type firstNbrIndex_InFirstVertex_thread =
                    vertexStartIndex_expect_thread - frontierAppend_thread_host_[firstVertexIndex_thread];

                if (firstNbrIndex_InFirstVertex_thread != 0)
                {
                    firstVertexIndex_thread++;
                }

                startIndex_thread = frontierAppend_thread_host_[firstVertexIndex_thread];

                const count_type chunId_first_in =
                    (firstVertexIndex_thread / chunkNum_frontierAppend_eachThread_host_ * chunkNum_frontierAppend_eachThread_host_ +
                     (firstVertexIndex_thread % chunkNum_frontierAppend_eachThread_host_)) *
                    chunkNum_avg;

                const count_type chunkId_last_in =
                    (lastVertexIndex_thread == (frontierAppendNum_thread_host_ - 1))
                        ? (visitedBitsetNum_host_ - 1)
                        : (((lastVertexIndex_thread + 1) / chunkNum_frontierAppend_eachThread_host_ * chunkNum_frontierAppend_eachThread_host_ +
                            ((lastVertexIndex_thread + 1) % chunkNum_frontierAppend_eachThread_host_)) *
                               chunkNum_avg -
                           1);

                for (count_type chunk_id = chunId_first_in; chunk_id <= chunkId_last_in; chunk_id++)
                {
                    uint32_t word32 = visitedBitset_host_[chunk_id];
                    vertex_id_type vertex_id_start = chunk_id * INT_SIZE;
                    while (word32 != 0)
                    {
                        if (word32 & 1)
                        {
                            frontier_out_host_[startIndex_thread] = vertex_id_start;
                            vertexValue_host_[vertex_id_start] =
                                std::min(vertexValue_host_[vertex_id_start], vertexValue_temp_host_[vertex_id_start]);
                            startIndex_thread++;
                        }
                        vertex_id_start++;
                        word32 = word32 >> 1;
                    }
                }
            }
        }

        return activeVertexNum;
    }

    count_type getVisitedIncAndSortedFrontierOut_Opt()
    {
        count_type activeVertexNum = 0;
        memset(frontierAppend_thread_host_, 0, sizeof(count_type) * frontierAppendNum_thread_host_);
        const count_type chunkNum_avg = visitedBitsetNum_host_ / frontierAppendNum_thread_host_;

        omp_par
        {
            const int threadId = omp_get_thread_num();
            const count_type stratIndex_thread = threadId * chunkNum_frontierAppend_eachThread_host_;

            /* 我们处理: sink + popcount */
            for (count_type index = 0; index < chunkNum_frontierAppend_eachThread_host_; index++)
            {
                const count_type chunId_first = chunkNum_avg * (stratIndex_thread + index);
                const count_type chunkId_last = ((stratIndex_thread + index) == (frontierAppendNum_thread_host_ - 1))
                                                    ? visitedBitsetNum_host_
                                                    : (((stratIndex_thread + index) + 1) * chunkNum_avg);
                for (count_type chunk_id = chunId_first; chunk_id < chunkId_last; chunk_id++)
                {
                    if (visitedBitset_host_[chunk_id] != 0)
                    {
                        if (visitedBitset_host_[chunk_id] == std::numeric_limits<uint32_t>::max())
                        {
                            frontierAppend_thread_host_[stratIndex_thread + index] += INT_SIZE;
                        }
                        else if (visitedBitset_host_[chunk_id] != 0)
                        {
                            uint32_t temp = visitedBitset_host_[chunk_id];
                            frontierAppend_thread_host_[stratIndex_thread + index] += CPJ::Bits::popcount_folly(temp);
                        }
                    }
                }
            }

#pragma omp barrier

#pragma omp single
            {
                activeVertexNum =
                    CPJ::SCAN::HOST::exclusive_scan_std(frontierAppend_thread_host_, frontierAppendNum_thread_host_, frontierAppend_thread_host_);
            }

            count_type startIndex_thread = 0;
            if (activeVertexNum != 0)
            {
                const count_type avg_thread = activeVertexNum / ThreadNum;
                const count_type vertexStartIndex_expect_thread = threadId * avg_thread;
                const count_type vertexEndIndex_expect_thread = ((threadId == (ThreadNum - 1)) ? activeVertexNum : (threadId + 1) * avg_thread) - 1;

                const count_type start_balance_thread =
                    std::distance(frontierAppend_thread_host_,
                                  std::upper_bound(frontierAppend_thread_host_, frontierAppend_thread_host_ + frontierAppendNum_thread_host_,
                                                   vertexStartIndex_expect_thread));
                const count_type end_balance_thread =
                    std::distance(frontierAppend_thread_host_,
                                  std::upper_bound(frontierAppend_thread_host_, frontierAppend_thread_host_ + frontierAppendNum_thread_host_,
                                                   vertexEndIndex_expect_thread + 1));

                count_type firstVertexIndex_thread = start_balance_thread - 1;
                count_type lastVertexIndex_thread;
                if (vertexEndIndex_expect_thread < (activeVertexNum - 1))
                {
                    lastVertexIndex_thread = end_balance_thread - 1;
                    if (frontierAppend_thread_host_[lastVertexIndex_thread] == (vertexEndIndex_expect_thread + 1)) lastVertexIndex_thread--;
                }
                else
                {
                    lastVertexIndex_thread = frontierAppendNum_thread_host_ - 1;
                }

                const degree_type firstNbrIndex_InFirstVertex_thread =
                    vertexStartIndex_expect_thread - frontierAppend_thread_host_[firstVertexIndex_thread];

                if (firstNbrIndex_InFirstVertex_thread != 0)
                {
                    firstVertexIndex_thread++;
                }

                startIndex_thread = frontierAppend_thread_host_[firstVertexIndex_thread];

                const count_type chunId_first_in =
                    (firstVertexIndex_thread / chunkNum_frontierAppend_eachThread_host_ * chunkNum_frontierAppend_eachThread_host_ +
                     (firstVertexIndex_thread % chunkNum_frontierAppend_eachThread_host_)) *
                    chunkNum_avg;

                const count_type chunkId_last_in =
                    (lastVertexIndex_thread == (frontierAppendNum_thread_host_ - 1))
                        ? (visitedBitsetNum_host_ - 1)
                        : (((lastVertexIndex_thread + 1) / chunkNum_frontierAppend_eachThread_host_ * chunkNum_frontierAppend_eachThread_host_ +
                            ((lastVertexIndex_thread + 1) % chunkNum_frontierAppend_eachThread_host_)) *
                               chunkNum_avg -
                           1);

                for (count_type chunk_id = chunId_first_in; chunk_id <= chunkId_last_in; chunk_id++)
                {
                    uint32_t word32 = visitedBitset_host_[chunk_id];
                    vertex_id_type vertex_id_start = chunk_id * INT_SIZE;
                    if (word32 == std::numeric_limits<uint32_t>::max())
                    {
                        std::iota(frontier_out_host_ + startIndex_thread, frontier_out_host_ + startIndex_thread + INT_SIZE, vertex_id_start);
                        startIndex_thread += INT_SIZE;
                    }
                    else if (word32 == 0x80000000)
                    {
                        vertex_id_type vertexId = vertex_id_start + 31;
                        frontier_out_host_[startIndex_thread] = vertexId;
                        startIndex_thread++;
                    }
                    else
                    {
                        while (word32 != 0)
                        {
                            if (word32 & 1)
                            {
                                frontier_out_host_[startIndex_thread] = vertex_id_start;
                                startIndex_thread++;
                            }
                            vertex_id_start++;
                            word32 = word32 >> 1;
                        }
                    }
                }
            }
        }

        return activeVertexNum;
    }

    inline std::tuple<count_type, count_type, countl_type, count_type, countl_type> balance_CPUGPU(const count_type activeVertexNum,
                                                                                                   const countl_type activeEdgeNum)
    {
        float cur_final_ratio = final_ratio_;
        count_type activeVertexNum_host_final{0};
        countl_type activeEdgeNum_host_final{0};
        count_type activeVertexNum_device_final{0};
        countl_type activeEdgeNum_device_final{0};

        countl_type activeEdge_forDevice_est = cur_final_ratio * activeEdgeNum;
        count_type cutVertexNum_forDevice_est =
            std::distance(frontier_degExSum_host_, thrust::upper_bound(thrust::host, frontier_degExSum_host_,
                                                                       frontier_degExSum_host_ + activeVertexNum, activeEdge_forDevice_est)) -
            1;
        assert_msg((cutVertexNum_forDevice_est < activeVertexNum) && (cutVertexNum_forDevice_est >= 1),
                   "cutVertexNum_forDevice_est = %zu, activeVertexNum = %zu", SCU64(cutVertexNum_forDevice_est), SCU64(activeVertexNum));
        assert_msg(frontier_degExSum_host_[cutVertexNum_forDevice_est] <= activeEdge_forDevice_est, "Error in find CG_Balance");

        vertex_id_type cutVertexId_forDevice_est = frontier_in_host_[cutVertexNum_forDevice_est - 1];
        if (cutVertexId_forDevice_est <= cutVertexId_device_)
        {
            activeEdgeNum_device_final =
                (cutVertexNum_forDevice_est != activeVertexNum) ? frontier_degExSum_host_[cutVertexNum_forDevice_est] : activeEdgeNum;
            activeEdgeNum_host_final = activeEdgeNum - activeEdgeNum_device_final;

            activeVertexNum_device_final = (cutVertexNum_forDevice_est != activeVertexNum) ? (cutVertexNum_forDevice_est) : activeVertexNum;
            activeVertexNum_host_final = activeVertexNum - activeVertexNum_device_final;

            // assert_msg(activeEdgeNum_device_final <= activeEdge_forDevice_est, "activeEdgeNum_device_final = %zu, activeEdge_forDevice_est = %zu",
            //            SCU64(activeEdgeNum_device_final), SCU64(activeEdge_forDevice_est));
            // assert_msg(activeEdgeNum_device_final <= activeEdgeNum, "activeEdgeNum_device_final = %zu, activeEdgeNum = %zu",
            //            SCU64(activeEdgeNum_device_final), SCU64(activeEdgeNum));
            // assert_msg(activeEdgeNum_host_final <= activeEdgeNum, "activeEdgeNum_host_final = %zu, activeEdgeNum = %zu",
            //            SCU64(activeEdgeNum_host_final), SCU64(activeEdgeNum));
            // assert_msg(activeVertexNum_device_final <= activeVertexNum, "activeVertexNum_device_final = %zu, activeVertexNum = %zu",
            //            SCU64(activeVertexNum_device_final), SCU64(activeVertexNum));
            // assert_msg(activeVertexNum_host_final <= activeVertexNum, "activeVertexNum_host_final = %zu, activeVertexNum = %zu",
            //            SCU64(activeVertexNum_host_final), SCU64(activeVertexNum));
        }
        else
        {
            activeVertexNum_device_final = std::distance( //! activeVertexNum <-> cutVertexNum_forDevice_est
                frontier_in_host_,
                thrust::upper_bound(thrust::host, frontier_in_host_, frontier_in_host_ + cutVertexNum_forDevice_est, cutVertexId_device_));
            assert_msg(activeVertexNum_device_final < cutVertexNum_forDevice_est,
                       "activeVertexNum_device_final = %zu, cutVertexNum_forDevice_est = %zu", SCU64(activeVertexNum_device_final),
                       SCU64(cutVertexNum_forDevice_est));
            // vertex_id_type cutVertexIndex_forDevice = activeVertexNum_device_final - 1;

            // assert_msg(frontier_in_host_[cutVertexIndex_forDevice] <= cutVertexId_device_,
            //            "左边应该小于, activeVertexNum_device_final = %zu, cutVertexIndex_forDevice = %zu, "
            //            "frontier_in_host_[cutVertexIndex_forDevice] = %zu, "
            //            "cutVertexId_device_ = %zu",
            //            SCU64(activeVertexNum_device_final), SCU64(cutVertexIndex_forDevice), SCU64(frontier_in_host_[cutVertexIndex_forDevice]),
            //            SCU64(cutVertexId_device_));
            // assert_msg(frontier_in_host_[activeVertexNum_device_final] > cutVertexId_device_,
            //            "右边应该大于, activeVertexNum_device_final = %zu,  frontier_in_host_[activeVertexNum_device_final] = %zu, "
            //            "cutVertexId_device_ = %zu",
            //            SCU64(activeVertexNum_device_final), SCU64(frontier_in_host_[activeVertexNum_device_final]), SCU64(cutVertexId_device_));

            activeEdgeNum_device_final = frontier_degExSum_host_[activeVertexNum_device_final];
            activeEdgeNum_host_final = activeEdgeNum - activeEdgeNum_device_final;
            activeVertexNum_host_final = activeVertexNum - activeVertexNum_device_final;

            // assert_msg(activeEdgeNum_device_final <= activeEdge_forDevice_est, "activeEdgeNum_device_final = %zu, activeEdge_forDevice_est = %zu",
            //            SCU64(activeEdgeNum_device_final), SCU64(activeEdge_forDevice_est));
            // assert_msg(activeEdgeNum_device_final <= activeEdgeNum, "activeEdgeNum_device_final = %zu, activeEdgeNum = %zu",
            //            SCU64(activeEdgeNum_device_final), SCU64(activeEdgeNum));
            // assert_msg(activeEdgeNum_host_final <= activeEdgeNum, "activeEdgeNum_host_final = %zu, activeEdgeNum = %zu",
            //            SCU64(activeEdgeNum_host_final), SCU64(activeEdgeNum));
            // assert_msg(activeVertexNum_device_final <= activeVertexNum, "activeVertexNum_device_final = %zu, activeVertexNum = %zu",
            //            SCU64(activeVertexNum_device_final), SCU64(activeVertexNum));
            // assert_msg(activeVertexNum_host_final <= activeVertexNum, "activeVertexNum_host_final = %zu, activeVertexNum = %zu",
            //            SCU64(activeVertexNum_host_final), SCU64(activeVertexNum));
        }

        return std::make_tuple(activeVertexNum_device_final, activeVertexNum_host_final, activeEdgeNum_host_final, activeVertexNum_device_final,
                               activeEdgeNum_device_final);
    }

    inline std::tuple<count_type, count_type, countl_type, count_type, countl_type> balance_CPUGPU_old(const count_type activeVertexNum,
                                                                                                       const countl_type activeEdgeNum)
    {
        float cur_final_ratio = final_ratio_;
        count_type activeVertexNum_host_final{0};
        countl_type activeEdgeNum_host_final{0};
        count_type activeVertexNum_device_final{0};
        countl_type activeEdgeNum_device_final{0};

        count_type cutVertexNum{0};

        double delta_host{0.0};
        double delta_device{0.0};

        double avgDegree_host{0.0};
        double avgDegree_device{0.0};

        const int tryNum = 1;
        for (int try_index = 0; try_index < tryNum; try_index++)
        {
            countl_type activeEdge_device = cur_final_ratio * activeEdgeNum;

            if (CGModel_lastIte_ == CGModel::GPU_ONLY)
            {
                assert_msg(false, "不应该出现");
                auto cutItor =
                    thrust::upper_bound(thrust::device, frontier_degExSum_device_, frontier_degExSum_device_ + activeVertexNum, activeEdge_device);
                int64_t uppder_dis = thrust::distance(frontier_degExSum_device_, cutItor);
                assert_msg(uppder_dis >= 2, "a...");
                cutVertexNum = uppder_dis - 1;
            }
            else
            {
                auto cutItor =
                    thrust::upper_bound(thrust::host, frontier_degExSum_host_, frontier_degExSum_host_ + activeVertexNum, activeEdge_device);
                int64_t uppder_dis = thrust::distance(frontier_degExSum_host_, cutItor);
                assert_msg(uppder_dis >= 2, "a..."); // 如果成立则cutVertexNum = 0
                cutVertexNum = uppder_dis - 1;
                assert_msg(frontier_degExSum_host_[cutVertexNum] <= activeEdge_device, "Error in find CG_Balance");
            }

            vertex_id_type cut_vertex_id = frontier_in_host_[cutVertexNum - 1]; // frontier_in_host_必须是有序的

            if (cut_vertex_id <= cutVertexId_device_)
            {
                activeEdgeNum_device_final = (cutVertexNum != activeVertexNum) ? frontier_degExSum_host_[cutVertexNum] : activeEdgeNum;
                activeEdgeNum_host_final = activeEdgeNum - activeEdgeNum_device_final;
                activeVertexNum_device_final = (cutVertexNum != activeVertexNum) ? (cutVertexNum) : activeVertexNum;
                activeVertexNum_host_final = activeVertexNum - activeVertexNum_device_final;
            }
            else
            {
                auto cutItor = thrust::upper_bound(thrust::host, frontier_in_host_, frontier_in_host_ + activeVertexNum, cutVertexId_device_);
                int64_t uppder_dis = thrust::distance(frontier_in_host_, cutItor);
                count_type num = uppder_dis;
                // vertex_id_type cut_vertex_index = num - 1;
                // assert_msg(frontier_in_host_[cut_vertex_index] <= cutVertexId_device_,
                //            "左边应该小于, cut_vertex_num = %zu, cut_vertex_index = %zu, frontier_in_host_[cut_vertex_index] = %zu, "
                //            "cutVertexId_device_ = %zu",
                //            SCU64(num), SCU64(cut_vertex_index), SCU64(frontier_in_host_[cut_vertex_index]), SCU64(cutVertexId_device_)); //!

                // assert_msg(frontier_in_host_[num] > cutVertexId_device_,
                //            "右边应该大于, cut_vertex_num = %zu,  frontier_in_host_[cut_vertex_num] = %zu, "
                //            "cutVertexId_device_ = %zu",
                //            SCU64(num), SCU64(frontier_in_host_[num]), SCU64(cutVertexId_device_));

                activeEdgeNum_device_final = frontier_degExSum_host_[num];
                activeEdgeNum_host_final = activeEdgeNum - activeEdgeNum_device_final;
                activeVertexNum_device_final = num;
                activeVertexNum_host_final = activeVertexNum - activeVertexNum_device_final;
                // assert_msg(activeEdgeNum_device_final < activeEdge_device,
                //            "activeEdgeNum_device_final = %zu, activeEdge_device = %zu, uppder_dis = %zu", SCU64(activeEdgeNum_device_final),
                //            SCU64(activeEdge_device), SCU64(uppder_dis));
                // assert_msg(activeVertexNum_device_final < cutVertexNum, "Error...");
                cutVertexNum = num;
            }
            assert_msg(frontier_in_host_[cutVertexNum - 1] <= cutVertexId_device_,
                       "frontier_in_host_[cutVertexNum - 1] = %zu, cutVertexId_device_ = %zu", SCU64(frontier_in_host_[cutVertexNum - 1]),
                       SCU64(cutVertexId_device_));

            avgDegree_host = SCD(activeEdgeNum_host_final) / SCD(activeVertexNum_host_final);
            avgDegree_device = SCD(activeEdgeNum_device_final) / SCD(activeVertexNum_device_final);

            countl_type delta = (activeEdgeNum_device_final > activeEdgeNum_host_final) ? (activeEdgeNum_device_final - activeEdgeNum_host_final)
                                                                                        : (activeEdgeNum_host_final - activeEdgeNum_device_final);

            const count_type TH_BALANCE = 100000000;
            if (delta < TH_BALANCE)
            {
                if constexpr (TEMP_TIME_DEBUG)
                {
                    Log_info("\tIte[%d]: [Sat] avgDegree_host = %.2lf, avgDegree_device = %.2lf, delta_host = %.2lf, delta_device = %.2lf", ite_,
                             avgDegree_host, avgDegree_device, delta_host, delta_device);
                    Log_info("\tIte[%d]: [Sat] ratio = %.2lf, totalV = %zu, totalE = %zu, totalHV = %zu, totalHE = %zu, totalDV = %zu, totalDE = %zu",
                             ite_, cur_final_ratio, SCU64(activeVertexNum), SCU64(activeEdgeNum), SCU64(activeVertexNum_host_final),
                             SCU64(activeEdgeNum_host_final), SCU64(activeVertexNum_device_final), SCU64(activeEdgeNum_device_final));
                }
                return std::make_tuple(cutVertexNum, activeVertexNum_host_final, activeEdgeNum_host_final, activeVertexNum_device_final,
                                       activeEdgeNum_device_final);
            }
            else
            {
                if (try_index == tryNum - 1)
                {
                    break;
                }

                delta_host = std::abs(avgDegree_host - noSink_avgDegree_);
                delta_device = std::abs(avgDegree_device - noSink_avgDegree_);

                countl_type delta_r = delta / 100000000;
                double try_r = std::pow(delta_r, 3) * 0.0023;
                /* GPU 比较慢*/
                if (delta_device > delta_host)
                {
                    assert_msg(final_ratio_ > try_r, "error ratio");
                    // cur_final_ratio = final_ratio - /*(delta_device - delta_host) * */ (delta_r * try_r);
                    cur_final_ratio = final_ratio_ - try_r;
                    // Msg_check("\tIte[%d]: [new-if]: cur_final_ratio = %.2lf, delta_device = %.2lf, delta_host = %.2lf, delta_r = %u", ite_,
                    //           cur_final_ratio, delta_device, delta_host, delta_r);
                }
                /* CPU 比较慢*/
                else
                {
                    cur_final_ratio = final_ratio_ + try_r;
                    assert_msg(cur_final_ratio < 1, "error ratio");
                    // Msg_check("\tIte[%d]: [new-else]: cur_final_ratio = %.2lf, delta_device = %.2lf, delta_host = %.2lf, delta_r = %u", ite_,
                    //           cur_final_ratio, delta_device, delta_host, delta_r);
                }
            }
        }
        if constexpr (TEMP_TIME_DEBUG)
        {
            Log_info("\tIte[%d]: [Sat] avgDegree_host = %.2lf, avgDegree_device = %.2lf, delta_host = %.2lf, delta_device = %.2lf", ite_,
                     avgDegree_host, avgDegree_device, delta_host, delta_device);
            Log_info("\tIte[%d]: [Sat] ratio = %.2lf, totalV = %zu, totalE = %zu, totalHV = %zu, totalHE = %zu, totalDV = %zu, totalDE = %zu", ite_,
                     cur_final_ratio, SCU64(activeVertexNum), SCU64(activeEdgeNum), SCU64(activeVertexNum_host_final),
                     SCU64(activeEdgeNum_host_final), SCU64(activeVertexNum_device_final), SCU64(activeEdgeNum_device_final));
        }
        return std::make_tuple(cutVertexNum, activeVertexNum_host_final, activeEdgeNum_host_final, activeVertexNum_device_final,
                               activeEdgeNum_device_final);
    }

    //! 此函数只有CPU_GPU 模式下才会进入
    void balance_model_device_inCPUGPU(const count_type activeVertexNum_device_final, const countl_type activeEdgeNum_device_final)
    {
        /* GPU Balance */
        CPJ::Timer tempTime;

        if (CGModel_lastIte_ == CGModel::CPU_GPU)
        {
            int numBlocks = (vertexNum_host_ + BLOCKSIZE_2 - 1) / BLOCKSIZE_2;
            SSSP_DEVICE_SPACE::merge_vertexValue_kernel<<<numBlocks, BLOCKSIZE_2, 0, stream>>>(vertexValue_device_, vertexValue_device2_,
                                                                                               vertexNum_host_, notSinkBitset_device_);
        }

        CUDA_CHECK(H2D(frontier_in_device_, frontier_in_host_, activeVertexNum_device_final));
        CUDA_CHECK(H2D(frontier_degExSum_device_, frontier_degExSum_host_, activeVertexNum_device_final));
        nBlock_ = (activeEdgeNum_device_final + TASK_PER_BLOCK - 1) / TASK_PER_BLOCK;

        thrust::counting_iterator<countl_type> cnt_iter(0);
        auto query_iter_first = thrust::make_transform_iterator(cnt_iter, Device::Set_blockTask_type());
        auto query_iter_last = thrust::make_transform_iterator(cnt_iter + nBlock_, Device::Set_blockTask_type());

        if (CGModel_lastIte_ == CGModel::CPU_GPU)
        {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::swap(vertexValue_device_, vertexValue_device2_);
        }
        thrust::upper_bound(thrust::device, frontier_degExSum_device_, frontier_degExSum_device_ + activeVertexNum_device_final, query_iter_first,
                            query_iter_last, frontier_balance_device_);

        nBlock_ = std::min(nBlock_, SCU64(MAX_BLOCKS));
        SSSP_DEVICE_SPACE::balance_model_CPUGPU(
            nBlock_, vertexValue_device_, csr_offset_device_, csr_dest_device_, csr_weight_device_, csr_destWeight_device_, frontier_in_device_,
            frontier_out_device_, activeVertexNum_device_final, activeEdgeNum_device_final, frontierNum_out_device_, frontier_degExSum_device_,
            frontier_balance_device_, notSinkBitset_device_, reinterpret_cast<int*>(visitedBitset_device_), SSSP_destWeight_);

        if (!USE_DEVICE_FOR_GET_FRONTIER)
        {
            CUDA_CHECK(D2H(visitedBitset_storeTempDevice_host_, visitedBitset_device_, visitedBitsetNum_host_));
            CUDA_CHECK(D2H(vertexValue_temp_host_, vertexValue_device_, vertexNum_host_));
        }
    }

}; // end of namespace [CG_SSSP]

} // namespace CPJ