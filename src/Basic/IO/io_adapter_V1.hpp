/* *********************************************************************************************************
 * @Author: pjCui
 * @Date: 2024-04-28 14:20:00
 * @LastEditTime: 2024-04-28 15:40:17
 * @FilePath: /2024-4-2/src/Basic/IO/io_adapter_V1.hpp
 * @Description: We have compactly implemented some functions in io_adapter.hpp
 * *********************************************************************************************************/

#pragma once

#include "Basic/Console/console_bar.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/IO/file_permission.hpp"
#include "Basic/Memory/hugePage.hpp"
#include "Basic/Other/errnoMsg.hpp"
#include "Basic/Other/fileSystem_CPJ.hpp"
#include "Basic/Timer/timer_CPJ.hpp"
#include <fcntl.h>
#include <functional>
#include <libgen.h>
#include <linux/fs.h>
#include <numeric>
#include <string>
#include <sys/ioctl.h>
#include <sys/stat.h>

namespace CPJ {
class IOAdaptor
{
  private:
    std::string filePath_;
    std::string fileName_;
    int fd_;
    ssize_t totalBytes_;

    static constexpr size_t SUPPORT_LINE_SIZE = 65535;
    static constexpr size_t SYNC_CHUNK_SIZE = 1 << 20; // 1 << 20 = 1048576 (4KB * 256)
    static constexpr size_t ASYNC_IO_PAGE_SIZE_POWER = 16;
    static constexpr size_t ASYNC_K_PAGESIZE = 1 << ASYNC_IO_PAGE_SIZE_POWER; // 1 << 16 = 65536 (64KB); 65536 / 4096 = 16
    static constexpr bool DEBUG = false;
    static constexpr bool USE_BAR = false;
    int threadId_forBar_{0};

  public:
    enum class File_type { BIN, NOT_BIN, UNCERTAIN };
    using DetermineFile_funcType = std::function<File_type(const char&)>;

  public:
    IOAdaptor() = delete;
    IOAdaptor(const std::string& filePath) : filePath_(filePath), fd_(-1), totalBytes_(-1) { fileName_ = CPJ::FS::getFileName(filePath); }
    IOAdaptor(IOAdaptor&& other)
        : filePath_{std::exchange(other.filePath_, {})}, fileName_(std::exchange(other.fileName_, {})), fd_{std::exchange(other.fd_, -1)},
          totalBytes_{std::exchange(other.totalBytes_, -1)}
    {
    }

    bool isFileOpen() const
    {
        bool isOpen = (fd_ >= 0);
        return isOpen;
    }

    void closeFile()
    {
        if (isFileOpen()) close(fd_);
    }

    ~IOAdaptor()
    {
        if (isFileOpen()) close(fd_);
    }

    /* *********************************************************************************************************
     * @description:
     * @param [char*] mode
     *                1. O_RDONLY: 只读打开文件;   2. O_WRONLY: 只写打开文件;    3. O_RDWR: 读写打开文件;
     *                4. O_CREAT: 如果文件不存在则创建文件; 5.O_TRUNC: 如果文件已经存在并且是写操作，将其长度截断为0;
     *                6. O_APPEND： 在写操作时，将数据追加到文件末尾; 7. O_EXCL:与 O_CREAT 一起使用时，如果文件已经存在，则会失败
     *                7. O_NONBLOCK： 非阻塞模式打开文件; 8. O_SYNC： 同步写入，要求每次写操作都立即写入到物理介质中;
     *                9. O_DIRECT： 绕过缓冲区，直接访问物理磁盘; 10. O_NOFOLLOW： 如果路径为符号链接，则打开失败;
     *                11. O_DIRECTORY： 如果路径不是目录，则打开失败。
     * @return [*]
     * *********************************************************************************************************/
    void openFile(const char* mode = "r")
    {
        std::string extension = CPJ::FS::getFileExtension(filePath_);
        assert_msg(extension != ".gz", "Invalid operation, not support Open (.gz) type file");
        assert_msg(extension != ".zip", "Invalid operation, not support Open (.zip) type file");

        // ^ 表示二进制模式
        if (strchr(mode, 'b') != NULL)
        {
            CPJ::FS::createDirectory(filePath_);
            fd_ = open(filePath_.c_str(), O_CREAT | O_RDWR, OWNER_READ_WRITE);
        }
        // ^ 表示以追加模式打开文件
        else if (strchr(mode, 'a') != NULL)
        {
            CPJ::FS::createDirectory(filePath_);
            fd_ = open(filePath_.c_str(), O_CREAT | O_RDWR | O_APPEND, OWNER_READ_WRITE);
        }
        // ^ 如果文件已经存在，会清空文件内容，然后写入新的数据。如果文件不存在，则创建新文件并写入数据。
        else if (strchr(mode, 'w') != NULL || strchr(mode, '+') != NULL)
        {
            CPJ::FS::createDirectory(filePath_);
            fd_ = open(filePath_.c_str(), O_CREAT | O_RDWR | O_TRUNC, OWNER_READ_WRITE);
        }
        // ^ 表示以只读模式打开文件
        else if (strchr(mode, 'r') != NULL)
        {
            fd_ = open(filePath_.c_str(), O_RDONLY);
        }
        assert_msg(fd_ >= 0, "Can Not Open File [%s] with mode('%s'), fd_ = %d, errMsg = %s", filePath_.c_str(), mode, fd_,
                   getErrorMsg(errno).c_str());
        if constexpr (DEBUG) Msg_info("Open File [%s] with mode('%s'), fd_ = %d", filePath_.c_str(), mode, fd_);
    }

    /* *********************************************************************************************************
     * @description: 判断一个文件是否存在
     * @return [*]
     * *********************************************************************************************************/
    bool isFileExist()
    {
        std::filesystem::path filePath_fs(filePath_);
        return std::filesystem::exists(filePath_fs);
    }

    /* *********************************************************************************************************
     * @description: 返回当前读取位置的偏移量 (相对于文件头)
     *               lseek 调用成功，返回值为新的文件偏移量，表示文件位置指针相对于[文件开头]的偏移量
     *               lseek 调用失败，返回值为-1，并设置errno为相应的错误码
     * @return [*]
     * *********************************************************************************************************/
    off_t getFileCurOffset()
    {
        assert_msg(fd_ != -1, "Please open File [%s] first, then call getFileCurOffset()", fileName_.c_str());
        off_t offset_return;
        offset_return = lseek(fd_, 0, SEEK_CUR);
        assert_msg(offset_return != -1, "Fail to get current File[%s] offset, errMsg = %s", fileName_.c_str(), getErrorMsg(errno).c_str());
        return offset_return;
    }

    /* *********************************************************************************************************
     * @description: 设置文件的读取位置
     *               lseek 调用成功，返回值为新的文件偏移量，表示文件位置指针相对于[文件开头]的偏移量
     *               lseek 调用失败，返回值为-1，并设置errno为相应的错误码
     * @return [*]
     * *********************************************************************************************************/
    void setFileCurOffset(const off_t offset, const int seek_from)
    {
        assert_msg(fd_ != -1, "Please open File [%s] first, then call setFileCurOffset()", fileName_.c_str());
        assert_msg((seek_from >= 0) && (seek_from <= 2), "Invalid <seek_from = %d>, which must set to: SEEK_SET or SEEK_CUR or SEEK_END", seek_from);

        off_t offset_return;
        offset_return = lseek(fd_, offset, seek_from);
        assert_msg(offset_return != -1, "Fail to set current File[%s] offset, errMsg = %s", fileName_.c_str(), getErrorMsg(errno).c_str());
    }

    /* *********************************************************************************************************
     * @description: 将文件文件的读取指针指向文件首部
     * @return [*]
     * *********************************************************************************************************/
    void setFileCurOffset_toBegin() { setFileCurOffset(0, SEEK_SET); }

    /* *********************************************************************************************************
     * @description: 将文件文件的读取指针指向文件尾部
     * @return [*]
     * *********************************************************************************************************/
    void setFileCurOffset_toEnd() { setFileCurOffset(0, SEEK_END); }

    /* *********************************************************************************************************
     * @description: 返回文件的总Bytes
     * @param [bool] runtime 如果设置为true, 则实时获取不会使用缓存，并更新totalBytes_
     * @return [*]
     * *********************************************************************************************************/
    ssize_t getFileSize(bool runtime = false)
    {
        if ((totalBytes_ >= 0) && (!runtime))
        {
            return totalBytes_;
        }
        else
        {
            assert_msg(fd_ != -1, "Please open File [%s] first, then call getFileSize()", fileName_.c_str());

            struct stat st;
            assert_msg(fstat(fd_, &st) >= 0, "File[%s]: fstat error, errMsg = %s", filePath_.c_str(), getErrorMsg(errno).c_str());

            /* 检查是否为快设备文件.*/
            /* 块特殊文件（block special file）是一种特殊类型的文件，提供特殊文件是为了使 I/O 设备看起来像文件一般。块设备是一种存储数据的设备，*/
            /* 例如硬盘驱动器、固态硬盘（SSD）和光驱等。块特殊文件允许用户访问和操作这些设备，就像访问普通文件一样。*/
            if (S_ISBLK(st.st_mode))
            {
                assert_msg(ioctl(fd_, BLKGETSIZE64, &totalBytes_) == 0, "Get size of Block special file error");
                return totalBytes_;
            }
            /* 判断是否为普通文件 */
            else if (S_ISREG(st.st_mode))
            {
                // if constexpr (DEBUG) Msg_info("File [%s] total bytes = %zd (bytes)", filePath_.c_str(), totalBytes_);
                totalBytes_ = st.st_size;
                return totalBytes_;
            }
            else
            {
                assert_msg(false, "Get size of file meet unknown file type");
                totalBytes_ = -1;
                return totalBytes_;
            }
        }
    }

    int getFileFd() const
    {
        assert_msg(fd_ != -1, "Please open File [%s] first, then call getCurFileOffset()", fileName_.c_str());
        return fd_;
    }
    const std::string& getfilePath() const { return filePath_; }
    const std::string& getFileName() const { return fileName_; }

    /* *********************************************************************************************************
     * @description: 此处我们根据自己的需求自定义了一个判断文件类型的函数,我们将其定义为静态的，因为它调用对象的任何属性
     *               如果一个文件包含了不可以打印的字符,我们断言这是Bin文件
     *               如果一个文件包含了空白字符, 我们断言这是一个非Bin文件
     *               除上面两种情况外无法判断
     * @param [char&] c 要判断的字符
     * @return [*]
     * *********************************************************************************************************/
    static File_type determineFileType_default(const char& c)
    {
        /* 可打印字符是指在标准ASCII字符集范围内(即ASCII码值从32到126之间)的字符，包括字母、数字、标点符号和一些特殊字符，如空格、换行符、回车符等*/
        if (!std::isprint(static_cast<unsigned char>(c)))
        {
            return File_type::BIN;
        }
        /* 我们的二进制文件中不能拥有空白字符, 默认情况下，以下字符被视为空白字符:*/
        /* 空格(0x20，' '); 换页符(0x0c，'\f'); 换行符(0x0a，'\n'); 回车符(0x0d，'\r'); 水平制表符(0x09，'\t'); 垂直制表符(0x0b，'\v')*/
        else if (std::isspace(static_cast<unsigned char>(c)))
        {
            return File_type::NOT_BIN;
        }
        return File_type::UNCERTAIN;
    }

    /* *********************************************************************************************************
     * @description: 判断一个文件的类型, 我们将文件类型分为了3类: Bin, NOT_BIN 和 UNCERTAIN
     *               支持自定义判断文件类型的决策, 默认采用 @see determineFileType_default(...)
     *               注: 在 DetermineFile_funcType 无法决策时, 我们最终返回File_type::UNCERTAIN是为了兼容性
     *                   但如果一个文件包含全部可打印字符, 可以自己认为是NOT_BIN, 这根据自己的实际需求定
     * @param [DetermineFile_funcType] determineFileType
     * @return [*]
     * *********************************************************************************************************/
    File_type getFileType(DetermineFile_funcType determineFileType = IOAdaptor::determineFileType_default)
    {
        ssize_t totalBytes = getFileSize();
        std::unique_ptr<char[]> buffer(static_cast<char*>(std::aligned_alloc(HugePage::PAGE_SIZE, HugePage::PAGE_SIZE)));
        assert_msg(buffer.get() != nullptr, "The File[%s] meet std::aligned_alloc error", fileName_.c_str());
        ssize_t remainedBytes = totalBytes;
        while (remainedBytes)
        {
            ssize_t readBytes = (remainedBytes >= HugePage::PAGE_SIZE) ? HugePage::PAGE_SIZE : remainedBytes;
            readBinBlock(buffer.get(), readBytes, (totalBytes - remainedBytes));
            remainedBytes -= readBytes;

            for (int char_index = 0; char_index < readBytes; char_index++)
            {
                char c = buffer[char_index];
                File_type fileType = determineFileType(c);
                if (fileType != File_type::UNCERTAIN) return fileType;
            }
        }
        return File_type::UNCERTAIN;
    }

    /* *********************************************************************************************************
     * @description: 支持并行读取当前的文件, 适合于Bin File
     * @param [char*] data 存储读取到文件数据的数组
     * @param [size_t] readBytes 要读取的总Bytes数
     * @param [size_t] offset 读取的文件的偏移量
     * @param [int] threadForBar 用于Bar的线程
     * @return [ssize_t] 最后一次pread的返回
     * *********************************************************************************************************/
    ssize_t readBinBlock(char* data, size_t readBytes, ssize_t offset, int threadForBar = 0) const
    {
        assert_msg(fd_ != -1, "Please open File [%s] first, then call readBlock()", fileName_.c_str());

        size_t barId = 0;
        Bar* bar{nullptr};
        if constexpr (USE_BAR)
        {
            if (threadForBar == threadId_forBar_) bar = new Bar(static_cast<bool>(readBytes >= BAR_SHOWN_BYTES_THRESHOLD));
        }

        ssize_t total_bytes_read = 0ull;
        ssize_t bytes_read = 0;
        while (total_bytes_read < readBytes)
        {
            bytes_read = pread(fd_, data + total_bytes_read, readBytes - total_bytes_read, offset + total_bytes_read);
            assert_msg(
                bytes_read > 0, "File[%s] pread error, bytes_read = %zd, already_bytes_read = %zd, ready_to_read = %zd, offset = %zd, errMsg = %s",
                fileName_.c_str(), bytes_read, total_bytes_read, readBytes - total_bytes_read, offset + total_bytes_read, getErrorMsg(errno).c_str());
            total_bytes_read += bytes_read;

            if constexpr (USE_BAR)
            {
                if ((threadForBar == threadId_forBar_) && (total_bytes_read > (10000 * barId))) //! 这里一定不能bar更新太快, 一定要慢慢的来
                {
                    bar->progress(static_cast<int64_t>(static_cast<double>(total_bytes_read) / readBytes * 100), static_cast<int64_t>(100),
                                  "[readBinBlock]: ");
                    barId++;
                }
            }
        }
        assert_msg(total_bytes_read == readBytes, "File[%s] readBlock error, total_bytes_read = %zd, readBytes = %zu", fileName_.c_str(),
                   total_bytes_read, readBytes);

        if constexpr (USE_BAR)
        {
            if (threadForBar == threadId_forBar_) bar->finish();
        }
        return bytes_read;
    }

    /* *********************************************************************************************************
     * @description: 将指定的totalBytes按照alignmentSize划分成partitionNum份
     * @param [T] partitionNum 划分的分数
     * @param [ssize_t] totalBytes 要划分的总bytes数
     * @param [ssize_t] alignmentSize 对齐的大小 (默认: ASYNC_K_PAGESIZE)
     * @return [*] 返回一个数组，对应的位置保存每个partition要读的bytes数
     * *********************************************************************************************************/
    template <typename T>
    size_t* partitionBinFile(T partitionNum, const ssize_t totalBytes, const ssize_t alignmentSize = ASYNC_K_PAGESIZE)
    {
        size_t* partition_bytesToRead = new size_t[partitionNum];
        memset(partition_bytesToRead, 0, sizeof(size_t) * (partitionNum));
        size_t partition_avgBytes = totalBytes / partitionNum;
        size_t partition_avgpageNum = (partition_avgBytes + alignmentSize - 1) / alignmentSize;
        for (int thread_id = 0; thread_id < partitionNum - 1; thread_id++)
        {
            partition_bytesToRead[thread_id] = partition_avgpageNum * alignmentSize;
        }
        int64_t remain_bytes = totalBytes - ((partitionNum - 1) * partition_avgpageNum * alignmentSize);
        if (remain_bytes < 0)
        {
            memset(partition_bytesToRead, 0, sizeof(size_t) * partitionNum);
            partition_bytesToRead[partitionNum - 1] = SCU64(totalBytes);
        }
        else
        {
            partition_bytesToRead[partitionNum - 1] = SCU64(remain_bytes);
        }

        if constexpr (DEBUG)
        {
            std::stringstream printInfo;
            printInfo << "File[" << fileName_ << "](" << totalBytes << "), each partition read bytes: " << std::endl;
            for (T partition_id = 0; partition_id < partitionNum; partition_id++)
            {
                printInfo << "T[" << partition_id << "]:" << partition_bytesToRead[partition_id] << ", ";
            }
            printInfo << std::endl;
            Msg_info("%s", printInfo.str().c_str());
        }
        return partition_bytesToRead;
    }

    /* *********************************************************************************************************
     * @description: 支持多线程并行读取一个二进制文件的部分, 读取按照PageSize对齐读取, 可惜暂时不支持Bar
     * @param [T*] data 要存储数据的内存块, 该块应该>=该文件的大小
     * @param [off_t] start 起始位置
     * @param [off_t] end 终止位置
     * @param [int] thread_num 要使用的线程数
     * @return [*]
     * *********************************************************************************************************/
    template <typename T>
    void readBinFilePartial_sync(T* data, off_t start, ssize_t size, int thread_num = 1)
    {
        if (thread_num > omp_get_max_threads())
        {
            Msg_warn("Read File[%s] used thread_num(%d) >= thread_max_num(%d), switch to thread_max_num automatically", fileName_.c_str(), thread_num,
                     omp_get_max_threads());
            thread_num = omp_get_max_threads();
        }
        else if (thread_num <= 0)
        {
            Msg_warn("Read File[%s] used error thread_num(%d), switch to use single thread automatically", fileName_.c_str(), thread_num);
            thread_num = 1;
        }

        ssize_t totalBytes = getFileSize();
        assert_msg((start >= 0) && ((start + size) <= totalBytes), "The File[%s] meet error start or size, start = %zd, size = %zd, totalBytes = %zd",
                   fileName_.c_str(), start, size, totalBytes);
        totalBytes = size;
        assert_msg(totalBytes % sizeof(T) == 0, "The File[%s] totalBytes can not aligned with sizoef(T)", fileName_.c_str());

        char* data_temp = (char*)data;
        std::unique_ptr<size_t[]> thread_bytesToRead(partitionBinFile(thread_num, totalBytes));

        if constexpr (USE_BAR)
        {
            size_t maxBytes_thread = 0;
            for (int thread_id = 0; thread_id < thread_num; thread_id++)
            {
                if (maxBytes_thread < thread_bytesToRead[thread_id])
                {
                    maxBytes_thread = thread_bytesToRead[thread_id];
                    threadId_forBar_ = thread_id;
                }
            }
            if constexpr (DEBUG) Msg_info("Thread[%d] will be used for Bar", threadId_forBar_);
        }

#pragma omp parallel for num_threads(thread_num)
        for (int thread_id = 0; thread_id < thread_num; thread_id++)
        {
            size_t data_offset = std::accumulate(thread_bytesToRead.get(), thread_bytesToRead.get() + thread_id, SCU64(0));
            size_t file_offset = start + data_offset;
            readBinBlock(data_temp + data_offset, thread_bytesToRead[thread_id], file_offset, thread_id);
        }
    }

    /* *********************************************************************************************************
     * @description: 支持多线程并行读取一个二进制文件的一部分, 读取按照PageSize对齐
     *               我们自己对齐申请所需的内存
     * @param [off_t] start 起始位置
     * @param [off_t] end 终止位置
     * @param [int] thread_num 要使用的线程数
     * @return [*] 读取到磁盘内容的内存
     * *********************************************************************************************************/
    template <typename T>
    T* readBinFilePartial_sync(off_t start, ssize_t size, int thread_num = 1)
    {
        CPJ::Timer* t;
        if constexpr (DEBUG) t = new CPJ::Timer();

        ssize_t totalBytes = getFileSize();
        assert_msg((start >= 0) && ((start + size) <= totalBytes), "The File[%s] meet error start or size, start = %zd, size = %zd, totalBytes = %zd",
                   fileName_.c_str(), start, size, totalBytes);

        char* buffer = (char*)std::aligned_alloc(HugePage::PAGE_SIZE, size);
        assert_msg(buffer != nullptr, "The File[%s] meet std::aligned_alloc error", fileName_.c_str());
        T* buffer_data = (T*)buffer;
        readBinFilePartial_sync(buffer_data, start, size, thread_num);

        if constexpr (DEBUG)
        {
            if ((start == 0) && (size == getFileSize()))
            {
                Msg_info("File[%s] read finished, readTotalBytes = %zu, time: %s", fileName_.c_str(), size, t->get_time_str().c_str());
            }
            else
            {
                Msg_info("File[%s] partital read finished, start = %zd, readBytes = %zu, time: %s", fileName_.c_str(), start, size,
                         t->get_time_str().c_str());
            }
        }

        return buffer_data;
    }

    /* *********************************************************************************************************
     * @description: 支持多线程并行读取整个二进制文件, 读取按照PageSize对齐读取, 可惜暂时不支持Bar
     * @param [T*] data 要存储数据的内存块, 该块应该>=该文件的大小
     * @param [int] thread_num 要使用的线程数
     * @return [*]
     * *********************************************************************************************************/
    template <typename T>
    void readBinFileEntire_sync(T* data, int thread_num = 1)
    {
        readBinFilePartial_sync(data, static_cast<ssize_t>(0), getFileSize(), thread_num);
    }

    /* *********************************************************************************************************
     * @description: 支持多线程并行读取整个二进制文件, 读取按照PageSize对齐
     *               我们自己对齐申请所需的内存
     * @param [int] thread_num 要使用的线程数
     * @return [*] 读取到磁盘内容的内存
     * *********************************************************************************************************/
    template <typename T>
    T* readBinFileEntire_sync(int thread_num = 1)
    {
        return readBinFilePartial_sync<T>(static_cast<ssize_t>(0), getFileSize(), thread_num);
    }

    /* *********************************************************************************************************
     * @description: 这里我们保留原先的方法, 支持Bin
     * @param [int] thread_num 要使用的线程数
     * @return [*] 读取到磁盘内容的内存
     * *********************************************************************************************************/
    template <typename T>
    T* readBinFileEntire_sync_old(int thread_num = 1)
    {
        CPJ::Timer* t;
        if constexpr (DEBUG) t = new CPJ::Timer();

        if (thread_num > omp_get_max_threads())
        {
            Msg_warn("Read File[%s] used thread_num(%d) >= thread_max_num(%d), switch to thread_max_num automatically", fileName_.c_str(), thread_num,
                     omp_get_max_threads());
            thread_num = omp_get_max_threads();
        }
        else if (thread_num <= 0)
        {
            Msg_warn("Read File[%s] used error thread_num(%d), switch to use single thread automatically", fileName_.c_str(), thread_num);
            thread_num = 1;
        }

        uint64_t total_bytes = getFileSize();
        uint64_t utilSize = sizeof(T);

        size_t bytes_to_read = total_bytes;
        size_t read_offset = 0;
        assert_msg(lseek(fd_, read_offset, SEEK_SET) == read_offset, "Read error In Function readBinFile_old(...)");
        assert_msg(total_bytes % utilSize == 0, "The File[%s] totalBytes can not aligned with sizoef(T)", fileName_.c_str());

        size_t length = total_bytes / utilSize;
        T* array_ = new T[length];
        size_t read_bytes = 0;
        size_t offset = 0;
        T* array_temp = new T[SYNC_CHUNK_SIZE];

        Bar* bar = new Bar(static_cast<bool>(length >= BAR_SHOWN_THRESHOLD));
        while (read_bytes < bytes_to_read)
        {
            bar->progress(static_cast<int64_t>(static_cast<double>(read_bytes) / bytes_to_read * 100), static_cast<int64_t>(100),
                          "[Load BinArray]: ");
            int64_t curr_read_bytes;
            if (bytes_to_read - read_bytes > utilSize * SYNC_CHUNK_SIZE)
            {
                curr_read_bytes = read(fd_, array_temp, utilSize * SYNC_CHUNK_SIZE);
            }
            else
            {
                curr_read_bytes = read(fd_, array_temp, bytes_to_read - read_bytes);
            }
            assert_msg(curr_read_bytes >= 0, "Read error In Function load_binFile(...)");
            read_bytes += curr_read_bytes;
            uint64_t curr_read = curr_read_bytes / utilSize;
#pragma omp parallel for num_threads(thread_num)
            for (uint64_t util = 0; util < curr_read; util++)
            {
                array_[util + offset] = array_temp[util];
            }
            offset += curr_read;
        }

        delete[] array_temp;
        bar->finish();

        if constexpr (DEBUG)
            Msg_info("File[%s] read finished, used [readBinFileEntire_sync_old], totalBytes = %zu, time: %s", fileName_.c_str(), total_bytes,
                     t->get_time_str().c_str());
        return array_;
    }

    /* *********************************************************************************************************
     * @description: 支持并行写入到当前的文件, 适合于Bin File
     * @param [T*] data 存储要写入的
     * @param [size_t] writeBytes 要写入的总Bytes数
     * @param [size_t] offset 写入的文件的偏移量
     * @return [*]
     * *********************************************************************************************************/
    template <typename T>
    ssize_t writeBinBlock(const T* data, size_t writeBytes, size_t offset, int threadForBar = 0) const
    {
        assert_msg(fd_ != -1, "Please open File [%s] first, then call writeBlock()", fileName_.c_str());

        size_t barId = 0;
        Bar* bar{nullptr};
        if constexpr (USE_BAR)
        {
            if (threadForBar == threadId_forBar_) bar = new Bar(static_cast<bool>(writeBytes >= BAR_SHOWN_BYTES_THRESHOLD_WRITE));
        }

        ssize_t total_bytes_written = 0ull;
        ssize_t bytes_written = 0;
        while (total_bytes_written < writeBytes)
        {
            bytes_written = pwrite(fd_, data + total_bytes_written, writeBytes - total_bytes_written, offset + total_bytes_written);
            assert_msg(bytes_written > 0, "File[%s] pwrite error, bytes_written = %zd, errMsg = %s", fileName_.c_str(), bytes_written,
                       getErrorMsg(errno).c_str());
            total_bytes_written += bytes_written;

            if constexpr (USE_BAR)
            {
                if ((threadForBar == threadId_forBar_) && (total_bytes_written > (10000 * barId))) //! 这里一定不能bar更新太快, 一定要慢慢的来
                {
                    bar->progress(static_cast<int64_t>(static_cast<double>(total_bytes_written) / writeBytes * 100), static_cast<int64_t>(100),
                                  "[writeBinBlock]: ");
                    barId++;
                }
            }
        }
        assert_msg(total_bytes_written == writeBytes, "File[%s] writeBlock error, total_bytes_written = %zd, writeBytes = %zu", fileName_.c_str(),
                   total_bytes_written, writeBytes);

        if constexpr (USE_BAR)
        {
            if (threadForBar == threadId_forBar_) bar->finish();
        }

        return bytes_written;
    }

    /* *********************************************************************************************************
     * @description: 支持多线程并行写入整个二进制文件, 写入按照PageSize对齐读取, 可惜暂时不支持Bar
     * @param [T*] data 存储数据的内存块
     * @param [size_t] wtiteBytes 要写入的数据总量
     * @param [int] thread_num 要使用的线程数
     * @return [*]
     * *********************************************************************************************************/
    template <typename T>
    void writeBinFile_sync(T* data, size_t wtiteBytes, int thread_num = 1)
    {
        CPJ::Timer* t;
        if constexpr (DEBUG) t = new CPJ::Timer();

        if (thread_num > omp_get_max_threads())
        {
            Msg_warn("Read File[%s] used thread_num(%d) >= thread_max_num(%d), switch to thread_max_num automatically", fileName_.c_str(), thread_num,
                     omp_get_max_threads());
            thread_num = omp_get_max_threads();
        }
        else if (thread_num <= 0)
        {
            Msg_warn("Read File[%s] used error thread_num(%d), switch to use single thread automatically", fileName_.c_str(), thread_num);
            thread_num = 1;
        }

        ssize_t totalBytes = wtiteBytes;
        assert_msg(totalBytes % sizeof(T) == 0, "The total write bytes can not aligned with sizoef(T)");

        char* data_temp = (char*)data;
        std::unique_ptr<size_t[]> thread_bytesToWrite(partitionBinFile(thread_num, totalBytes));

        if constexpr (USE_BAR)
        {
            size_t maxBytes_thread = 0;
            for (int thread_id = 0; thread_id < thread_num; thread_id++)
            {
                if (maxBytes_thread < thread_bytesToWrite[thread_id])
                {
                    maxBytes_thread = thread_bytesToWrite[thread_id];
                    threadId_forBar_ = thread_id;
                }
            }
            if constexpr (DEBUG) Msg_info("Thread[%d] will be used for Bar", threadId_forBar_);
        }

        if constexpr (DEBUG)
        {
            std::stringstream printInfo;
            printInfo << "The total write bytes (" << totalBytes << "), each thread write bytes: " << std::endl;
            for (int thread_id = 0; thread_id < thread_num; thread_id++)
            {
                printInfo << "T[" << thread_id << "]:" << thread_bytesToWrite[thread_id] << ", ";
            }
            printInfo << std::endl;
            Msg_info("%s", printInfo.str().c_str());
        }

#pragma omp parallel for num_threads(thread_num)
        for (int thread_id = 0; thread_id < thread_num; thread_id++)
        {
            size_t offset = std::accumulate(thread_bytesToWrite.get(), thread_bytesToWrite.get() + thread_id, SCU64(0));
            writeBinBlock(data_temp + offset, thread_bytesToWrite[thread_id], offset, thread_id);
        }

        if constexpr (DEBUG)
            Msg_info("File[%s] write finished, used [writeBinFile_sync], totalBytes = %zu, time: %s", fileName_.c_str(), totalBytes,
                     t->get_time_str().c_str());
    }
};
} // namespace CPJ