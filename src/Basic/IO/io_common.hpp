#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Console/console_bar.hpp"
#include "Basic/Thread/omp_def.hpp"

#include <cstdint>
#include <fcntl.h>
#include <linux/fs.h> /* 包含BLKGETSIZE64 命令的定义 */
#include <string>
#include <sys/ioctl.h> /* ioctl*/
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace CPJ {

/* ***************************************************************************************
 * Func: Get the Total File Size In Bytes
 *
 * @param [std::string filename]  The Graph File
 * ***************************************************************************************/
inline uint64_t get_file_size(std::string filename)
{
    struct stat st;
    assert_msg(stat(filename.c_str(), &st) == 0, "Bin File [%s] Not Exist", filename.c_str());
    return st.st_size;
}

template <typename T>
T* load_binFile_cStyle(std::string path, uint64_t length)
{
    const uint64_t chunk_size = 1 << 20;
    uint64_t total_bytes = get_file_size(path.c_str());
    uint64_t utilSize = sizeof(T);
    assert_msg(
        ((total_bytes / utilSize) == length),
        "(total_bytes / utilSize != length) In Function load_binFile(...), file = %s, fileLength = %zu, length = %zu, (Perhaps the countl_type "
        "was set incorrectly)",
        path.c_str(), (total_bytes / utilSize), length);

    size_t bytes_to_read = total_bytes;
    size_t read_offset = 0;

    int fin = open(path.c_str(), O_RDONLY);
    assert_msg(lseek(fin, read_offset, SEEK_SET) == read_offset, "Read error In Function load_binFile(...)");

    T* array_ = new T[length];
    size_t read_bytes = 0;
    size_t offset = 0;
    T* array_temp = new T[chunk_size];

    Bar* bar = new Bar(static_cast<bool>(length >= BAR_SHOWN_THRESHOLD));

    while (read_bytes < bytes_to_read)
    {
        bar->progress(static_cast<int64_t>(static_cast<double>(read_bytes) / bytes_to_read * 100), static_cast<int64_t>(100), "[Load BinArray]: ");

        int64_t curr_read_bytes;
        if (bytes_to_read - read_bytes > utilSize * chunk_size)
        {
            curr_read_bytes = read(fin, array_temp, utilSize * chunk_size);
        }
        else
        {
            curr_read_bytes = read(fin, array_temp, bytes_to_read - read_bytes);
        }
        assert_msg_clear(curr_read_bytes >= 0, "Read error In Function load_binFile(...)");
        read_bytes += curr_read_bytes;
        uint64_t curr_read = curr_read_bytes / utilSize; // Number of edges that have been read
        omp_par_for(uint64_t util = 0; util < curr_read; util++) { array_[util + offset] = array_temp[util]; }
        offset += curr_read;
    }

    delete[] array_temp;
    assert_msg_clear(close(fin) == 0, "[load_binArray] Close Error!");

    bar->finish();
    Msg_info("Load_binArray:[%s] Finished !", basename((char*)(path.c_str())));

    return array_;

clear:
    if (array_temp != nullptr) delete[] array_temp;
    if (array_ != nullptr) delete[] array_;

    STOP;
}

} // namespace CPJ