#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include <bits/iterator_concepts.h>
#include <filesystem>
#include <vector>

namespace CPJ {

class FS
{
  public:
    static constexpr bool FS_DEBUG = false;

    /* *********************************************************************************************************
     * @description: 递归的创建文件夹
     * @param [string&] filePath
     * @return [*]
     * *********************************************************************************************************/
    static void createDirectory(const std::string& filePath)
    {
        std::filesystem::path filePath_fs(filePath);
        assert_msg(!filePath_fs.empty(), "Invalid file path, filePath = %s", filePath.c_str());

        if (filePath_fs.has_filename())
        {
            filePath_fs = filePath_fs.parent_path();
            if (filePath_fs.empty())
            {
                Msg_warn("The Path[%s] is a empty Directory, skiping create...", filePath_fs.c_str());
                return;
            }
        }

        if (!std::filesystem::is_directory(filePath_fs))
        {
            std::error_code ec;
            bool isCreate = std::filesystem::create_directories(filePath_fs, ec);
            assert_msg(isCreate, "create_directories failed, errCode = %d, errMsg = %s", ec.value(), ec.message().c_str());
            if constexpr (FS_DEBUG) Msg_info("Directory[%s] create success", filePath_fs.string().c_str());
        }
        else if constexpr (FS_DEBUG)
        {
            Msg_info("Directory[%s] already exist", filePath_fs.string().c_str());
        }
    }

    /* *********************************************************************************************************
     * @description: 检查文件或目录是否存在
     * @param [string&] filePath
     * @return [bool]
     * *********************************************************************************************************/
    static bool isExist(const std::string& filePath)
    {
        std::filesystem::path filePath_fs(filePath);
        assert_msg(!filePath_fs.empty(), "Invalid file path, filePath = %s", filePath.c_str());
        return std::filesystem::exists(filePath_fs);
    }

    /* *********************************************************************************************************
     * @description: 检查是否为目录, 当目录不存在时也会判定为false
     * @param [string&] filePath
     * @return [bool]
     * *********************************************************************************************************/
    static bool isDirectory(const std::string& filePath)
    {
        std::filesystem::path filePath_fs(filePath);
        assert_msg(!filePath_fs.empty(), "Invalid file path, filePath = %s", filePath.c_str());
        return std::filesystem::is_directory(filePath_fs);
    }

    /* *********************************************************************************************************
     * @description: 检查是否为文件, 当文件不存在时也会判定为false
     * @param [string&] filePath
     * @return [bool]
     * *********************************************************************************************************/
    static bool isFile(const std::string& filePath)
    {
        std::filesystem::path filePath_fs(filePath);
        assert_msg(!filePath_fs.empty(), "Invalid file path, filePath = %s", filePath.c_str());
        return std::filesystem::is_regular_file(filePath_fs);
    }

    /* *********************************************************************************************************
     * @description: 获取文件名, 如: my.txt, 如果没有文件名就返回 ""
     * @param [string&] filePath
     * @return [string]
     * *********************************************************************************************************/
    static std::string getFileName(const std::string& filePath)
    {
        std::filesystem::path filePath_fs(filePath);
        assert_msg(!filePath_fs.empty(), "Invalid file path, filePath = %s", filePath.c_str());
        if (filePath_fs.has_filename())
        {
            return filePath_fs.filename().string();
        }
        return "";
    }

    /* *********************************************************************************************************
     * @description: 获取文件的扩展名, 如: .txt, 如果没有扩展名就返回 ""
     * @param [string&] filePath
     * @return [string]
     * *********************************************************************************************************/
    static std::string getFileExtension(const std::string& filePath)
    {
        std::filesystem::path filePath_fs(filePath);
        assert_msg(!filePath_fs.empty(), "Invalid file path, filePath = %s", filePath.c_str());
        if (filePath_fs.has_extension())
        {
            return filePath_fs.extension().string();
        }
        return "";
    }

    /* *********************************************************************************************************
     * @description: 获取文件的主体名, 如: my, 如果没有主体名就返回 ""
     * @param [string&] filePath
     * @return [string]
     * *********************************************************************************************************/
    static std::string getFileStem(const std::string& filePath)
    {
        std::filesystem::path filePath_fs(filePath);
        assert_msg(!filePath_fs.empty(), "Invalid file path, filePath = %s", filePath.c_str());
        if (filePath_fs.has_stem())
        {
            return filePath_fs.stem().string();
        }
        return "";
    }

    /* *********************************************************************************************************
     * @description: 获取目录[dirPath]下的所有以[extension]为扩展名的文件 (深度为1-level）
     * @param [string&] dirPath 要遍历的path
     * @param [string&] extension 指定的扩展名, 如果extension = ".*", 则会返回所有的文件, 包括没有扩展名的
     * @return [*] 返回包含符合条件文件名的vector
     * *********************************************************************************************************/
    static std::vector<std::string> getAllFile(const std::string& dirPath, const std::string& extension = ".*")
    {
        bool isAll = (extension == ".*");
        std::vector<std::string> file_vec;
        assert_msg(isDirectory(dirPath), "The Path [%s] is not a directory", dirPath.c_str());
        for (const auto& entry : std::filesystem::directory_iterator(dirPath))
        {
            const auto& file_string = entry.path().string();
            bool is_file = isFile(file_string);
            if (is_file && (isAll || (getFileExtension(file_string) == extension)))
            {
                file_vec.push_back(file_string);
            }
        }
        return file_vec;
    }

    /* *********************************************************************************************************
     * @description: 获取对应目录, 如: /home/omnisky/a.txt -> 返回/home/omnisky, 注意返回的没有最后的'/'
     * @param [string&] filePath
     * @return [*]
     * *********************************************************************************************************/
    static std::string getDirectory(const std::string& filePath)
    {
        std::filesystem::path filePath_fs(filePath);
        assert_msg(!filePath_fs.empty(), "Invalid file path, filePath = %s", filePath.c_str());

        return filePath_fs.parent_path().string();
    }

    /* *********************************************************************************************************
     * @description: 删除指定的文件
     * @param [string&] filePath
     * @return [*]
     * *********************************************************************************************************/
    static void deleteFile(const std::string& filePath)
    {
        std::filesystem::path filePath_fs(filePath);
        assert_msg(!filePath_fs.empty(), "Invalid file path, filePath = %s", filePath.c_str());

        assert_msg(isFile(filePath_fs), "[%s] is not file", filePath.c_str());

        try
        {
            if (!std::filesystem::remove(filePath_fs))
            {
                Msg_error("Delete File[%s] failed", filePath.c_str())
            }
        } catch (const std::filesystem::filesystem_error& e)
        {
            assert_msg(false, "Error deleting file: %s", e.what());
        }
    }

    /* *********************************************************************************************************
     * @description: 在[fileDir]目录下寻找以[prefixStr]开头的所有文件
     * @param [const std::string&] fileDir 指定寻找的目录
     * @param [const std::string&] prefixStr 文件前缀
     * @return [std::vector<std::string>] 所有满足需求的文件名集合
     * *********************************************************************************************************/
    static std::vector<std::string> findFile_withPrefix(const std::string& fileDir, const std::string& prefixStr)
    {
        std::vector<std::string> vec;
        try
        {
            for (const auto& entry : std::filesystem::directory_iterator(fileDir))
            {
                if (entry.is_regular_file() && entry.path().filename().string().rfind(prefixStr, 0) == 0) // 判断文件名是否以 prefixStr 开头
                {
                    vec.push_back(entry.path().string()); // 将文件路径作为字符串添加到结果中
                }
            }
        } catch (const std::filesystem::filesystem_error& e)
        {
            Msg_error("Error[%s] in Func[%s]", e.what(), __FUNCTION__);
        }

        return vec;
    }
};

} // namespace CPJ