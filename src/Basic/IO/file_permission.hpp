#pragma once

namespace CPJ {

/* 八进制 */
#define FILE_PERMISSION_OWNER_READ 0400    // 当前用户可读取
#define FILE_PERMISSION_OWNER_WRITE 0200   // 当前用户可写入
#define FILE_PERMISSION_OWNER_EXECUTE 0100 // 当前用户可执行

#define FILE_PERMISSION_GROUP_READ 0040    // 用户组可读取
#define FILE_PERMISSION_GROUP_WRITE 0020   // 用户组可写入
#define FILE_PERMISSION_GROUP_EXECUTE 0010 // 用户组可执行

#define FILE_PERMISSION_OTHER_READ 0004    // 其他用户可读取
#define FILE_PERMISSION_OTHER_WRITE 0002   // 其他用户可写入
#define FILE_PERMISSION_OTHER_EXECUTE 0001 // 其他用户可执行

/* 文件所有者可读取和写入，但不能执行(0600) */
#define OWNER_READ_WRITE (FILE_PERMISSION_OWNER_READ | FILE_PERMISSION_OWNER_WRITE)
/* 文件所有者可读取、写入和执行 (0700) */
#define OWNER_READ_WRITE_EXECUTE (OWNER_READ_WRITE | FILE_PERMISSION_OWNER_EXECUTE)

/* 文件所有者和用户组可读取，但不能写入和执行 (0640) */
#define GROUP_READ (OWNER_READ_WRITE | FILE_PERMISSION_GROUP_READ)
/* 文件所有者和用户组可读取和写入，但不能执行 (0660) */
#define GROUP_READ_WRITE (OWNER_READ_WRITE | FILE_PERMISSION_GROUP_READ | FILE_PERMISSION_GROUP_WRITE)
/* 文件所有者和用户组可读取、写入和执行 (0770) */
#define GROUP_READ_WRITE_EXECUTE (GROUP_READ_WRITE | FILE_PERMISSION_OWNER_EXECUTE | FILE_PERMISSION_GROUP_EXECUTE)

/* 所有用户都可以读取文件，但不能写入和执行 (0644) (默认) */
#define ANY_READ (GROUP_READ | FILE_PERMISSION_OTHER_READ)
/* 所有用户都可以读取和写入文件，但不能执行 (0666) */
#define ANY_READ_WRITE (GROUP_READ_WRITE | FILE_PERMISSION_OTHER_READ | FILE_PERMISSION_OTHER_WRITE)
/* 所有用户都可以读取、写入和执行文件 (0777) */
#define ANY_READ_WRITE_EXECUTE (ANY_READ_WRITE | FILE_PERMISSION_OWNER_EXECUTE | FILE_PERMISSION_GROUP_EXECUTE | FILE_PERMISSION_OTHER_EXECUTE)

} // namespace CPJ