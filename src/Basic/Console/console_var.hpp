#pragma once

#include <mpi.h>
#include <string>

class Console_Val
{
  public:
    static int serverId;
    static int serverNum;
    static std::string serverName;
};

int Console_Val::serverId = -1;
int Console_Val::serverNum = -1;
std::string Console_Val::serverName = "Unknow";
