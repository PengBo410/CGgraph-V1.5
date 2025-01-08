<span style="font-size: larger; cursor: pointer;">&#9758;</span> [Switch to Chinese](README.md)
## CGgraph-V1.5
Enabling CPU and GPU Collaborative Processing of Graph Data. Source Code for `CGgraph: An Ultra-fast Graph Processing System on Modern Commodity CPU-GPU Co-processor`. The current version is V1.5, and we will continue to update and optimize it.

---
## The dependencies of CGgraph include:
- Host
	- C++  (>11.4.0)
	- OpenMP
	- TBB
- Device
    - CUDA (>12.3)
	- Thrust
	- CUB
- Command line tool
	- [gflags](https://github.com/gflags/gflags)
---
## The data type of CGgraph
- **Location**：src/Basic/Type/data_type.hpp
- **Explanation**：
	- `vertex_id_type`： data type representing vertices in graph data.
	- `edge_data_type`： data type representing edges in graph data.
	- `vertex_data_type`： representation of the data type for vertex values in graph data, for example, BFS is typically uint32_t, while PageRank is typically float/double.
	- `count_type`： representation of the data type for the number of vertices in graph data; if the total number of vertices in the graph exceeds the maximum representable range of uint32_t, `count_type` needs to be set to uint64_t.
	- `countl_type`： representation of the data type for the number of edges in graph data; if the total number of vertices in the graph exceeds the maximum representable range of uint32_t, `count_type` needs to be set to uint64_t.
	- `degree_type`： representing of the data type for degree of vertices in graph data.
- **Note**： Please ensure precise definitions of each data type. For example, if the total number of edges does not exceed the maximum representable range of uint32_t, `countl_type` should not be defined as uint64_t. The program will provide appropriate reminders in such cases.
---
## The input data of CGgraph
- **Location**：src/Basic/Graph/graphFileList.hpp
- **Explanation**：
	1.The input of CGgraph is a binary CSR file. Each CSR file of the graph needs to include three files:
		- csrOffsetFile： the starting position of each vertex in the edge array in the graph, data type is `countl_type`, with a length of total number of vertices plus one.
		- csrDestFile：the destination vertices of each edge in the graph, data type is `vertex_id_type`, with a length of total number of edges.
		- csrWeightFile：the weight  of each edge in the graph, data type is `edge_data_type`, with a length of total number of edges.
     2.graphFileList is a compilation of graph data files, where the files to be processed can be added before compilation. For example, adding graph data for the ***graphName*** such as *"friendster"* and *"uk-union"*：
     ```cpp
	 if (graphName == "friendster") 
	 {
		graphFile.vertices = 124836180; // vertices
	    graphFile.edges = 1806067135; // edges
		graphFile.csrOffsetFile = "/data/bin/friendster/native_csrOffset_u32.bin"; //file path
        graphFile.csrDestFile = "/data/bin/friendster/native_csrDest_u32.bin";     //file path
        graphFile.csrWeightFile = "/data/bin/friendster/native_csrWeight_u32.bin"; //file path
	 } 
	 else if (graphName == "uk-union")
	 {
		graphFile.vertices = 133633040; // vertices
        graphFile.edges = 5475109924;// edges
		graphFile.csrOffsetFile = "/data/bin/friendster/native_csrOffset_u32.bin"; //file path
        graphFile.csrDestFile = "/data/bin/friendster/native_csrDest_u32.bin";     //file path
        graphFile.csrWeightFile = "/data/bin/friendster/native_csrWeight_u32.bin"; //file path
	 }
	 ```

---
## Run CGgraph
- During runtime, CGgraph identifies different graph data based on the defined ***graphName***. Additionally, ***graphName*** is invoked multiple times within the program.
- After compiling CGgraph, the program generates an executable file named **CGgraphV1.5**. You can access the following information by executing the command` $ CGgraphV1.5 --help` in the terminal:：
 ```
  -algorithm (The Algorithm To Be Run: [0]:BFS, [1]:SSSP) type: int32
      default: 0
    -gpuMemory (The GPU Memory Type: [0]:GPU_MEM, [1]:UVM, [2]:ZERO_COPY)
      type: int32 default: 0
    -graphName (The Graph Name) type: string default: "friendster"
    -root (The Root For BFS/SSSP Or MaxIte For PageRank) type: int64
      default: 0
    -runs (The Number Of Times That The Algorithm Needs To Run) type: int32
      default: 5
    -useDeviceId (The GPU ID To Be Used) type: int32 default: 0
```
- **Explanation**:
   1. gpuMemory：used to define the type of GPU memory to be utilized. Since CGgraph is a system that collaboratively executes on both CPU and GPU, considering the limited GPU memory capacity, we categorize the types of GPU memory usage as follows:
	  - When the GPU memory is sufficient to accommodate all graph data, enabling `GPU_MEM` can provide better performance.
	  - When the GPU memory can only hold a portion of the graph data, and this portion is larger than a certain percentage R% of the graph data, enabling `GPU_MEM` will lead to improved performance；
	  - When the GPU memory can only accommodate a portion of the graph data, but this portion is less than R% of the graph data, users need to enable `UVM` or `ZERO_COPY` (the program will provide corresponding prompts). In practice, enabling `ZERO_COPY` can yield better performance in certain scenarios.
	2.  In CGgraph-V1.5, BFS and SSSP algorithms have been optimized first. The optimization for the remaining WCC and PageRank algorithms will be updated shortly；
	3. When running CGgraph-V1.5 for the first time, it is recommended to select a graph dataset that can be entirely accommodated within the GPU memory. This ensures that the program can thoroughly evaluate the performance differences between the CPU and GPU；
-  **例子**:
	If you want to run the BFS algorithm on the graph data file with ***graphName*** set to friendster, you should execute:
`$ CGgraphV1.5 -graphName friendster -algorithm 0 -gpuMemory 0 -root 100 -runs 5  -useDeviceId 0`
	If the program runs successfully, the console will display the following results (12Cores CPU + 2560Cores GPU):
	![fig](./fig/run_example.png)
	Of course, a significant amount of console output can sometimes slightly impact the performance of CGgraphV1.5
---

## Other sharing
The following content is a lightweight tool I have personally packaged and find quite good. Feel free to try it out. If there are any areas that are not suitable, please feel welcome to provide feedback.

### Console printing and log output
- **Location**：
  - src/Basic/Console/console_V3_3.hpp
  - src/Basic/Log/log.hpp
- **Explanation**：
	- `Msg_info` White output
	- `Msg_check` Green output
	- `Msg_finish` Green output
	- `Msg_warn` Yellow output
	- `Msg_error` Red output
	- `Msg_node` Print only on the master node in a distributed environment
	- `Log_info` Output to a specified log file
	- `assert_msg` Assertion with printing
	- `assert_msg_clear`   Assertion with printing, error will jump to a specified goto
- **With smart**：There will be alternative solutions in higher versions of C++, for example
  	- `Msg_info_smart` White output (no need to manually specify the data type to print)
- **说明**：
  - `ISCONSOLE` If set to 0, the above definitions will not be printed to the console.
  - `ISLOG` If set to 0, the above definitions will not be logged.
  - Note: Logging increases the program's execution time but facilitates debugging. Use it according to your needs.
- **Example**：
	- Used： ![fig](./fig/example_1.png)
	- Effect：![fig](./fig/example_2.png)


  		