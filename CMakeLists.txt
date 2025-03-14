cmake_minimum_required(VERSION 3.10)
project(MyProject)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
set(OpenCV_DIR  "F:/work/opencv454/opencv/build/x64/vc15/lib")
find_package(OpenCV REQUIRED) # 指定你使用的版本，例如 4.5.4
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加源文件
set(SOURCES
    main.cpp
    ConfidenceCompute.cpp
    CostVolCompute.cpp
    CostVolFilter.cpp
    DataDeal.cpp
    DataParameter.cpp
    DepthComputeToolOne.cpp
    DepthComputeToolTwo.cpp
    DisparityRefinement.cpp
    GlobalOptimization.cpp
    ImageRander.cpp
    SceneDepthCompute.cpp
    ToolOneTestDemo.cpp
    ToolTwoTestDemo.cpp
    VirtualDepthCompute.cpp
)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 链接 OpenCV 库
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# 可选：生成 compile_commands.json 以支持代码补全
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)