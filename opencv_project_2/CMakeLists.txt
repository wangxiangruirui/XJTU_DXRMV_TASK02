cmake_minimum_required(VERSION 3.10)
project(opencv_project_2)

set(CMAKE_CXX_STANDARD 14)

# 查找OpenCV包
find_package(OpenCV REQUIRED)

# 包含OpenCV头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(opencv_project_2 src/main.cpp)

# 链接OpenCV库
target_link_libraries(opencv_project_2 ${OpenCV_LIBS})
