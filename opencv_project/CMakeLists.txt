cmake_minimum_required(VERSION 3.10)
project(opencv_project)

set(CMAKE_CXX_STANDARD 11)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(opencv_project src/main.cpp)

# Link OpenCV libraries
target_link_libraries(opencv_project ${OpenCV_LIBS})
