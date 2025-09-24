cmake_minimum_required(VERSION 3.18)
project(NeuralNetProject)

option(ENABLE_CUDA "Build with CUDA support" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------
# OpenCV2 setup
# ---------------------------
# If pkg-config is installed for OpenCV2
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv)

include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

# ---------------------------
# CUDA setup (optional)
# ---------------------------
if(ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    message(STATUS "Building with CUDA")
    add_definitions(-DUSE_CUDA)
endif()

# ---------------------------
# Source files and executable
# ---------------------------
set(SOURCES main.cpp)

if(ENABLE_CUDA)
    set(CUDA_SOURCES main.cu)
    cuda_add_executable(nn_gpu ${CUDA_SOURCES})
    target_include_directories(nn_gpu PRIVATE ${OPENCV_INCLUDE_DIRS})
    target_link_libraries(nn_gpu ${OPENCV_LIBRARIES})
else()
    add_executable(nn_cpu ${SOURCES})
    target_include_directories(nn_cpu PRIVATE ${OPENCV_INCLUDE_DIRS})
    target_link_libraries(nn_cpu ${OPENCV_LIBRARIES})
endif()
