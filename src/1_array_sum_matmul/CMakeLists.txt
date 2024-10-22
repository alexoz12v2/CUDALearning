# Minimum required version of CMake
cmake_minimum_required(VERSION 3.18)

# Project name and language
project(MyCUDAProject LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the include directory for header files
include_directories(include)

# Find CUDA package (ensures nvcc is found)
enable_language(CUDA)

# Source files
set(SOURCE_FILES main.cpp src/cuda_kernel.cu)

# Executable target
add_executable(lab ${SOURCE_FILES})

# Link CUDA libraries
target_link_libraries(lab ${CUDA_LIBRARIES})