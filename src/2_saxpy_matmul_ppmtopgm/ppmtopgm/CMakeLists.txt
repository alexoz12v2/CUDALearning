cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(image-processing-exercise-2
  VERSION 1.0.0.0
  DESCRIPTION "Exercise 2"
)

enable_language(CUDA)

add_custom_target(copy-image ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/image.ppm)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/image.ppm
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_SOURCE_DIR}/imglib/sample_640_426.ppm
          ${CMAKE_CURRENT_BINARY_DIR}/image.ppm
  DEPENDS ${CMAKE_SOURCE_DIR}/imglib/sample_640_426.ppm
)

add_executable(lab-img-2
  ${CMAKE_CURRENT_BINARY_DIR}/image.ppm
)

target_sources(lab-img-2
  PRIVATE main.cpp src/kernels.cu include/kernels.cuh imglib/img.cpp imglib/img.h
)

target_compile_features(lab-img-2
  PRIVATE cuda_std_20 cxx_std_20
)

target_include_directories(lab-img-2
  PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}
)

set_target_properties(lab-img-2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(lab-img-2
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-G>
    $<$<COMPILE_LANGUAGE:CUDA>:-g>
)