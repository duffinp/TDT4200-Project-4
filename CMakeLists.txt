#
# Specify minimum CMake version and project name
#
cmake_minimum_required (VERSION 3.0)

set(CMAKE_C_COMPILER /usr/bin/gcc-6)
set(CMAKE_CXX_COMPILER /usr/bin/g++-6)

project(gpurender)

# Use this if CMake complains about the installation directory not being found.
# The installation directory unfortunately depends on your operating system and/or distribution :(
#set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-9.1)

find_package(CUDA REQUIRED)

# These may need to be changed depending on your GPU
list(APPEND CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=\"sm_52,compute_52\" -lcurand -lineinfo)


#
# CMake setup
#
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set (CMAKE_VERBOSE_MAKEFILE 0) # 1 should be used for debugging
set (CMAKE_SUPPRESS_REGENERATION TRUE) # Suppresses ZERO_CHECK
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp /W3 /std:c++latest")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -lpthread -Wall -Wextra -std=c++11")
endif()

#
# Add files
#
file (GLOB_RECURSE PROJECT_HEADERS src/*.hpp
                                   src/*.h)
file (GLOB_RECURSE PROJECT_SOURCES src/*.cpp
                                   src/*.cxx)
file (GLOB_RECURSE PROJECT_KERNELS src/*.cu
                                   src/*.cuh)
file (GLOB         PROJECT_CONFIGS CMakeLists.txt
                                   README.rst
                                  .gitignore
                                  .gitmodules)

#
# Organizing files
#
source_group ("headers" FILES ${PROJECT_HEADERS})
source_group ("sources" FILES ${PROJECT_SOURCES})
source_group ("kernels" FILES ${PROJECT_KERNELS})

#
# Set executable and target link libraries
#
add_definitions (-DPROJECT_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\")
include_directories(${CUDA_INCLUDE_DIRS})
cuda_add_executable (${PROJECT_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS} ${PROJECT_CONFIGS} ${PROJECT_KERNELS})
set_target_properties (${PROJECT_NAME} PROPERTIES
RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME})