#******************************************************************************
# Copyright 2016 Ramsundar K G. All Rights Reserved.
#
# This source code is licensed as defined by the LICENSE file found in the
# root directory of this source tree.
#
# Author: Ramsundar K G (kgram007@gmail.com)
#
# This file is a part of C++ implementation of Eulerian Motion Magnification
# adapted from https://github.com/wzpan/QtEVM
#
#******************************************************************************

# CMake version check
cmake_minimum_required(VERSION 2.8)

# Project Name
project(Eulerian_Motion_Magnification)

# Dependent packages
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED program_options)
find_package(OpenMP)

# Using CXX Flags: Optimization (-O3) and OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -std=c++11")

# Includes
include_directories("include", "src")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# Executable file directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# Executable
add_executable(${PROJECT_NAME}
        src/main.cpp
        src/euler_mag.cpp
        include/timer.h
        include/euler_mag.hpp)

# Link libraries
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${OpenMP_LIBS}
        ${Boost_LIBRARIES}
        )
