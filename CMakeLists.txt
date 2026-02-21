cmake_minimum_required(VERSION 3.12.0)
project(FSI CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED "ON")

# Sets cmake in release mode so the compiler can use -O2 or -O3
if(NOT CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# Print message status containing compilation informations
message(STATUS)
message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    add_definitions(-DBUILD_TYPE_DEBUG)
endif()

# Locate deal.II initializing its variables
find_package(deal.II 9.5.1 REQUIRED
    HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix}
)
deal_ii_initialize_cached_variables()

# Add compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-conversion -Wmissing-braces -Wnon-virtual-dtor")

add_executable(${PROJECT_NAME}
    src/FSI_parallel.cpp
    src/main.cpp
)

deal_ii_setup_target(${PROJECT_NAME})