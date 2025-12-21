cmake_minimum_required(VERSION 3.20)
project(parallax-compiler VERSION 0.1.0)

find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(clang-plugin)
add_subdirectory(llvm-pass)
