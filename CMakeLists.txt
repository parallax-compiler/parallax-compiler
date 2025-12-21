cmake_minimum_required(VERSION 3.20)
project(parallax-compiler VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find LLVM (accept any version)
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Find Clang
find_package(Clang REQUIRED CONFIG)

# Compiler plugin library
add_library(parallax-plugin SHARED
    src/lambda_extractor.cpp
    src/lambda_compiler.cpp
    src/spirv_generator.cpp
    src/execution_policy.cpp
)

target_include_directories(parallax-plugin
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CLANG_INCLUDE_DIRS}
)

# Link against LLVM and Clang libraries
# LLVM 21+ uses monolithic library
if(LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL "21.0")
    # Use monolithic LLVM library for LLVM 21+
    target_link_libraries(parallax-plugin
        PRIVATE
            LLVM
    )
else()
    # Use component libraries for older LLVM
    llvm_map_components_to_libnames(llvm_libs support core irreader)
    target_link_libraries(parallax-plugin
        PRIVATE
            ${llvm_libs}
    )
endif()

# Note: Clang libraries are optional for this module
# The actual Clang integration happens at a higher level

# Installation
install(TARGETS parallax-plugin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/parallax
    DESTINATION include
)

# Clang Plugin
add_subdirectory(src/plugin)
