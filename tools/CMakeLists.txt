cmake_minimum_required(VERSION 3.20)

# Find LLVM and Clang
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

# parallax-transform tool
add_executable(parallax-transform
    parallax-transform.cpp
)

# Link against Clang libraries
# For LLVM 21+, use monolithic libraries
if(LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL "21.0")
    target_link_libraries(parallax-transform
        PRIVATE
            clang-cpp
            LLVM
    )
else()
    # Use component libraries for older versions
    target_link_libraries(parallax-transform
        PRIVATE
            clangTooling
            clangFrontend
            clangDriver
            clangSerialization
            clangParse
            clangSema
            clangAnalysis
            clangEdit
            clangAST
            clangLex
            clangBasic
            clangRewrite
    )
    llvm_map_components_to_libnames(llvm_libs support)
    target_link_libraries(parallax-transform
        PRIVATE
            ${llvm_libs}
    )
endif()

install(TARGETS parallax-transform
    RUNTIME DESTINATION bin
)
