add_library(parallax-clang-plugin MODULE
    ParallaxPlugin.cpp
    ParallaxPluginV2.cpp
    ParallaxRewriter.cpp
)

target_include_directories(parallax-clang-plugin
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../parallax-runtime/include
)

# Link against Clang libraries and parallax-plugin
# On Arch Linux with LLVM/Clang 21+, we use the monolithic libclang-cpp.so
target_link_libraries(parallax-clang-plugin
    PRIVATE
        clang-cpp
        parallax-plugin
)

install(TARGETS parallax-clang-plugin
    LIBRARY DESTINATION lib
)
