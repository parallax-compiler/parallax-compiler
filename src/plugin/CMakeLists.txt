add_library(parallax-clang-plugin MODULE
    ParallaxPlugin.cpp
)

# Link against Clang libraries
target_link_libraries(parallax-clang-plugin
    PRIVATE
    clangAST
    clangBasic
    clangFrontend
    clangLex
    clangTooling
)
