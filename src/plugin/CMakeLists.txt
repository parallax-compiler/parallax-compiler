add_library(parallax-clang-plugin MODULE
    ParallaxPlugin.cpp
)

# Link against Clang libraries
# On Arch Linux with LLVM/Clang 21+, we use the monolithic libclang-cpp.so
target_link_libraries(parallax-clang-plugin
    PRIVATE
    clang-cpp
)
