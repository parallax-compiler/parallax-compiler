# Parallax Compiler

Clang plugin and LLVM passes for automatic GPU offloading of C++ parallel algorithms.

## Building

Requires LLVM 18+ and Clang development headers.

```bash
mkdir build && cd build
cmake .. -DLLVM_DIR=/usr/lib/llvm-18/cmake
make -j$(nproc)
```

## Usage

```bash
clang++ -fplugin=libparallax-plugin.so -fparallel-backend=vulkan your-code.cpp -lparallax-runtime
```

## Architecture

See [parallax-docs](https://github.com/parallax-compiler/parallax-docs) for detailed architecture.

## Contributing

See [CONTRIBUTING.md](https://github.com/parallax-compiler/.github/blob/main/CONTRIBUTING.md).
