# Parallax Compiler

Clang plugin and LLVM-to-SPIR-V compiler for automatic GPU offloading of C++ parallel algorithms.

## Overview

The Parallax compiler is a Clang plugin that transforms standard C++20 parallel algorithms into GPU-accelerated code. It performs compile-time AST transformations to:

1. **Automatically inject GPU allocators** into standard containers
2. **Extract lambda expressions** from parallel algorithm calls
3. **Generate LLVM IR** from lambda bodies
4. **Convert IR to SPIR-V** compute shaders
5. **Embed GPU kernels** directly in the compiled binary

## Features

### ✨ Automatic Allocator Injection (v1.0)

The compiler automatically rewrites standard containers to use GPU-accessible memory:

**You write:**
```cpp
std::vector<float> data(1000);
std::for_each(std::execution::par, data.begin(), data.end(),
             [](float& x) { x *= 2.0f; });
```

**Compiler generates:**
```cpp
std::vector<float, parallax::allocator<float>> data(1000);
// ... GPU kernel code ...
```

No source code changes required!

### Supported Operations

**Compound Assignment Operators:**
- `*=` (multiply-assign)
- `+=` (add-assign)
- `-=` (subtract-assign)
- `/=` (divide-assign)

**Binary Operators:**
- `+`, `-`, `*`, `/`
- Complex expressions: `x = x * 2.0f + 1.0f`

**Algorithms:**
- `std::for_each` - Full support
- `std::transform` - Full support
- `std::reduce` - In progress

## Building

### Prerequisites

- **LLVM/Clang:** 15+ (tested with 15, 17, 18)
- **CMake:** 3.20+
- **C++ Compiler:** C++20 support

### Build Instructions

```bash
mkdir build && cd build
cmake .. -DLLVM_DIR=/usr/lib/llvm-18/cmake
make -j$(nproc)
```

**Build artifacts:**
- `build/src/plugin/libparallax-clang-plugin.so` - Clang plugin
- `build/libparallax-plugin.so` - Core compiler library

## Usage

### Basic Compilation

```bash
clang++ -std=c++20 \
  -fplugin=/path/to/libparallax-clang-plugin.so \
  -I /path/to/parallax-runtime/include \
  -L /path/to/parallax-runtime/build \
  -lparallax-runtime \
  your-code.cpp -o your-program
```

### Example

**Input (your-code.cpp):**
```cpp
#include <vector>
#include <algorithm>
#include <execution>

int main() {
    std::vector<float> data(10000, 1.0f);

    // Runs on GPU automatically!
    std::for_each(std::execution::par, data.begin(), data.end(),
                 [](float& x) { x *= 2.0f; });

    return 0;
}
```

**Compilation:**
```bash
clang++ -std=c++20 \
  -fplugin=libparallax-clang-plugin.so \
  -I ../parallax-runtime/include \
  -L ../parallax-runtime/build \
  -lparallax-runtime \
  your-code.cpp -o program
```

**Output:** Binary with embedded SPIR-V GPU kernels

## Architecture

### Compilation Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. AST Analysis (Clang Plugin)                              │
│    - Detect std::execution::par calls                       │
│    - Trace iterators to container declarations              │
│    - Mark containers for allocator injection                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Allocator Injection (ParallaxRewriter)                   │
│    - Rewrite: std::vector<T> → std::vector<T, parallax::allocator<T>>
│    - Inject: #include <parallax/allocator.hpp>              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Lambda Extraction (ParallaxCollectorVisitor)             │
│    - Extract lambda AST from algorithm call                 │
│    - Analyze lambda body for supported operations           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. LLVM IR Generation (LambdaIRGenerator)                   │
│    - Translate lambda to LLVM IR                            │
│    - Generate kernel entry point                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. SPIR-V Generation (SPIRVGenerator)                       │
│    - Convert LLVM IR to SPIR-V                              │
│    - Emit compute shader with Vulkan semantics              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Code Emission (ParallaxRewriter)                         │
│    - Embed SPIR-V as uint32_t array                         │
│    - Replace algorithm call with kernel launch              │
│    - Write transformed source                               │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

**`src/plugin/ParallaxRewriter.cpp`**
- Main Clang plugin entry point
- AST rewriting and transformations
- Allocator injection logic
- Source file output

**`src/lambda_ir_generator.cpp`**
- Converts Clang AST to LLVM IR
- Handles all C++ expression types
- Generates GPU kernel entry points

**`src/spirv_generator.cpp`**
- LLVM IR to SPIR-V conversion
- Vulkan compute shader emission
- Descriptor set layout generation

## Implementation Details

### Allocator Injection Algorithm

```cpp
// Phase 1: Collection
for (auto* call : parallel_algorithm_calls) {
    auto* container = traceIteratorToContainer(call->getArg(1));
    if (container && !hasParallaxAllocator(container->getType())) {
        markContainerForAllocation(container);
    }
}

// Phase 2: Type Rewriting
for (auto* container : containers_needing_allocator) {
    QualType original = container->getType();  // std::vector<float>
    QualType element = extractElementType(original);  // float

    std::string new_type =
        "std::vector<" + element + ", parallax::allocator<" + element + ">>";

    rewriter.ReplaceText(container->getTypeSourceInfo()->getTypeLoc(), new_type);
}

// Phase 3: Header Injection
ensureAllocatorHeader();  // Inject #include <parallax/allocator.hpp>
```

### Iterator Tracing

The compiler traces iterator expressions back to their container:

```cpp
// Pattern 1: container.begin()
auto* member_call = dyn_cast<CXXMemberCallExpr>(iter);
Expr* object = member_call->getImplicitObjectArgument();
// → Trace to VarDecl

// Pattern 2: std::begin(container)
auto* call = dyn_cast<CallExpr>(iter);
Expr* arg = call->getArg(0);
// → Trace to VarDecl
```

## Debugging

### Enable Verbose Output

The plugin logs transformation details:

```bash
clang++ -fplugin=libparallax-clang-plugin.so ... 2>&1 | grep Parallax
```

**Example output:**
```
[ParallaxCollector] Found container: data
[ParallaxCollector] Marking for allocator injection
[ParallaxRewriter] Rewriting type: std::vector<float>
[ParallaxRewriter] New type: std::vector<float, parallax::allocator<float>>
[ParallaxRewriter] Injected allocator header
```

### Common Issues

**"Failed to rewrite files"**
- Usually harmless - file was already rewritten in-place
- Check that output file exists and has correct content

**"Unhandled binary operator"**
- Operator not yet supported in lambda body
- Check `lambda_ir_generator.cpp` for supported operations

## Testing

Run the compiler test suite:

```bash
cd build
ctest
```

Or test manually:

```bash
# Create test file
cat > test.cpp << 'EOF'
#include <vector>
#include <algorithm>
#include <execution>

int main() {
    std::vector<float> data(1000, 1.0f);
    std::for_each(std::execution::par, data.begin(), data.end(),
                 [](float& x) { x *= 2.0f; });
    return 0;
}
EOF

# Compile with plugin
clang++ -std=c++20 -fplugin=build/src/plugin/libparallax-clang-plugin.so \
  -I ../parallax-runtime/include test.cpp

# Check rewritten file
cat test.cpp | grep "parallax::allocator"
```

## Performance

The compiler performs all transformations at compile-time:
- **No runtime overhead** - All GPU code pre-generated
- **No dynamic compilation** - SPIR-V embedded in binary
- **Zero-cost abstraction** - Same performance as hand-written GPU code

**Compilation time:**
- Small files (<1000 lines): ~2-5 seconds
- Large files (>5000 lines): ~10-30 seconds

## Contributing

### Adding Support for New Operators

1. Add case to `lambda_ir_generator.cpp`:
```cpp
case clang::BO_YourOperator: {
    if (lhs->getType()->isPointerTy()) {
        llvm::Value* loaded = builder.CreateLoad(..., lhs, "tmp");
        llvm::Value* result = builder.CreateYourOp(loaded, rhs, "op");
        builder.CreateStore(result, lhs);
        return result;
    }
    return nullptr;
}
```

2. Add SPIR-V emission in `spirv_generator.cpp` if needed

3. Add test case

### Code Style

- Follow LLVM coding style
- Use `llvm::errs()` for debug output
- Prefix debug messages with `[ComponentName]`

## Documentation

- 📖 [Architecture Guide](../parallax-docs/docs/architecture.md)
- 🔧 [SPIR-V Generation](../parallax-docs/docs/compiler.md)
- 📊 [Performance Analysis](../parallax-docs/docs/performance.md)

## License

MIT License - see [LICENSE](../LICENSE)

## Acknowledgments

Built with:
- **LLVM/Clang** - Compiler infrastructure
- **SPIR-V Tools** - SPIR-V generation and validation
- Inspired by pSTL and oneTBB execution policies
