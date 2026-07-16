# Parallax Compiler

Clang plugin and LLVM-to-SPIR-V compiler for automatic GPU offloading of C++ parallel algorithms.

## Overview

The Parallax compiler is a Clang plugin that offloads standard C++20 parallel algorithms
(`std::algo(std::execution::par, …)`) to Vulkan/SPIR-V — the nvc++/dpc++ shape, but for
Vulkan compute. It **separates interception from codegen**, the way mature stdpar
toolchains do:

1. **Interception (a library, not pattern-matching).** Unmodified `std::execution::par`
   calls are transparently routed to `parallax::` overloads that funnel every call through a
   small set of named templates (`parallax::detail::device_*`). The plugin keys on those
   template *instantiations* — deterministic, and robust inside the generic lambda wrappers
   real benchmark suites use (where AST call-site matching is not).
2. **Codegen (our own C++ → SPIR-V backend).** For each funnel instantiation the plugin runs
   Clang CodeGen on the user callable, lowers the LLVM IR to a SPIR-V compute kernel, and
   embeds it in the binary. Heavy primitives (reduce/scan/sort/compaction) are our own
   fixed SPIR-V skeletons.
3. **Software unified memory.** A single host-mapped, GPU-addressable arena stages data;
   pointer values are relocated in-shader (`gpu = dev_base + (host - host_base)`), so the
   model is correct on both integrated (UMA) and discrete GPUs.

No source changes are required: the same ISO C++ that runs on the CPU offloads to the GPU,
and anything the backend can't lower cleanly falls back to a correct CPU execution.

## What works now

Everything below offloads to the GPU from **unmodified** `std::execution::par` code and is
verified end-to-end on software Vulkan (lavapipe) in CI — a 49-gate integration probe plus
the real [pSTL-Bench](https://github.com/parlab-tuwien/pSTL-Bench) suite, which runs to
completion with **20 algorithm instances offloaded** to the GPU.

### Offloading algorithms

| Family | Algorithms | Notes |
|---|---|---|
| **Map (in place)** | `for_each`, `fill`, `generate` | element-wise; `fill`/`generate` bind their captured value/generator |
| **Map (in→out)** | `transform` | unary; **capturing** ops supported; input/output types may differ (e.g. `float`→`double`) |
| **Fold** | `reduce`, `transform_reduce` | default `+`, or a **custom binary op** for `reduce` |
| **Prefix scan** | `inclusive_scan`, `exclusive_scan` | multi-block, default `+` |
| **Sort** | `sort` | bitonic, ascending (default `<`) |
| **Predicate fold** | `count_if`, `all_of`, `any_of`, `none_of` | predicate → count on the GPU |
| **Compaction** | `copy_if`, `remove_if`, `partition`, `unique` | flags → scan → scatter; returns the kept count / partition point |

**Element types:** 32- and 64-bit `float` and integer (`float`, `double`, `int`, `int64`).

### Callable codegen (the C++ → SPIR-V backend)

Arbitrary user lambdas/functors compile directly to SPIR-V, including:

- **Captures** — scalar, 8-byte (`double`/`int64`), and by-value POD **structs** (flattened)
- **Control flow** — `if`/`else` and `for`/`while` loops (structured `OpSelectionMerge` /
  `OpLoopMerge`); ternaries via `OpSelect`
- **Helper calls** — free/member functions inlined into the kernel body
- **Numeric casts** and `<cmath>` math (`sqrt`, `sin`, … → `GLSL.std.450`)
- **Struct elements** — `for_each` over a container of a POD struct
- **Pointer-chasing** — dereferencing a host pointer stored in the data (software UM)

Anything the backend can't lower is left on the CPU (correct ISO fallback), never
mis-compiled.

### Not yet offloaded

- Custom **sort comparator** / custom **scan op** (default `<` / `+` only) → CPU
- Algorithms without a GPU skeleton (search, merge, set operations, `min`/`max_element`,
  `partial_sort`, `transform_{in,ex}clusive_scan`, …) → CPU backend

## Building

### Prerequisites

- **LLVM/Clang:** 21 (CI builds and verifies against LLVM 21; opaque pointers required)
- **CMake:** 3.20+
- **C++ Compiler:** C++20 support
- **SPIR-V Tools** (`spirv-val`/`spirv-dis`) for validating emitted kernels
- A Vulkan 1.2+ device to run (CI uses **lavapipe**, the software rasterizer)

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

The plugin is a Clang **ReplaceAction** (it rewrites the source in place: transparent
routing + embedded SPIR-V), so it is driven with `-Xclang -plugin -Xclang parallax`, **not**
`-fplugin`.

### Example

**Input (your-code.cpp) — stock ISO C++, no `parallax::` names:**
```cpp
#include <vector>
#include <algorithm>
#include <execution>

int main() {
    std::vector<float> data(10000, 1.0f);

    // Offloads to the GPU; identical CPU semantics if no device is present.
    std::for_each(std::execution::par, data.begin(), data.end(),
                  [](float& x) { x = x * 2.0f + 1.0f; });

    return 0;
}
```

**Transparent offload** (`PARALLAX_TRANSPARENT=1`) runs the plugin in two passes — pass 1
routes `std::…(par, …)` to the `parallax::` funnels, pass 2 compiles the now-instantiated
funnels and embeds the SPIR-V — then links against the runtime:

```bash
PLUGIN=build/src/plugin/libparallax-clang-plugin.so
FLAGS="-std=c++20 -I ../parallax-runtime/include -include parallax/stdpar.hpp \
  -Xclang -load -Xclang $PLUGIN -Xclang -plugin -Xclang parallax"

PARALLAX_TRANSPARENT=1 clang++ $FLAGS -c your-code.cpp -o /dev/null   # pass 1: route
PARALLAX_TRANSPARENT=1 clang++ $FLAGS -c your-code.cpp -o /dev/null   # pass 2: funnel + embed
clang++ -std=c++20 -include parallax/stdpar.hpp your-code.cpp \
  -I ../parallax-runtime/include -L ../parallax-runtime/out -lparallax-runtime -o program
```

For whole projects, `scripts/parallax-cxx` is a drop-in `CMAKE_CXX_COMPILER` wrapper that
performs the route → funnel → build passes per translation unit (this is how the pSTL-Bench
harness builds the suite).

**Output:** a native binary with the GPU kernels embedded as SPIR-V.

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

## Compilation model

- **Kernels generated at compile time** — SPIR-V is embedded in the binary; there is no
  runtime shader compilation or dynamic translation layer.
- **Two-pass transparent build** — routing then funnel codegen (see Usage); a project
  wrapper (`scripts/parallax-cxx`) hides this behind a normal compiler invocation.
- **Correctness first** — every emitted kernel is `spirv-val`-clean, and coverage is gated
  in CI on lavapipe. Performance tuning (subgroup reductions, spec-constant workgroup
  sizing, discrete-GPU migration) is on the roadmap, not the current focus.

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
