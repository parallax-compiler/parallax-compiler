# pSTL-Bench × Parallax GPU-offload — CI harness

Build and run the **full** [pSTL-Bench](https://github.com/parlab-tuwien/pSTL-Bench)
Google-Benchmark suite compiled through the Parallax transparent-offload toolchain,
dispatching kernels on software Vulkan (lavapipe) in CI.

The mechanism is the one already CI-verified in
`parallax-compiler/.github/workflows/integration-probe.yml`: a Clang plugin routes
`std::for_each(std::execution::par,...)` → `parallax::for_each`, funnels a SPIR-V
kernel registrar, and the runtime dispatches on Vulkan. Here we wrap that dance in
a `CMAKE_CXX_COMPILER` shim (`scripts/parallax-cxx`) so CMake drives it for us.

pSTL-Bench is header-only benchmarks (`include/pstl/benchmarks/**`) `#include`d into a
single TU `src/main.cpp`; the `*_std.h` wrappers call `std::<algo>(std::execution::par,...)`.
So the wrapper's PASS 1 rewrites those shared headers **in place** while compiling
`main.cpp`. That in-place rewrite is why the build **must be serialized (`-j1`)**.

---

## 0. Assumptions / layout

Sibling checkouts, mirroring the integration-probe job:

```
$GITHUB_WORKSPACE/
  parallax-compiler/     # this repo (contains scripts/parallax-cxx)
  parallax-runtime/      # checked out from parallax-compiler/parallax-runtime @ main
```

Runner: `ubuntu-latest`. Network access is REQUIRED (CPM fetches google/benchmark
and CPM.cmake itself over https).

---

## 1. Install toolchain (LLVM 21 + lavapipe + TBB + cmake/ninja)

Reference: `integration-probe.yml` → "Install LLVM/Clang 21, Vulkan + lavapipe,
spirv-tools". We add `libtbb-dev` (link-time safety for the host fallback) and
`libomp-*-dev` (pSTL-Bench's GNU backend `find_package(OpenMP REQUIRED)`).

```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 21 all
sudo apt-get update
sudo apt-get install -y cmake ninja-build \
  libvulkan-dev vulkan-tools mesa-vulkan-drivers \
  vulkan-validationlayers spirv-tools \
  libtbb-dev libomp-dev libnuma-dev
```

Sanity-check the software ICD (identical to the probe's "Show Vulkan ICDs" step):

```bash
echo "== ICD manifests =="; ls -la /usr/share/vulkan/icd.d/ || true
echo "== vulkaninfo =="; vulkaninfo --summary 2>&1 | head -40 || echo "vulkaninfo failed"
```

---

## 2. Build the Parallax runtime and plugin

Verbatim from `integration-probe.yml` ("Build runtime" / "Build plugin"):

```bash
# Runtime (provides libparallax-runtime.so + include/parallax/stdpar.hpp)
cmake -S parallax-runtime -B parallax-runtime/out -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPARALLAX_BUILD_TESTS=OFF
cmake --build parallax-runtime/out -j

# Plugin (the ReplaceAction that routes/funnels)
cmake -S parallax-compiler -B parallax-compiler/out -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm \
  -DClang_DIR=/usr/lib/llvm-21/lib/cmake/clang
cmake --build parallax-compiler/out -j
```

---

## 3. Export the env the wrapper needs

`scripts/parallax-cxx` reads exactly these:

```bash
export CLANGXX=/usr/lib/llvm-21/bin/clang++
export PARALLAX_PLUGIN="$PWD/parallax-compiler/out/src/plugin/libparallax-clang-plugin.so"
export PARALLAX_RT_INCLUDE="$PWD/parallax-runtime/include"   # holds parallax/stdpar.hpp
chmod +x parallax-compiler/scripts/parallax-cxx
# optional: export PARALLAX_CXX_DEBUG=1   # trace the PASS1/PASS2/BUILD sub-commands
```

Runtime dispatch env (from the probe job):

```bash
export LIBGL_ALWAYS_SOFTWARE=1
export LD_LIBRARY_PATH="$PWD/parallax-runtime/out:$LD_LIBRARY_PATH"
# Let the Vulkan loader auto-discover the lavapipe ICD in /usr/share/vulkan/icd.d/.
# Do NOT set VK_ICD_FILENAMES to a guessed path — the probe found that leaves the
# loader with zero ICDs (VK_ERROR_INCOMPATIBLE_DRIVER, -9).
```

---

## 4. Clone and configure pSTL-Bench

```bash
git clone --depth 1 https://github.com/parlab-tuwien/pSTL-Bench /tmp/pstlb

# Pre-fetch CPM cache location so repeat configures don't re-download (see risks).
export CPM_SOURCE_CACHE="$PWD/.cpm-cache"

cmake -S /tmp/pstlb -B /tmp/pstlb/out -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$PWD/parallax-compiler/scripts/parallax-cxx" \
  -DPSTL_BENCH_BACKEND=GNU \
  -DPSTL_BENCH_DATA_TYPE=float \
  -DPSTL_BENCH_SEQUENTIAL=OFF \
  -DPSTL_BENCH_MIN_INPUT_SIZE=1024 \
  -DPSTL_BENCH_MAX_INPUT_SIZE=65536 \
  -DPSTL_BENCH_RANGE_MULTIPLIER=8
```

Notes on the flags (read from `/tmp/pstlb/CMakeLists.txt`):
- `PSTL_BENCH_BACKEND=GNU` selects the `GNU.cmake` backend, which only does
  `find_package(OpenMP REQUIRED)` and defines `PSTL_BENCH_USE_GNU`. The `*_std.h`
  wrappers still emit stock `std::<algo>(std::execution::par,...)`, which is what
  the Parallax plugin intercepts. (An **empty** backend also works — it just skips
  the OpenMP find — but GNU is the closest "CPU parallel std" shape and keeps
  `benchmark_naming` populated.) Avoid TBB here: `TBB.cmake` pulls the libstdc++
  PSTL headers (`PSTL_BENCH_USE_PSTL`), whose own `par` implementation can shadow
  what we want the plugin to route; GNU keeps the call sites plain.
- `PSTL_BENCH_DATA_TYPE=float` → `-DPSTL_ELEM_T=float` (the plugin's f32 path is a
  verified GATE; `int`/`double` also have GATEs if you want a matrix).
- `PSTL_BENCH_SEQUENTIAL=OFF` so we measure/offload the `par` path, not the seq one.
- The MIN/MAX/RANGE keep the Google-Benchmark input-size sweep tiny so a `-j1`
  build + lavapipe run finish inside a CI budget.
- CMake caches `CMAKE_CXX_COMPILER`; the shim is used for **every** TU including
  google/benchmark's, which is fine — those TUs contain no `std::par` and hit the
  wrapper's passthrough compile (PASS1/2 rewrite nothing, real compile proceeds).

---

## 5. Build — SERIALIZED (mandatory)

```bash
# -j1 / --parallel 1 is REQUIRED: PASS 1 rewrites shared headers IN PLACE; a
# parallel build races on a half-rewritten header and corrupts the sources.
cmake --build /tmp/pstlb/out --parallel 1 -v 2>&1 | tee build.log
```

Expect PASS1/PASS2 plugin chatter on stderr (routed refs, registrars). A clean
build produces `/tmp/pstlb/out/pSTL-Bench`.

---

## 6. Run on lavapipe and capture which benchmarks offloaded

```bash
export LIBGL_ALWAYS_SOFTWARE=1
export LD_LIBRARY_PATH="$PWD/parallax-runtime/out:$LD_LIBRARY_PATH"

# Small sweep; keep repetitions low so lavapipe finishes quickly.
PARALLAX_DEBUG=1 /tmp/pstlb/out/pSTL-Bench \
  --benchmark_min_time=0.01s \
  --benchmark_repetitions=1 \
  2>&1 | tee run.log

# Which kernels actually dispatched to the GPU vs fell back to host serial std::
echo "== offloaded kernels ==";      grep -c "Successfully loaded kernel" run.log || true
echo "== host-fallback / misses =="; grep -aE "MISS|fallback" run.log || true
echo "== benchmarks that ran ==";    grep -aE "^BM_|_std/|/manual_time" run.log | head -40 || true
```

Success signal per benchmark family: at least one `Successfully loaded kernel`
line correlated with a `*_std` benchmark, and Google-Benchmark reporting the row
without a verification abort. pSTL-Bench's own `verification.h` checks results
against a serial reference; a wrong-result offload will surface as a benchmark
error/abort, so a clean full run == correctness across the offloaded algorithms.
Un-offloaded algorithms silently fall back to serial `std::` inside `parallax::`
and still report — so "the suite ran to completion" is the floor, "N kernels
loaded" is the real coverage number.

---

## 7. Likely failure modes and mitigations

1. **CPM network fetch fails** (`get_cpm.cmake` downloads CPM.cmake; `CMakeLists.txt`
   `CPMAddPackage(google/benchmark 1.9.5)`). If the runner has no/limited network,
   configure aborts with `Failed to add google benchmark`.
   *Mitigate:* set `CPM_SOURCE_CACHE` to a persisted/cached dir (step 4) and warm it
   in a network-enabled step; or vendor benchmark and pass
   `-DCPM_benchmark_SOURCE=/path`. Keep the configure step on a networked runner.

2. **`-j1` slowness / job timeout.** Serial build of `main.cpp` (which transitively
   includes the whole benchmark set) plus google/benchmark is slow, and each
   Parallax TU pays THREE clang invocations (PASS1 + PASS2 + real).
   *Mitigate:* trim the input-size range (step 4), build `benchmark` first in a
   normal parallel `cmake --build ... --target benchmark -j` step (it has no
   `std::par`, so racing is safe) THEN do the final `--parallel 1` link/compile of
   pSTL-Bench itself; raise the GitHub job `timeout-minutes`.

3. **Header-rewrite corruption on rebuild / incremental builds.** PASS 1 edits
   headers in place; a second configure+build over the same tree re-routes an
   already-routed source (`parallax::parallax::...`) or leaves duplicate registrars,
   and a crashed/parallel build can leave a header half-rewritten.
   *Mitigate:* treat the checkout as **single-use** — always build from a fresh
   `git clone` (or `git checkout -- .` / `git clean -fdx` the pSTL-Bench tree)
   before each configure; never reuse `/tmp/pstlb` across builds; enforce
   `--parallel 1`. If a build fails mid-way, discard the tree, don't retry in place.

Secondary: OpenMP not found (add `libomp-dev`, step 1); lavapipe reporting zero
devices (don't override `VK_ICD_FILENAMES`, step 3); NUMA header present pulls
`-lnuma` (install `libnuma-dev`).
