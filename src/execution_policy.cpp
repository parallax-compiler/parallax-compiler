#include "parallax/execution_policy.hpp"
#include "parallax/lambda_compiler.hpp"
#include "parallax/spirv_generator.hpp"
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <execution>

// NOTE: This is a compiler-side stub for ExecutionPolicyImpl
// The actual implementation with runtime integration is in the samples

namespace parallax {

// Global lambda compiler (stub for compiler module)
static std::unique_ptr<LambdaCompiler> g_lambda_compiler;

// Kernel cache: lambda signature → compiled SPIR-V
static std::unordered_map<std::string, std::vector<uint32_t>> g_kernel_cache;

ExecutionPolicyImpl& ExecutionPolicyImpl::instance() {
    static ExecutionPolicyImpl impl;
    return impl;
}

void ExecutionPolicyImpl::initialize(VulkanBackend* backend, MemoryManager* memory) {
    // Stub implementation - actual runtime integration happens in samples
    // This is just the compiler-side framework
    if (!g_lambda_compiler) {
        g_lambda_compiler = std::make_unique<LambdaCompiler>();
    }
}

// Template instantiations are in the header as inline implementations
// The actual runtime integration with KernelLauncher happens in the samples

} // namespace parallax
