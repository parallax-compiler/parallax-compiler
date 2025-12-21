#include "parallax/execution_policy.hpp"
#include "parallax/lambda_compiler.hpp"
#include "parallax/kernel_launcher.hpp"
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

// Global launcher pointer for template header access
KernelLauncher* g_global_launcher_ptr = nullptr;
static std::unique_ptr<KernelLauncher> g_launcher_storage;

void ExecutionPolicyImpl::initialize(VulkanBackend* backend, MemoryManager* memory) {
    if (!g_lambda_compiler) {
        g_lambda_compiler = std::make_unique<LambdaCompiler>();
    }
    
    // Create launcher if backend is provided
    if (backend && memory) {
        g_launcher_storage = std::make_unique<KernelLauncher>(backend, memory);
        g_global_launcher_ptr = g_launcher_storage.get();
    }
}

// Template instantiations are in the header as inline implementations
// The actual runtime integration with KernelLauncher happens in the samples

} // namespace parallax
