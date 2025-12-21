#include "parallax/execution_policy.hpp"
#include "parallax/lambda_compiler.hpp"
#include "parallax/spirv_generator.hpp"
#include <parallax/kernel_launcher.hpp>
#include <parallax/vulkan_backend.hpp>
#include <parallax/unified_buffer.hpp>
#include <unordered_map>
#include <memory>

namespace parallax {

// Global lambda compiler
static std::unique_ptr<LambdaCompiler> g_lambda_compiler;

// Kernel cache: lambda signature → compiled SPIR-V
static std::unordered_map<std::string, std::vector<uint32_t>> g_kernel_cache;

ExecutionPolicyImpl& ExecutionPolicyImpl::instance() {
    static ExecutionPolicyImpl impl;
    return impl;
}

void ExecutionPolicyImpl::initialize(VulkanBackend* backend, MemoryManager* memory) {
    backend_ = backend;
    memory_ = memory;
    if (backend_ && memory_) {
        launcher_ = std::make_unique<KernelLauncher>(backend_, memory_);
        g_lambda_compiler = std::make_unique<LambdaCompiler>();
    }
}

template<typename Iterator, typename UnaryFunction>
void ExecutionPolicyImpl::for_each_impl(Iterator first, Iterator last, UnaryFunction f) {
    if (!launcher_ || !g_lambda_compiler) {
        // Fallback to CPU
        std::for_each(std::execution::seq, first, last, f);
        return;
    }
    
    // Get data pointer and size
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    value_type* data = &(*first);
    size_t count = std::distance(first, last);
    
    try {
        // Get kernel name
        std::string kernel_name = g_lambda_compiler->get_kernel_name(f);
        
        // Check cache
        std::vector<uint32_t> spirv;
        auto it = g_kernel_cache.find(kernel_name);
        if (it != g_kernel_cache.end()) {
            spirv = it->second;
        } else {
            // Compile lambda to SPIR-V
            spirv = g_lambda_compiler->compile(f);
            g_kernel_cache[kernel_name] = spirv;
        }
        
        // Load kernel
        if (!launcher_->load_kernel(kernel_name, spirv.data(), spirv.size() * sizeof(uint32_t))) {
            std::for_each(std::execution::seq, first, last, f);
            return;
        }
        
        // Launch kernel on GPU
        launcher_->launch(kernel_name, data, count, 1.0f);
        
    } catch (...) {
        // Fallback to CPU on any error
        std::for_each(std::execution::seq, first, last, f);
    }
}

template<typename InputIt, typename OutputIt, typename UnaryOperation>
OutputIt ExecutionPolicyImpl::transform_impl(InputIt first, InputIt last,
                                             OutputIt d_first,
                                             UnaryOperation unary_op) {
    if (!launcher_ || !g_lambda_compiler) {
        return std::transform(std::execution::seq, first, last, d_first, unary_op);
    }
    
    try {
        // Get kernel name
        std::string kernel_name = g_lambda_compiler->get_kernel_name(unary_op);
        
        // Check cache
        std::vector<uint32_t> spirv;
        auto it = g_kernel_cache.find(kernel_name);
        if (it != g_kernel_cache.end()) {
            spirv = it->second;
        } else {
            // Compile lambda
            spirv = g_lambda_compiler->compile(unary_op);
            g_kernel_cache[kernel_name] = spirv;
        }
        
        // Load and launch kernel
        using value_type = typename std::iterator_traits<InputIt>::value_type;
        value_type* input = &(*first);
        value_type* output = &(*d_first);
        size_t count = std::distance(first, last);
        
        if (launcher_->load_kernel(kernel_name, spirv.data(), spirv.size() * sizeof(uint32_t))) {
            launcher_->launch(kernel_name, input, count, 1.0f);
            std::copy(input, input + count, output);
            return d_first + count;
        }
    } catch (...) {}
    
    return std::transform(std::execution::seq, first, last, d_first, unary_op);
}

// Explicit template instantiations
template void ExecutionPolicyImpl::for_each_impl<float*, std::function<void(float&)>>(
    float*, float*, std::function<void(float&)>);

template float* ExecutionPolicyImpl::transform_impl<float*, float*, std::function<float(float)>>(
    float*, float*, float*, std::function<float(float)>);

} // namespace parallax

// Interception of std::for_each with std::execution::par
// This is done via template specialization or compiler plugin
// For now, users can explicitly call parallax::ExecutionPolicyImpl::instance().for_each_impl()
// In production, this would be automatic via Clang plugin or LD_PRELOAD
