#include "parallax/execution_policy.hpp"
#include "parallax/lambda_extractor.hpp"
#include "parallax/spirv_generator.hpp"
#include <parallax/kernel_launcher.hpp>
#include <parallax/vulkan_backend.hpp>
#include <parallax/unified_buffer.hpp>

namespace parallax {

ExecutionPolicyImpl& ExecutionPolicyImpl::instance() {
    static ExecutionPolicyImpl impl;
    return impl;
}

void ExecutionPolicyImpl::initialize(VulkanBackend* backend, MemoryManager* memory) {
    backend_ = backend;
    memory_ = memory;
    if (backend_ && memory_) {
        launcher_ = std::make_unique<KernelLauncher>(backend_, memory_);
    }
}

template<typename Iterator, typename UnaryFunction>
void ExecutionPolicyImpl::for_each(parallax_execution_policy,
                                    Iterator first, Iterator last,
                                    UnaryFunction f) {
    if (!launcher_) {
        // Fallback to CPU
        std::for_each(std::execution::seq, first, last, f);
        return;
    }
    
    // Get data pointer and size
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    value_type* data = &(*first);
    size_t count = std::distance(first, last);
    
    // In a full implementation, we would:
    // 1. Extract lambda IR using LambdaExtractor
    // 2. Generate SPIR-V using SPIRVGenerator
    // 3. Load and launch kernel
    
    // For now, use pre-compiled kernel if available
    // This is a placeholder - full implementation would do JIT compilation
    
    // Fallback to CPU for now
    std::for_each(std::execution::seq, first, last, f);
}

template<typename InputIt, typename OutputIt, typename UnaryOperation>
OutputIt ExecutionPolicyImpl::transform(parallax_execution_policy,
                                        InputIt first, InputIt last,
                                        OutputIt d_first,
                                        UnaryOperation unary_op) {
    if (!launcher_) {
        return std::transform(std::execution::seq, first, last, d_first, unary_op);
    }
    
    // Similar to for_each - would do JIT compilation in full implementation
    return std::transform(std::execution::seq, first, last, d_first, unary_op);
}

// Explicit template instantiations for common types
template void ExecutionPolicyImpl::for_each<float*, void(*)(float&)>(
    parallax_execution_policy, float*, float*, void(*)(float&));

template float* ExecutionPolicyImpl::transform<float*, float*, float(*)(float)>(
    parallax_execution_policy, float*, float*, float*, float(*)(float));

} // namespace parallax
