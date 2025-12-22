#ifndef PARALLAX_EXECUTION_POLICY_IMPL_HPP
#define PARALLAX_EXECUTION_POLICY_IMPL_HPP

#include "parallax/execution_policy.hpp"
#include "parallax/lambda_compiler.hpp"
#include "parallax/kernel_launcher.hpp"
#include "parallax/vulkan_backend.hpp"
#include <algorithm>
#include <execution>
#include <unordered_set>
#include <iostream>
#include "parallax/kernel_launcher.hpp"

namespace parallax {

// Implementation of template methods for ExecutionPolicyImpl
// This header bridges the Compiler module and Runtime module

template<typename Iterator, typename UnaryFunction>
void ExecutionPolicyImpl::for_each_impl(Iterator first, Iterator last, UnaryFunction f) {
    // Access runtime components via global instance or singleton logic if needed
    // For this benchmark/sample, we assume the singleton has been initialized
    // but since we removed the members from the header, we need a way to get them.
    
    // In a real plugin scenario, checking for the backend would be robust.
    // Here we'll use a hack for the sample: we expect the user to provide the components
    // via a "RuntimeBridge" or similar, OR we restore the members but as void* to hide types?
    
    // BETTER APPROACH: Fallback to std::for_each only for now to verify linking
    // Real implementation requires the runtime headers to be visible here.
    
    // Since we included kernel_launcher.hpp above, we can use it!
    // But we don't have access to the *instance's* helper pointers if they aren't in the class.
    
    // WORKAROUND: We will use a static global in this translation unit or similar.
    // Or simpler: Just recompile ExecutionPolicy.cpp with runtime support for the sample?
    // No, that's messy.
    
    // Let's rely on the LambdaCompiler directly here.
    
    static LambdaCompiler compiler;
    static std::unordered_map<std::string, std::vector<uint32_t>> cache;
    
    try {
        // 1. Compile
        std::string name = compiler.get_kernel_name(f);
        if (cache.find(name) == cache.end()) {
            std::cerr << "Parallax JIT: Compiling " << name << "..." << std::endl;
            cache[name] = compiler.compile(f);
        }
        
        // 2. Launch
        // Access global launcher defined in execution_policy.cpp
        extern KernelLauncher* g_global_launcher_ptr;
        
        if (g_global_launcher_ptr) {
            // Load if not loaded (Launcher caches but we need to load explicitly first time)
            // Note: Launcher checks 'pipelines_' map, but we need to pass SPIR-V code
            // The loading API is distinct from launch.
            
            // We should modify load_kernel to check cache internally or call it every time.
            // Current load_kernel: pipelines_[name] = data; (overwrites?)
            // We should only load if not present in launcher.
            // But we don't have is_loaded API. Calling load_kernel is safe (just re-creates pipeline).
            // Optimization: Track loaded state locally.
            static std::unordered_set<std::string> loaded_kernels;
            if (loaded_kernels.find(name) == loaded_kernels.end()) {
                g_global_launcher_ptr->load_kernel(name, cache[name].data(), cache[name].size() * 4);
                loaded_kernels.insert(name);
            }
            
            // Assume input is contiguous and 'first' is a pointer
            // CAUTION: This assumes specific iterator type (float* or similar)
            // Ideally we check if it's a pointer to unified memory.
            auto* data_ptr = &(*first);
            size_t count = std::distance(first, last);
            
            if (g_global_launcher_ptr->launch(name, (void*)data_ptr, count)) {
                g_global_launcher_ptr->sync(); // Ensure synchronous completion for ISO compliance
                return; // GPU execution successful
            } else {
                std::cerr << "Parallax JIT: Launch failed for " << name << std::endl;
            }
        }
        
        // Fallback to CPU if launch failed or launcher not available
        std::for_each(std::execution::seq, first, last, f);
        
    } catch (const std::exception& e) {
        std::cerr << "GPU Execution Failed: " << e.what() << std::endl;
        std::for_each(std::execution::seq, first, last, f);
    }
}

template<typename InputIt, typename OutputIt, typename UnaryOperation>
OutputIt ExecutionPolicyImpl::transform_impl(InputIt first, InputIt last,
                                             OutputIt d_first,
                                             UnaryOperation unary_op) {
    static LambdaCompiler compiler;
    static std::unordered_map<std::string, std::vector<uint32_t>> cache;
    static std::unordered_set<std::string> loaded_kernels;
    
    try {
        // 1. Compile with 2 arguments (input, output)
        std::string name = compiler.get_kernel_name(unary_op, 2);
        if (cache.find(name) == cache.end()) {
            cache[name] = compiler.compile(unary_op, 2);
        }
        
        // 2. Launch
        extern KernelLauncher* g_global_launcher_ptr;
        
        if (g_global_launcher_ptr) {
            if (loaded_kernels.find(name) == loaded_kernels.end()) {
                g_global_launcher_ptr->load_kernel(name, cache[name].data(), cache[name].size() * 4);
                loaded_kernels.insert(name);
            }
            
            auto* in_ptr = &(*first);
            auto* out_ptr = &(*d_first);
            size_t count = std::distance(first, last);
            
            if (g_global_launcher_ptr->launch_transform(name, (void*)in_ptr, (void*)out_ptr, count)) {
                g_global_launcher_ptr->sync(); // Ensure synchronous completion for ISO compliance
                return d_first + count; // GPU execution successful
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "GPU Transform Failed: " << e.what() << std::endl;
    }
    
    // Fallback
    return std::transform(std::execution::seq, first, last, d_first, unary_op);
}

template<typename InputIt, typename T, typename BinaryOperation>
T ExecutionPolicyImpl::reduce_impl(InputIt first, InputIt last, T init, BinaryOperation binary_op) {
    // Reduction requires specialized GPU logic (atomics/shuffles)
    // For now, we fallback to CPU seq to maintain results parity
    return std::reduce(std::execution::seq, first, last, init, binary_op);
}

} // namespace parallax

#endif // PARALLAX_EXECUTION_POLICY_IMPL_HPP
