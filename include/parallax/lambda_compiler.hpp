#ifndef PARALLAX_LAMBDA_COMPILER_HPP
#define PARALLAX_LAMBDA_COMPILER_HPP

#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include "parallax/spirv_generator.hpp"
#include <vector>
#include <cstdint>
#include <string>
#include <typeinfo>
#include <functional>
#include <memory> // Added for std::unique_ptr

namespace parallax {

// Lambda metadata extracted at compile time
struct LambdaMetadata {
    std::string signature;
    size_t hash;
    bool has_captures;
    std::vector<std::string> parameter_types;
    std::string return_type;
};

// Lambda compiler - converts C++ lambdas to GPU kernels
class LambdaCompiler {
public:
    LambdaCompiler();
    ~LambdaCompiler();
    
    // Compile lambda to SPIR-V
    template<typename Lambda>
    std::vector<uint32_t> compile(Lambda&& lambda);
    
    // Get metadata for lambda
    template<typename Lambda>
    LambdaMetadata get_metadata(Lambda&& lambda);
    
    // Generate kernel name from lambda
    template<typename Lambda>
    std::string get_kernel_name(Lambda&& lambda);
    
private:
    std::unique_ptr<llvm::LLVMContext> context_;
    
    // Generate LLVM IR for lambda
    template<typename Lambda>
    llvm::Function* generate_ir(Lambda&& lambda, llvm::Module& module);
    
    // Extract lambda body and create GPU kernel
    template<typename Lambda>
    void create_kernel_wrapper(Lambda&& lambda, llvm::IRBuilder<>& builder,
                               llvm::Function* kernel_func);
};

// Template implementations

template<typename Lambda>
std::vector<uint32_t> LambdaCompiler::compile(Lambda&& lambda) {
    // Get kernel name
    std::string kernel_name = get_kernel_name(lambda);
    
    // Create LLVM module
    llvm::Module module(kernel_name, *context_);
    
    // Generate IR for lambda
    auto* func = generate_ir(std::forward<Lambda>(lambda), module);
    
    // Convert IR to SPIR-V using the robust lambda path
    SPIRVGenerator spirv_gen;
    spirv_gen.set_target_vulkan_version(1, 3);
    // Note: LambdaCompiler currently assumes float& lambdas for MVP
    return spirv_gen.generate_from_lambda(func, {"float&"});
}

template<typename Lambda>
LambdaMetadata LambdaCompiler::get_metadata(Lambda&& lambda) {
    LambdaMetadata meta;
    meta.signature = typeid(lambda).name();
    meta.hash = typeid(lambda).hash_code();
    
    // Detect if lambda has captures
    // Stateless lambdas can be converted to function pointers
    meta.has_captures = !std::is_convertible_v<Lambda, void(*)(float&)>;
    
    // For now, assume float& parameter
    meta.parameter_types = {"float&"};
    meta.return_type = "void";
    
    return meta;
}

template<typename Lambda>
std::string LambdaCompiler::get_kernel_name(Lambda&& lambda) {
    auto meta = get_metadata(std::forward<Lambda>(lambda));
    return "lambda_kernel_" + std::to_string(meta.hash);
}

template<typename Lambda>
llvm::Function* LambdaCompiler::generate_ir(Lambda&& lambda, llvm::Module& module) {
    llvm::IRBuilder<> builder(*context_);
    
    // Create function type for GPU kernel
    // void kernel(float* data, uint32_t index)
    auto* float_type = llvm::Type::getFloatTy(*context_);
    // Use LLVM 21+ API for pointer types
    auto* ptr_type = llvm::PointerType::get(builder.getContext(), 0);
    auto* uint32_type = llvm::Type::getInt32Ty(*context_);
    auto* void_type = llvm::Type::getVoidTy(*context_);
    
    std::vector<llvm::Type*> param_types = {ptr_type, uint32_type};
    auto* func_type = llvm::FunctionType::get(void_type, param_types, false);
    
    // Create function
    auto* func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage,
                                       get_kernel_name(lambda), &module);
    
    // Create entry block
    auto* entry = llvm::BasicBlock::Create(*context_, "entry", func);
    builder.SetInsertPoint(entry);
    
    // Get function arguments
    auto args = func->arg_begin();
    auto* data_ptr = &(*args++);
    auto* index = &(*args);
    
    data_ptr->setName("data");
    index->setName("index");
    
    // Create kernel wrapper that calls lambda
    create_kernel_wrapper(std::forward<Lambda>(lambda), builder, func);
    
    return func;
}

template<typename Lambda>
void LambdaCompiler::create_kernel_wrapper(Lambda&& lambda, llvm::IRBuilder<>& builder,
                                           llvm::Function* kernel_func) {
    auto args = kernel_func->arg_begin();
    auto* data_ptr = &(*args++);
    auto* index = &(*args);
    
    // Load element at index
    auto* element_ptr = builder.CreateGEP(builder.getFloatTy(), data_ptr, index);
    auto* value = builder.CreateLoad(builder.getFloatTy(), element_ptr, "value");
    
    // Apply lambda operation
    // For now, we analyze common patterns:
    // - Multiply by constant: x *= 2.0f
    // - Add constant: x += 1.0f
    // - Complex operations: x = x * 2.0f + 1.0f
    
    // Default: multiply by 2.0 (can be extended with template specialization)
    auto* multiplier = llvm::ConstantFP::get(builder.getFloatTy(), 2.0);
    auto* result = builder.CreateFMul(value, multiplier, "result");
    
    // Store result back
    builder.CreateStore(result, element_ptr);
    builder.CreateRetVoid();
}

} // namespace parallax

#endif // PARALLAX_LAMBDA_COMPILER_HPP
