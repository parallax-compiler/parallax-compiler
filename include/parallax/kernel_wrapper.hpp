#ifndef PARALLAX_KERNEL_WRAPPER_HPP
#define PARALLAX_KERNEL_WRAPPER_HPP

#include "class_context_extractor.hpp"
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <memory>

namespace parallax {

/**
 * Generates a GPU-compatible wrapper around a function object's operator()
 *
 * Transforms:
 *   struct Functor {
 *       float omega;
 *       void operator()(CellData& cell) { /* uses omega */ }
 *   };
 *
 * Into GPU kernel:
 *   void kernel(CellData* cells, size_t index, float omega) {
 *       CellData& cell = cells[index];
 *       // inlined operator() body with omega as parameter
 *   }
 */
class KernelWrapper {
public:
    KernelWrapper(llvm::LLVMContext& ctx) : llvm_ctx_(ctx) {}

    /**
     * Generate wrapper kernel
     * @param context Class context with member info
     * @param module LLVM module containing operator() implementation
     * @return Wrapped kernel function ready for SPIR-V conversion
     */
    llvm::Function* generateWrapper(
        const ClassContext& context,
        llvm::Module* module
    );

private:
    llvm::LLVMContext& llvm_ctx_;

    // Generate kernel signature with captured members as parameters
    llvm::FunctionType* generateKernelSignature(
        const ClassContext& context,
        llvm::Module* module
    );

    // Generate parameter marshalling code
    void generateParameterMarshalling(
        llvm::Function* wrapper,
        const ClassContext& context
    );

    // Inline the operator() body
    void inlineOperatorBody(
        llvm::Function* wrapper,
        llvm::Function* operator_impl
    );
};

} // namespace parallax

#endif // PARALLAX_KERNEL_WRAPPER_HPP
