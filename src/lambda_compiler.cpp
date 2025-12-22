#include "parallax/lambda_compiler.hpp"
#include "parallax/spirv_generator.hpp"
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>

namespace parallax {

LambdaCompiler::LambdaCompiler()
    : context_(std::make_unique<llvm::LLVMContext>()) {}

LambdaCompiler::~LambdaCompiler() = default;

// Template implementations are in the header.
// This file can contain explicit instantiations or non-template helpers.

} // namespace parallax
