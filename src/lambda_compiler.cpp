#include "parallax/lambda_compiler.hpp"
#include "parallax/spirv_generator.hpp"
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

namespace parallax {

LambdaCompiler::LambdaCompiler()
    : context_(std::make_unique<llvm::LLVMContext>()) {}

LambdaCompiler::~LambdaCompiler() = default;

// Explicit instantiation for common lambda types will be in execution_policy.cpp

} // namespace parallax
