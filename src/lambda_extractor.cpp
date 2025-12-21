#include "parallax/lambda_extractor.hpp"
#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Tooling/Tooling.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>

namespace parallax {

LambdaVisitor::LambdaVisitor(clang::ASTContext* context)
    : context_(context) {}

bool LambdaVisitor::VisitLambdaExpr(clang::LambdaExpr* lambda) {
    ExtractedLambda extracted = extract_lambda_info(lambda);
    lambdas_.push_back(std::move(extracted));
    return true;
}

bool LambdaVisitor::VisitCallExpr(clang::CallExpr* call) {
    if (auto* callee = call->getDirectCallee()) {
        std::string func_name = callee->getNameAsString();
        
        if (is_parallel_algorithm(func_name)) {
            for (unsigned i = 0; i < call->getNumArgs(); ++i) {
                if (auto* lambda = clang::dyn_cast<clang::LambdaExpr>(
                        call->getArg(i)->IgnoreImplicit())) {
                    ExtractedLambda extracted = extract_lambda_info(lambda);
                    extracted.name = func_name + "_lambda_" + std::to_string(lambdas_.size());
                    lambdas_.push_back(std::move(extracted));
                }
            }
        }
    }
    return true;
}

bool LambdaVisitor::is_parallel_algorithm(const std::string& func_name) {
    return func_name == "for_each" || 
           func_name == "transform" ||
           func_name == "reduce" ||
           func_name == "transform_reduce";
}

ExtractedLambda LambdaVisitor::extract_lambda_info(clang::LambdaExpr* lambda) {
    ExtractedLambda result;
    
    auto loc = lambda->getBeginLoc();
    result.source_location = loc.printToString(context_->getSourceManager());
    
    auto* lambda_class = lambda->getLambdaClass();
    
    for (auto& capture : lambda->captures()) {
        if (capture.capturesVariable()) {
            auto* var = capture.getCapturedVar();
            result.captured_variables.push_back(var->getNameAsString());
        }
    }
    
    auto* call_op = lambda_class->getLambdaCallOperator();
    result.return_type = call_op->getReturnType().getAsString();
    
    for (auto* param : call_op->parameters()) {
        result.parameter_types.push_back(param->getType().getAsString());
    }
    
    result.name = "lambda_" + std::to_string(reinterpret_cast<uintptr_t>(lambda));
    
    // Generate LLVM IR for the lambda
    // This is where we'd use Clang's CodeGen to generate IR
    // For now, create a placeholder module
    result.ir_module = std::make_unique<llvm::Module>(result.name, *llvm_context_);
    
    return result;
}

LambdaConsumer::LambdaConsumer(clang::ASTContext* context)
    : visitor_(context) {}

void LambdaConsumer::HandleTranslationUnit(clang::ASTContext& context) {
    visitor_.TraverseDecl(context.getTranslationUnitDecl());
}

std::unique_ptr<clang::ASTConsumer> LambdaExtractorAction::CreateASTConsumer(
    clang::CompilerInstance& compiler,
    llvm::StringRef file) {
    return std::make_unique<LambdaConsumer>(&compiler.getASTContext());
}

// Global LLVM context for lambda extraction
static llvm::LLVMContext g_llvm_context;

LambdaExtractor::LambdaExtractor()
    : llvm_context_(&g_llvm_context) {}

LambdaExtractor::~LambdaExtractor() = default;

std::vector<ExtractedLambda> LambdaExtractor::extract_from_file(
    const std::string& filename) {
    
    // Create a simple compilation database
    std::vector<std::string> args = {
        "clang++",
        "-std=c++20",
        "-I/usr/include",
        filename
    };
    
    // Use Clang tooling to parse and extract
    auto action = std::make_unique<LambdaExtractorAction>();
    
    // For a complete implementation, we would:
    // 1. Set up proper CompilerInstance
    // 2. Run the action on the file
    // 3. Extract the results
    
    // Simplified version that works with the framework
    std::vector<ExtractedLambda> results;
    
    // In production, this would use clang::tooling::runToolOnCodeWithArgs
    // and properly extract the lambda information
    
    return results;
}

std::vector<ExtractedLambda> LambdaExtractor::extract_from_source(
    const std::string& source) {
    
    // Parse source code and extract lambdas
    auto action = std::make_unique<LambdaExtractorAction>();
    
    // Use Clang's tooling API to parse the source
    // This would create a CompilerInstance, parse the code,
    // and run our visitor to extract lambda information
    
    std::vector<ExtractedLambda> results;
    return results;
}

} // namespace parallax
