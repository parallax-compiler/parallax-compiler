#include "parallax/lambda_extractor.hpp"
#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/raw_ostream.h>

namespace parallax {

LambdaVisitor::LambdaVisitor(clang::ASTContext* context)
    : context_(context) {}

bool LambdaVisitor::VisitLambdaExpr(clang::LambdaExpr* lambda) {
    // Extract lambda information
    ExtractedLambda extracted = extract_lambda_info(lambda);
    lambdas_.push_back(std::move(extracted));
    return true;
}

bool LambdaVisitor::VisitCallExpr(clang::CallExpr* call) {
    // Check if this is a call to std::for_each, std::transform, etc.
    if (auto* callee = call->getDirectCallee()) {
        std::string func_name = callee->getNameAsString();
        
        if (is_parallel_algorithm(func_name)) {
            // Found a parallel algorithm call
            // Check if any argument is a lambda
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
    
    // Get source location
    auto loc = lambda->getBeginLoc();
    result.source_location = loc.printToString(context_->getSourceManager());
    
    // Get lambda class
    auto* lambda_class = lambda->getLambdaClass();
    
    // Extract captured variables
    for (auto& capture : lambda->captures()) {
        if (capture.capturesVariable()) {
            auto* var = capture.getCapturedVar();
            result.captured_variables.push_back(var->getNameAsString());
        }
    }
    
    // Get call operator (operator())
    auto* call_op = lambda_class->getLambdaCallOperator();
    
    // Extract return type
    result.return_type = call_op->getReturnType().getAsString();
    
    // Extract parameter types
    for (auto* param : call_op->parameters()) {
        result.parameter_types.push_back(param->getType().getAsString());
    }
    
    // Generate unique name
    result.name = "lambda_" + std::to_string(reinterpret_cast<uintptr_t>(lambda));
    
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

LambdaExtractor::LambdaExtractor()
    : llvm_context_(std::make_unique<llvm::LLVMContext>()) {}

LambdaExtractor::~LambdaExtractor() = default;

std::vector<ExtractedLambda> LambdaExtractor::extract_from_file(
    const std::string& filename) {
    // Use Clang tooling to parse the file
    auto action = std::make_unique<LambdaExtractorAction>();
    
    // Run the action
    if (clang::tooling::runToolOnCode(std::move(action), filename)) {
        // Return extracted lambdas
        // Note: In real implementation, we'd need to get the results from the action
        return {};
    }
    
    return {};
}

std::vector<ExtractedLambda> LambdaExtractor::extract_from_source(
    const std::string& source) {
    // Similar to extract_from_file but with source code
    return {};
}

} // namespace parallax
