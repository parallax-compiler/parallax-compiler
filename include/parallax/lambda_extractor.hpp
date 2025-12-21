#ifndef PARALLAX_LAMBDA_EXTRACTOR_HPP
#define PARALLAX_LAMBDA_EXTRACTOR_HPP

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <memory>
#include <string>
#include <vector>

namespace parallax {

// Represents an extracted lambda function
struct ExtractedLambda {
    std::string name;
    std::string source_location;
    std::unique_ptr<llvm::Module> ir_module;
    std::vector<std::string> captured_variables;
    std::string return_type;
    std::vector<std::string> parameter_types;
};

// AST Visitor to find lambda expressions
class LambdaVisitor : public clang::RecursiveASTVisitor<LambdaVisitor> {
public:
    explicit LambdaVisitor(clang::ASTContext* context);
    
    bool VisitLambdaExpr(clang::LambdaExpr* lambda);
    bool VisitCallExpr(clang::CallExpr* call);
    
    const std::vector<ExtractedLambda>& get_lambdas() const { return lambdas_; }
    
private:
    clang::ASTContext* context_;
    std::vector<ExtractedLambda> lambdas_;
    
    bool is_parallel_algorithm(const std::string& func_name);
    ExtractedLambda extract_lambda_info(clang::LambdaExpr* lambda);
};

// AST Consumer
class LambdaConsumer : public clang::ASTConsumer {
public:
    explicit LambdaConsumer(clang::ASTContext* context);
    
    void HandleTranslationUnit(clang::ASTContext& context) override;
    
    const std::vector<ExtractedLambda>& get_lambdas() const {
        return visitor_.get_lambdas();
    }
    
private:
    LambdaVisitor visitor_;
};

// Frontend Action
class LambdaExtractorAction : public clang::ASTFrontendAction {
public:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
        clang::CompilerInstance& compiler,
        llvm::StringRef file) override;
};

// Main API for lambda extraction
class LambdaExtractor {
public:
    LambdaExtractor();
    ~LambdaExtractor();
    
    // Extract lambdas from C++ source file
    std::vector<ExtractedLambda> extract_from_file(const std::string& filename);
    
    // Extract lambdas from C++ source code
    std::vector<ExtractedLambda> extract_from_source(const std::string& source);
    
private:
    std::unique_ptr<llvm::LLVMContext> llvm_context_;
};

} // namespace parallax

#endif // PARALLAX_LAMBDA_EXTRACTOR_HPP
