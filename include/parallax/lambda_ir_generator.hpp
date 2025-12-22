#ifndef PARALLAX_LAMBDA_IR_GENERATOR_HPP
#define PARALLAX_LAMBDA_IR_GENERATOR_HPP

#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/ASTContext.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/CodeGen/ModuleBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/IRBuilder.h>
#include <memory>
#include <string>
#include <vector>

namespace parallax {

/**
 * Generates LLVM IR from Clang Lambda AST nodes
 * Uses Clang's CodeGen infrastructure for accurate translation
 */
class LambdaIRGenerator {
public:
    explicit LambdaIRGenerator(clang::CompilerInstance& CI);
    ~LambdaIRGenerator();

    /**
     * Generate LLVM IR for a lambda expression
     * @param lambda The lambda expression AST node
     * @param context Clang AST context
     * @return LLVM Module containing the lambda function
     */
    std::unique_ptr<llvm::Module> generateIR(
        clang::LambdaExpr* lambda,
        clang::ASTContext& context
    );

    /**
     * Generate LLVM IR using manual IR construction (fallback)
     * Useful for simple lambdas when CodeGen is not available
     */
    std::unique_ptr<llvm::Module> generateIRManual(
        clang::LambdaExpr* lambda,
        clang::ASTContext& context
    );

    /**
     * Extract information about lambda captures
     */
    struct CaptureInfo {
        std::string name;
        clang::QualType type;
        bool is_by_reference;
    };
    std::vector<CaptureInfo> extractCaptures(clang::LambdaExpr* lambda);

    /**
     * Get the lambda call operator method
     */
    clang::CXXMethodDecl* getLambdaCallOperator(clang::LambdaExpr* lambda);

private:
    clang::CompilerInstance& CI_;
    std::unique_ptr<llvm::LLVMContext> llvm_context_;

    // Helper: Convert Clang type to LLVM type
    llvm::Type* convertType(clang::QualType type);

    // Helper: Translate statement to LLVM IR
    void translateStmt(
        clang::Stmt* stmt,
        llvm::IRBuilder<>& builder,
        clang::ASTContext& context,
        std::map<const clang::VarDecl*, llvm::Value*>& var_map
    );

    // Helper: Translate expression to LLVM IR
    llvm::Value* translateExpr(
        clang::Expr* expr,
        llvm::IRBuilder<>& builder,
        clang::ASTContext& context,
        std::map<const clang::VarDecl*, llvm::Value*>& var_map
    );
};

} // namespace parallax

#endif // PARALLAX_LAMBDA_IR_GENERATOR_HPP
