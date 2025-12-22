#ifndef PARALLAX_PLUGIN_ACTION_H
#define PARALLAX_PLUGIN_ACTION_H

#include <clang/Frontend/FrontendAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <llvm/ADT/StringRef.h>

namespace parallax {

// Forward declaration of new rewriter-based consumer
class ParallaxASTConsumerV2;

class ParallaxASTVisitor : public clang::RecursiveASTVisitor<ParallaxASTVisitor> {
public:
    explicit ParallaxASTVisitor(clang::ASTContext *Context) 
        : Context(Context) {
        llvm::errs() << "Parallax: AST Visitor instantiated\n";
    }

    bool VisitCallExpr(clang::CallExpr *Call);
    
    bool shouldVisitTemplateInstantiations() const { return true; }
    bool shouldWalkTypesOfTypeLocs() const { return true; }

private:
    clang::ASTContext *Context;
    
    std::string getSourceText(clang::SourceLocation Start, clang::SourceLocation End);
    
    // Checks if the function called is std::for_each with std::execution::par
    bool isParallaxCandidate(clang::CallExpr *Call);
};

class ParallaxASTConsumer : public clang::ASTConsumer {
public:
    explicit ParallaxASTConsumer(clang::ASTContext *Context) : Visitor(Context) {
        llvm::errs() << "Parallax: AST Consumer instantiated\n";
    }

    void HandleTranslationUnit(clang::ASTContext &Context) override {
        llvm::errs() << "Parallax: HandleTranslationUnit started\n";
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
        llvm::errs() << "Parallax: HandleTranslationUnit finished\n";
    }

private:
    ParallaxASTVisitor Visitor;
};

class ParallaxPluginAction : public clang::PluginASTAction {
protected:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI,
                                                          llvm::StringRef) override {
        llvm::errs() << "Parallax: CreateASTConsumer called\n";
        return std::make_unique<ParallaxASTConsumer>(&CI.getASTContext());
    }

    bool ParseArgs(const clang::CompilerInstance &CI,
                   const std::vector<std::string> &args) override {
        llvm::errs() << "Parallax: ParseArgs called\n";
        return true;
    }

    ActionType getActionType() override {
        return AddBeforeMainAction;
    }
};

} // namespace parallax

#endif // PARALLAX_PLUGIN_ACTION_H
