#ifndef PARALLAX_PLUGIN_ACTION_H
#define PARALLAX_PLUGIN_ACTION_H

#include <clang/Frontend/FrontendAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <llvm/ADT/StringRef.h>

namespace parallax {

class ParallaxASTVisitor : public clang::RecursiveASTVisitor<ParallaxASTVisitor> {
public:
    explicit ParallaxASTVisitor(clang::ASTContext *Context) : Context(Context) {}

    bool VisitCallExpr(clang::CallExpr *Call);

private:
    clang::ASTContext *Context;
    
    // Checks if the function called is std::for_each with std::execution::par
    bool isParallaxCandidate(clang::CallExpr *Call);
};

class ParallaxASTConsumer : public clang::ASTConsumer {
public:
    explicit ParallaxASTConsumer(clang::ASTContext *Context) : Visitor(Context) {}

    void HandleTranslationUnit(clang::ASTContext &Context) override {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }

private:
    ParallaxASTVisitor Visitor;
};

class ParallaxPluginAction : public clang::PluginASTAction {
protected:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI,
                                                          llvm::StringRef) override {
        return std::make_unique<ParallaxASTConsumer>(&CI.getASTContext());
    }

    bool ParseArgs(const clang::CompilerInstance &CI,
                   const std::vector<std::string> &args) override {
        return true;
    }
};

} // namespace parallax

#endif // PARALLAX_PLUGIN_ACTION_H
