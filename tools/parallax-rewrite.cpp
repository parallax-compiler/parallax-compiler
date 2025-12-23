// Standalone source rewriter for Parallax allocator injection
#include <clang/Tooling/Tooling.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Refactoring.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <llvm/Support/CommandLine.h>

using namespace clang;
using namespace clang::tooling;

static llvm::cl::OptionCategory ParallaxCategory("parallax-rewrite options");

// Simple visitor that detects std::for_each with execution::par
class AllocatorInjectorVisitor : public RecursiveASTVisitor<AllocatorInjectorVisitor> {
public:
    explicit AllocatorInjectorVisitor(Rewriter &R, ASTContext &Context)
        : TheRewriter(R), Context(Context) {}
    
    bool VisitVarDecl(VarDecl *VD) {
        // Check if this is a std::vector declaration
        QualType QT = VD->getType();
        std::string TypeStr = QT.getAsString();
        
        if (TypeStr.find("std::vector") != std::string::npos &&
            TypeStr.find("parallax::allocator") == std::string::npos) {
            
            // Check if this variable is used in a parallel algorithm
            // (simplified: we'll inject for all vectors)
            
            llvm::outs() << "Found vector: " << VD->getNameAsString() 
                        << " of type " << TypeStr << "\n";
            
            // TODO: Rewrite type
            // TypeSourceInfo *TSI = VD->getTypeSourceInfo();
            // if (TSI) {
            //     SourceRange TypeRange = TSI->getTypeLoc().getSourceRange();
            //     TheRewriter.ReplaceText(TypeRange, NewType);
            // }
        }
        
        return true;
    }

private:
    Rewriter &TheRewriter;
    ASTContext &Context;
};

class AllocatorInjectorConsumer : public ASTConsumer {
public:
    explicit AllocatorInjectorConsumer(Rewriter &R, ASTContext &Context)
        : Visitor(R, Context) {}
    
    void HandleTranslationUnit(ASTContext &Context) override {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }

private:
    AllocatorInjectorVisitor Visitor;
};

class AllocatorInjectorAction : public ASTFrontendAction {
public:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(
        CompilerInstance &CI, StringRef file) override {
        TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
        return std::make_unique<AllocatorInjectorConsumer>(
            TheRewriter, CI.getASTContext());
    }
    
    void EndSourceFileAction() override {
        // Write rewritten source to stdout
        TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID())
            .write(llvm::outs());
    }

private:
    Rewriter TheRewriter;
};

int main(int argc, const char **argv) {
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, ParallaxCategory);
    if (!ExpectedParser) {
        llvm::errs() << ExpectedParser.takeError();
        return 1;
    }
    CommonOptionsParser &OptionsParser = ExpectedParser.get();
    ClangTool Tool(OptionsParser.getCompilations(),
                   OptionsParser.getSourcePathList());
    
    return Tool.run(newFrontendActionFactory<AllocatorInjectorAction>().get());
}
