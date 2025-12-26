#include "ParallaxPlugin.h"
#include <clang/Frontend/FrontendPluginRegistry.h>
#include <clang/AST/Decl.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Lex/Lexer.h>
#include <iostream>
#include <llvm/Support/raw_ostream.h>

// Global initializer to confirm plugin loading
struct GlobalInit {
    GlobalInit() {
        llvm::errs() << "Parallax: Plugin Library Shared Object Loaded\n";
    }
} g_parallax_init;

namespace parallax {

bool ParallaxASTVisitor::VisitCallExpr(clang::CallExpr *Call) {
    if (!Call) return false;
    clang::FunctionDecl *FD = Call->getDirectCallee();
    
    if (isParallaxCandidate(Call)) {
        std::string Name = FD->getNameInfo().getName().getAsString();
        llvm::errs() << "Parallax: [MATCHED] Candidate STL call: " << Name << "\n";
        
        // Rewriting logic is temporarily disabled for debugging
        /*
        std::string NewCall;
        ...
        */
    }
    return true;
}

std::string ParallaxASTVisitor::getSourceText(clang::SourceLocation Start, clang::SourceLocation End) {
    clang::SourceLocation RealEnd = clang::Lexer::getLocForEndOfToken(End, 0, Context->getSourceManager(), Context->getLangOpts());
    if (Start.isInvalid() || RealEnd.isInvalid()) return "";
    return std::string(Context->getSourceManager().getCharacterData(Start),
                       Context->getSourceManager().getCharacterData(RealEnd) - Context->getSourceManager().getCharacterData(Start));
}

bool ParallaxASTVisitor::isParallaxCandidate(clang::CallExpr *Call) {
    if (!Call) return false;
    clang::FunctionDecl *FD = Call->getDirectCallee();
    if (!FD) return false;
    
    std::string Name = FD->getQualifiedNameAsString();
    
    // Check if it's for_each or transform in std namespace
    if (Name != "std::for_each" && Name != "std::transform" && Name != "std::reduce" &&
        Name != "std::execution::for_each" && Name != "std::execution::transform") {
        return false;
    }
    
    if (Call->getNumArgs() < 3) return false;
    
    auto* Arg0 = Call->getArg(0)->IgnoreImplicit();
    std::string ArgType = Arg0->getType().getAsString();
    
    if (ArgType.find("parallel_policy") != std::string::npos ||
        ArgType.find("par") != std::string::npos) {
        return true;
    }
    
    return false;
}

} // namespace parallax

// Old plugin disabled - V2 plugin is now active
// static clang::FrontendPluginRegistry::Add<parallax::ParallaxPluginAction>
//     X("parallax-plugin", "Parallax Automatic Offload Plugin");
