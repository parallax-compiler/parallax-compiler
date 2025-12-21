#include "ParallaxPlugin.h"
#include <clang/Frontend/FrontendPluginRegistry.h>
#include <clang/AST/Decl.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <iostream>

namespace parallax {

bool ParallaxASTVisitor::VisitCallExpr(clang::CallExpr *Call) {
    if (isParallaxCandidate(Call)) {
        // This is where we will inject the parallax interception code
        // For v0.4.0 alpha, we just log detection
        llvm::outs() << "Parallax: Detected std::for_each with std::execution::par at " 
                     << Call->getBeginLoc().printToString(Context->getSourceManager()) << "\n";
                     
        // TODO: Use Rewriter to replace std::for_each(...) 
        // with parallax::ExecutionPolicyImpl::instance().for_each_impl(...)
    }
    return true;
}

bool ParallaxASTVisitor::isParallaxCandidate(clang::CallExpr *Call) {
    clang::FunctionDecl *FD = Call->getDirectCallee();
    if (!FD) return false;
    
    std::string Name = FD->getNameInfo().getName().getAsString();
    
    // Check if it's for_each
    if (Name != "for_each") return false;
    
    // Check constraints (namespace std, first arg is execution policy)
    // This is a simplified check for the prototype
    
    // TODO: Verify namespace and argument types rigorously
    
    // Check first argument type for std::execution::parallel_policy
    if (Call->getNumArgs() < 3) return false;
    
    auto* Arg0 = Call->getArg(0)->IgnoreImplicit();
    std::string ArgType = Arg0->getType().getAsString();
    
    if (ArgType.find("execution::parallel_policy") != std::string::npos ||
        ArgType.find("execution::par") != std::string::npos) {
        return true;
    }
    
    return false;
}

} // namespace parallax

static clang::FrontendPluginRegistry::Add<parallax::ParallaxPluginAction>
    X("parallax-plugin", "Parallax Automatic Offload Plugin");
