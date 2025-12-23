// parallax-transform.cpp - AST-based source transformation for Parallax
// Automatically injects parallax::allocator into containers used with parallel algorithms

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/CommandLine.h>
#include <set>
#include <map>

using namespace clang;
using namespace clang::tooling;
using namespace llvm;

static cl::OptionCategory ParallaxCategory("parallax-transform options");
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::extrahelp MoreHelp("\nTransforms C++ source to inject parallax::allocator\n");

// Track which containers are used in parallel algorithms
class ContainerUsageCollector : public RecursiveASTVisitor<ContainerUsageCollector> {
public:
    explicit ContainerUsageCollector(ASTContext& Context)
        : Context(Context) {}

    bool VisitCallExpr(CallExpr* Call) {
        if (!isParallelAlgorithm(Call)) {
            return true;
        }

        errs() << "[ContainerCollector] Found parallel algorithm call\n";

        // Extract iterators and trace to containers
        Expr* FirstIter = nullptr;
        Expr* LastIter = nullptr;

        if (Call->getNumArgs() >= 3) {
            FirstIter = Call->getArg(1);  // Skip execution policy
            LastIter = Call->getArg(2);
        }

        if (FirstIter && LastIter) {
            const VarDecl* Container = traceIteratorToContainer(FirstIter);
            if (Container) {
                errs() << "[ContainerCollector] Marking container: "
                       << Container->getNameAsString() << "\n";
                ContainersNeedingAllocator.insert(Container);
            }
        }

        return true;
    }

    const std::set<const VarDecl*>& getContainersNeedingAllocator() const {
        return ContainersNeedingAllocator;
    }

private:
    ASTContext& Context;
    std::set<const VarDecl*> ContainersNeedingAllocator;

    bool isParallelAlgorithm(CallExpr* Call) {
        if (!Call || !Call->getDirectCallee()) return false;

        std::string Name = Call->getDirectCallee()->getQualifiedNameAsString();
        if (Name != "std::for_each" && Name != "std::transform") {
            return false;
        }

        if (Call->getNumArgs() < 3) return false;

        Expr* Policy = Call->getArg(0)->IgnoreImplicit();
        std::string PolicyType = Policy->getType().getAsString();

        return PolicyType.find("parallel_policy") != std::string::npos ||
               PolicyType.find("parallel_unsequenced_policy") != std::string::npos ||
               PolicyType.find("par") != std::string::npos;
    }

    const VarDecl* traceIteratorToContainer(Expr* IterExpr) {
        if (!IterExpr) return nullptr;

        Expr* E = IterExpr->IgnoreImplicit();

        // Pattern 1: container.begin() / container.end()
        if (auto* MemberCall = dyn_cast<CXXMemberCallExpr>(E)) {
            Expr* Object = MemberCall->getImplicitObjectArgument();
            if (Object) {
                Object = Object->IgnoreImplicit();
                if (auto* DeclRef = dyn_cast<DeclRefExpr>(Object)) {
                    if (auto* VD = dyn_cast<VarDecl>(DeclRef->getDecl())) {
                        return VD;
                    }
                }
            }
        }

        // Pattern 2: std::begin(container) / std::end(container)
        if (auto* Call = dyn_cast<CallExpr>(E)) {
            if (auto* Func = Call->getDirectCallee()) {
                std::string FuncName = Func->getQualifiedNameAsString();
                if ((FuncName == "std::begin" || FuncName == "std::end") &&
                    Call->getNumArgs() >= 1) {
                    Expr* Arg = Call->getArg(0)->IgnoreImplicit();
                    if (auto* DeclRef = dyn_cast<DeclRefExpr>(Arg)) {
                        if (auto* VD = dyn_cast<VarDecl>(DeclRef->getDecl())) {
                            return VD;
                        }
                    }
                }
            }
        }

        // Pattern 3: Raw pointer from .data()
        if (auto* MemberCall = dyn_cast<CXXMemberCallExpr>(E)) {
            if (auto* MethodDecl = MemberCall->getMethodDecl()) {
                if (MethodDecl->getNameAsString() == "data") {
                    Expr* Object = MemberCall->getImplicitObjectArgument();
                    if (Object) {
                        Object = Object->IgnoreImplicit();
                        if (auto* DeclRef = dyn_cast<DeclRefExpr>(Object)) {
                            if (auto* VD = dyn_cast<VarDecl>(DeclRef->getDecl())) {
                                return VD;
                            }
                        }
                    }
                }
            }
        }

        // Pattern 4: Direct variable reference (raw pointer)
        if (auto* DeclRef = dyn_cast<DeclRefExpr>(E)) {
            if (auto* VD = dyn_cast<VarDecl>(DeclRef->getDecl())) {
                // Check if this is a pointer derived from a container
                QualType Type = VD->getType();
                if (Type->isPointerType()) {
                    // Try to trace initialization
                    if (VD->hasInit()) {
                        return traceIteratorToContainer(VD->getInit());
                    }
                }
                return VD;
            }
        }

        return nullptr;
    }
};

// Rewrite container types to inject allocator
class AllocatorInjector : public RecursiveASTVisitor<AllocatorInjector> {
public:
    explicit AllocatorInjector(Rewriter& R, ASTContext& Context,
                               const std::set<const VarDecl*>& Containers)
        : TheRewriter(R), Context(Context), ContainersToRewrite(Containers) {}

    bool VisitVarDecl(VarDecl* VD) {
        if (ContainersToRewrite.find(VD) == ContainersToRewrite.end()) {
            return true;
        }

        if (!canRewriteContainer(VD)) {
            return true;
        }

        QualType OriginalType = VD->getType();
        std::string TypeStr = OriginalType.getAsString();

        // Skip if already has parallax::allocator
        if (TypeStr.find("parallax::allocator") != std::string::npos) {
            errs() << "[AllocatorInjector] Skipping (already has allocator): "
                   << TypeStr << "\n";
            return true;
        }

        errs() << "[AllocatorInjector] Rewriting type: " << TypeStr << "\n";

        // Get type source info
        TypeSourceInfo* TSI = VD->getTypeSourceInfo();
        if (!TSI) {
            errs() << "[AllocatorInjector] Warning: No type source info\n";
            return true;
        }

        SourceRange TypeRange = TSI->getTypeLoc().getSourceRange();

        // Extract element type from template
        QualType ElementType;
        std::string ContainerTemplate;

        QualType BaseType = OriginalType.getNonReferenceType();
        if (const auto* TemplSpec = BaseType->getAs<TemplateSpecializationType>()) {
            if (auto* TemplDecl = TemplSpec->getTemplateName().getAsTemplateDecl()) {
                ContainerTemplate = TemplDecl->getQualifiedNameAsString();
            }

            ArrayRef<TemplateArgument> Args = TemplSpec->template_arguments();
            if (Args.size() > 0 && Args[0].getKind() == TemplateArgument::Type) {
                ElementType = Args[0].getAsType();
            }
        }

        if (ElementType.isNull()) {
            errs() << "[AllocatorInjector] Warning: Could not extract element type\n";
            return true;
        }

        // Build new type with allocator
        std::string ElementTypeStr = ElementType.getAsString();
        std::string NewType;

        if (ContainerTemplate == "std::vector") {
            NewType = "std::vector<" + ElementTypeStr +
                     ", parallax::allocator<" + ElementTypeStr + ">>";
        } else if (ContainerTemplate == "std::deque") {
            NewType = "std::deque<" + ElementTypeStr +
                     ", parallax::allocator<" + ElementTypeStr + ">>";
        } else {
            errs() << "[AllocatorInjector] Warning: Unsupported container: "
                   << ContainerTemplate << "\n";
            return true;
        }

        errs() << "[AllocatorInjector] New type: " << NewType << "\n";

        // Perform the rewrite
        TheRewriter.ReplaceText(TypeRange, NewType);

        return true;
    }

private:
    Rewriter& TheRewriter;
    ASTContext& Context;
    const std::set<const VarDecl*>& ContainersToRewrite;

    bool canRewriteContainer(VarDecl* VD) {
        // Don't rewrite function parameters
        if (isa<ParmVarDecl>(VD)) {
            return false;
        }

        // Don't rewrite global variables
        if (VD->hasGlobalStorage() && !VD->isStaticLocal()) {
            return false;
        }

        return true;
    }
};

// Main AST consumer
class ParallaxTransformConsumer : public ASTConsumer {
public:
    explicit ParallaxTransformConsumer(Rewriter& R, ASTContext& Context)
        : TheRewriter(R), Context(Context), Collector(Context), Injector(R, Context, Containers) {}

    void HandleTranslationUnit(ASTContext& Ctx) override {
        errs() << "[ParallaxTransform] Phase 1: Collecting containers...\n";

        // Phase 1: Find all containers used in parallel algorithms
        Collector.TraverseDecl(Ctx.getTranslationUnitDecl());
        Containers = Collector.getContainersNeedingAllocator();

        errs() << "[ParallaxTransform] Found " << Containers.size()
               << " containers needing allocator injection\n";

        if (Containers.empty()) {
            return;
        }

        // Inject header
        SourceLocation StartLoc = Ctx.getSourceManager().getLocForStartOfFile(
            Ctx.getSourceManager().getMainFileID()
        );

        // Find last #include to insert after it
        SourceLocation InsertLoc = StartLoc;

        // Simple heuristic: insert at start of file
        TheRewriter.InsertTextBefore(InsertLoc,
            "#include <parallax/allocator.hpp>\n");

        errs() << "[ParallaxTransform] Phase 2: Injecting allocators...\n";

        // Phase 2: Rewrite container types
        Injector.TraverseDecl(Ctx.getTranslationUnitDecl());

        errs() << "[ParallaxTransform] Transformation complete\n";
    }

private:
    Rewriter& TheRewriter;
    ASTContext& Context;
    ContainerUsageCollector Collector;
    std::set<const VarDecl*> Containers;
    AllocatorInjector Injector;
};

// Frontend action
class ParallaxTransformAction : public ASTFrontendAction {
public:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(
        CompilerInstance& CI, StringRef File) override {

        errs() << "[ParallaxTransform] Processing file: " << File << "\n";

        TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
        return std::make_unique<ParallaxTransformConsumer>(
            TheRewriter, CI.getASTContext());
    }

    void EndSourceFileAction() override {
        // Output transformed source to stdout
        const RewriteBuffer* RewriteBuf = TheRewriter.getRewriteBufferFor(
            TheRewriter.getSourceMgr().getMainFileID());

        if (RewriteBuf) {
            outs() << std::string(RewriteBuf->begin(), RewriteBuf->end());
        } else {
            errs() << "[ParallaxTransform] No changes made\n";
        }
    }

private:
    Rewriter TheRewriter;
};

int main(int argc, const char** argv) {
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, ParallaxCategory);
    if (!ExpectedParser) {
        errs() << ExpectedParser.takeError();
        return 1;
    }

    CommonOptionsParser& OptionsParser = ExpectedParser.get();
    ClangTool Tool(OptionsParser.getCompilations(),
                   OptionsParser.getSourcePathList());

    return Tool.run(newFrontendActionFactory<ParallaxTransformAction>().get());
}
