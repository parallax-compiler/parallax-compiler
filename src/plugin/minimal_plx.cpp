#include <clang/Frontend/FrontendAction.h>
#include <clang/Frontend/FrontendPluginRegistry.h>
#include <clang/AST/ASTConsumer.h>
#include <llvm/Support/raw_ostream.h>

using namespace clang;

namespace {

class MinimalConsumer : public ASTConsumer {
public:
    void HandleTranslationUnit(ASTContext &Context) override {
        llvm::errs() << "Minimal: HandleTranslationUnit\n";
    }
};

class MinimalAction : public PluginASTAction {
protected:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef) override {
        llvm::errs() << "Minimal: CreateASTConsumer\n";
        return std::make_unique<MinimalConsumer>();
    }

    bool ParseArgs(const CompilerInstance &CI, const std::vector<std::string> &args) override {
        llvm::errs() << "Minimal: ParseArgs\n";
        return true;
    }

    ActionType getActionType() override {
        return AddBeforeMainAction;
    }
};

struct GlobalInit {
    GlobalInit() { llvm::errs() << "Minimal: Plugin Loaded\n"; }
} g_init;

} // namespace

static FrontendPluginRegistry::Add<MinimalAction> X("minimal-plx", "Minimal test plugin");
