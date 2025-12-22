#include "ParallaxPlugin.h"
#include <clang/Frontend/FrontendPluginRegistry.h>
#include <llvm/Support/raw_ostream.h>

// Include the rewriter consumer (defined in ParallaxRewriter.cpp)
namespace parallax {
    class ParallaxASTConsumerV2;
}

// External function to create the V2 consumer
extern std::unique_ptr<clang::ASTConsumer> createParallaxASTConsumerV2(clang::CompilerInstance& CI);

namespace parallax {

// Global initializer to confirm plugin loading
struct GlobalInit {
    GlobalInit() {
        llvm::errs() << "========================================\n";
        llvm::errs() << "Parallax Plugin V2 Loaded\n";
        llvm::errs() << "Automatic GPU offload enabled\n";
        llvm::errs() << "========================================\n";
    }
} g_parallax_init_v2;

/**
 * Plugin action for Parallax V2 with code rewriting
 */
class ParallaxPluginActionV2 : public clang::PluginASTAction {
protected:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
        clang::CompilerInstance &CI,
        llvm::StringRef file) override {

        llvm::errs() << "[Parallax] Processing file: " << file << "\n";

        return createParallaxASTConsumerV2(CI);
    }

    bool ParseArgs(const clang::CompilerInstance &CI,
                   const std::vector<std::string> &args) override {

        for (const auto& arg : args) {
            llvm::errs() << "[Parallax] Argument: " << arg << "\n";

            if (arg == "-help") {
                llvm::errs() << "Parallax Plugin Options:\n";
                llvm::errs() << "  -enable-rewrite : Enable code rewriting (default: on)\n";
                llvm::errs() << "  -disable-rewrite : Disable code rewriting (detection only)\n";
                llvm::errs() << "  -verbose : Enable verbose output\n";
            }
        }

        return true;
    }

    ActionType getActionType() override {
        // Replace action to perform rewriting
        return ReplaceAction;
    }
};

} // namespace parallax

// Register the plugin
static clang::FrontendPluginRegistry::Add<parallax::ParallaxPluginActionV2>
    X("parallax", "Parallax Automatic GPU Offload Plugin");

// Also register under "parallax-plugin" for compatibility
static clang::FrontendPluginRegistry::Add<parallax::ParallaxPluginActionV2>
    Y("parallax-plugin", "Parallax Automatic GPU Offload Plugin");
