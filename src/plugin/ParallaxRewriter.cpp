#include "ParallaxPlugin.h"
#include "parallax/lambda_ir_generator.hpp"
#include "parallax/spirv_generator.hpp"
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Lex/Lexer.h>
#include <llvm/Support/raw_ostream.h>
#include <sstream>
#include <iomanip>

namespace parallax {

/**
 * Transformation information for a single parallel algorithm call
 */
struct TransformInfo {
    clang::CallExpr* call_expr;
    clang::LambdaExpr* lambda;
    clang::Expr* first_iterator;
    clang::Expr* last_iterator;
    clang::Expr* output_iterator;  // For transform
    std::string algorithm_name;
    clang::QualType element_type;

    // Generated artifacts
    std::string kernel_name;
    std::vector<uint32_t> spirv;
};

/**
 * AST Rewriter for Parallax transformations
 */
class ParallaxRewriter {
public:
    ParallaxRewriter(clang::SourceManager& SM,
                     clang::LangOptions& LO,
                     clang::CompilerInstance& CI)
        : rewriter_(SM, LO), CI_(CI), SM_(SM) {}

    /**
     * Add a transformation to be applied
     */
    void addTransform(TransformInfo& info) {
        transforms_.push_back(info);
    }

    /**
     * Apply all transformations
     */
    void applyAllTransformations() {
        for (auto& transform : transforms_) {
            applyTransformation(transform);
        }
    }

    /**
     * Write rewritten files
     */
    bool writeRewrittenFiles() {
        return !rewriter_.overwriteChangedFiles();
    }

private:
    clang::Rewriter rewriter_;
    clang::CompilerInstance& CI_;
    clang::SourceManager& SM_;
    std::vector<TransformInfo> transforms_;

    /**
     * Apply a single transformation
     */
    void applyTransformation(TransformInfo& transform);

    /**
     * Generate replacement code for a transformed call
     */
    std::string generateReplacementCode(TransformInfo& transform);

    /**
     * Get source text for an AST node
     */
    std::string getSourceText(clang::SourceRange range);

    /**
     * Generate SPIR-V data array string
     */
    std::string generateSPIRVArray(const std::string& name,
                                   const std::vector<uint32_t>& spirv);
};

void ParallaxRewriter::applyTransformation(TransformInfo& transform) {
    llvm::errs() << "[ParallaxRewriter] Transforming call at "
                 << transform.call_expr->getBeginLoc().printToString(SM_) << "\n";

    // Generate replacement code
    std::string replacement = generateReplacementCode(transform);

    // Replace the original call
    clang::SourceRange call_range = transform.call_expr->getSourceRange();

    // Find the semicolon after the call if it's an expression statement
    clang::SourceLocation end_loc = call_range.getEnd();
    end_loc = clang::Lexer::getLocForEndOfToken(end_loc, 0, SM_, CI_.getLangOpts());

    clang::SourceRange full_range(call_range.getBegin(), end_loc);

    rewriter_.ReplaceText(full_range, replacement);

    llvm::errs() << "[ParallaxRewriter] Replacement code:\n" << replacement << "\n";
}

std::string ParallaxRewriter::generateReplacementCode(TransformInfo& transform) {
    std::ostringstream ss;

    ss << "{\n";
    ss << "  /* Parallax GPU offload for " << transform.algorithm_name << " */\n";
    ss << "  /* Runtime API: parallax/runtime.h */\n";

    // 1. SPIR-V data array
    ss << generateSPIRVArray(transform.kernel_name, transform.spirv);

    // 2. Kernel handle (lazy initialization)
    ss << "  static parallax_kernel_t " << transform.kernel_name << " = nullptr;\n";
    ss << "  if (!" << transform.kernel_name << ") {\n";
    ss << "    " << transform.kernel_name << " = parallax_kernel_load(\n";
    ss << "      " << transform.kernel_name << "_spirv,\n";
    ss << "      sizeof(" << transform.kernel_name << "_spirv) / sizeof(uint32_t)\n";
    ss << "    );\n";
    ss << "  }\n\n";

    // 3. Extract iterator range
    std::string first_it = getSourceText(transform.first_iterator->getSourceRange());
    std::string last_it = getSourceText(transform.last_iterator->getSourceRange());

    ss << "  size_t __plx_count = std::distance(" << first_it << ", " << last_it << ");\n";

    // 4. Launch kernel (different for transform vs for_each)
    if (transform.algorithm_name == "transform" && transform.output_iterator) {
        // Transform: two buffers (input and output)
        std::string output_it = getSourceText(transform.output_iterator->getSourceRange());
        ss << "  auto __plx_in_ptr = &(*" << first_it << ");\n";
        ss << "  auto __plx_out_ptr = &(*" << output_it << ");\n\n";
        ss << "  parallax_kernel_launch_transform(" << transform.kernel_name
           << ", __plx_in_ptr, __plx_out_ptr, __plx_count);\n";
    } else {
        // For_each: single buffer (in-place)
        ss << "  auto __plx_ptr = &(*" << first_it << ");\n\n";
        ss << "  parallax_kernel_launch(" << transform.kernel_name
           << ", __plx_ptr, __plx_count);\n";
    }

    ss << "}";

    return ss.str();
}

std::string ParallaxRewriter::generateSPIRVArray(
    const std::string& name,
    const std::vector<uint32_t>& spirv) {

    std::ostringstream ss;
    ss << "  static const uint32_t " << name << "_spirv[] = {\n";
    ss << "    ";

    for (size_t i = 0; i < spirv.size(); i++) {
        ss << "0x" << std::hex << std::setw(8) << std::setfill('0') << spirv[i];
        if (i + 1 < spirv.size()) {
            ss << ", ";
        }
        if ((i + 1) % 8 == 0 && i + 1 < spirv.size()) {
            ss << "\n    ";
        }
    }

    ss << "\n  };\n\n";
    return ss.str();
}

std::string ParallaxRewriter::getSourceText(clang::SourceRange range) {
    clang::SourceLocation start = range.getBegin();
    clang::SourceLocation end = range.getEnd();

    end = clang::Lexer::getLocForEndOfToken(end, 0, SM_, CI_.getLangOpts());

    if (start.isInvalid() || end.isInvalid()) {
        return "";
    }

    const char* start_ptr = SM_.getCharacterData(start);
    const char* end_ptr = SM_.getCharacterData(end);

    return std::string(start_ptr, end_ptr - start_ptr);
}

/**
 * Collector visitor - Phase 1: Collect transformations
 */
class ParallaxCollectorVisitor : public clang::RecursiveASTVisitor<ParallaxCollectorVisitor> {
public:
    ParallaxCollectorVisitor(clang::ASTContext& context,
                             clang::CompilerInstance& CI,
                             ParallaxRewriter& rewriter)
        : context_(context), CI_(CI), rewriter_(rewriter),
          ir_generator_(CI) {}

    bool VisitCallExpr(clang::CallExpr* call) {
        if (!isParallelAlgorithm(call)) {
            return true;
        }

        llvm::errs() << "[ParallaxCollector] Found parallel algorithm call\n";

        // Extract transformation info
        TransformInfo info;
        info.call_expr = call;
        info.algorithm_name = extractAlgorithmName(call);
        info.lambda = extractLambda(call);
        extractIterators(call, info.first_iterator, info.last_iterator);

        // For transform, extract output iterator
        if (info.algorithm_name == "transform" && call->getNumArgs() >= 5) {
            info.output_iterator = call->getArg(3);  // d_first
        } else {
            info.output_iterator = nullptr;
        }

        info.kernel_name = generateKernelName(info);

        if (!info.lambda) {
            llvm::errs() << "[ParallaxCollector] Warning: Could not extract lambda\n";
            return true;
        }

        // Generate LLVM IR for lambda
        auto module = ir_generator_.generateIR(info.lambda, context_);

        if (!module) {
            llvm::errs() << "[ParallaxCollector] Error: Failed to generate IR\n";
            return true;
        }

        // Find the lambda function in the module
        llvm::Function* lambda_func = nullptr;
        for (auto& func : *module) {
            if (!func.isDeclaration()) {
                lambda_func = &func;
                break;
            }
        }

        if (!lambda_func) {
            llvm::errs() << "[ParallaxCollector] Error: No function in module\n";
            return true;
        }

        // Generate SPIR-V
        SPIRVGenerator spirv_gen;
        spirv_gen.set_target_vulkan_version(1, 2);

        // Determine parameter types based on algorithm
        std::vector<std::string> param_types;
        if (info.algorithm_name == "for_each") {
            param_types.push_back("float&");
        } else if (info.algorithm_name == "transform") {
            param_types.push_back("float");
            param_types.push_back("float&");
        }

        info.spirv = spirv_gen.generate_from_lambda(lambda_func, param_types);

        if (info.spirv.empty()) {
            llvm::errs() << "[ParallaxCollector] Error: SPIR-V generation failed\n";
            return true;
        }

        llvm::errs() << "[ParallaxCollector] Generated " << info.spirv.size()
                     << " SPIR-V words\n";

        // Add to transformations
        rewriter_.addTransform(info);

        return true;
    }

private:
    clang::ASTContext& context_;
    clang::CompilerInstance& CI_;
    ParallaxRewriter& rewriter_;
    LambdaIRGenerator ir_generator_;

    bool isParallelAlgorithm(clang::CallExpr* call);
    std::string extractAlgorithmName(clang::CallExpr* call);
    clang::LambdaExpr* extractLambda(clang::CallExpr* call);
    void extractIterators(clang::CallExpr* call, clang::Expr*& first, clang::Expr*& last);
    std::string generateKernelName(const TransformInfo& info);
};

bool ParallaxCollectorVisitor::isParallelAlgorithm(clang::CallExpr* call) {
    if (!call) return false;

    clang::FunctionDecl* func = call->getDirectCallee();
    if (!func) return false;

    std::string name = func->getQualifiedNameAsString();

    // Check for parallel algorithms
    if (name != "std::for_each" && name != "std::transform" && name != "std::reduce") {
        return false;
    }

    // Check if first argument is std::execution::par
    if (call->getNumArgs() < 3) return false;

    clang::Expr* policy_arg = call->getArg(0)->IgnoreImplicit();
    std::string policy_type = policy_arg->getType().getAsString();

    return (policy_type.find("parallel_policy") != std::string::npos ||
            policy_type.find("par") != std::string::npos);
}

std::string ParallaxCollectorVisitor::extractAlgorithmName(clang::CallExpr* call) {
    if (!call || !call->getDirectCallee()) return "";

    std::string full_name = call->getDirectCallee()->getQualifiedNameAsString();

    // Extract just the algorithm name
    size_t pos = full_name.rfind("::");
    if (pos != std::string::npos) {
        return full_name.substr(pos + 2);
    }

    return full_name;
}

clang::LambdaExpr* ParallaxCollectorVisitor::extractLambda(clang::CallExpr* call) {
    // Lambda is typically the last argument
    if (call->getNumArgs() < 3) return nullptr;

    clang::Expr* last_arg = call->getArg(call->getNumArgs() - 1)->IgnoreImplicit();

    // Check if it's a lambda expression directly
    if (auto* lambda = llvm::dyn_cast<clang::LambdaExpr>(last_arg)) {
        return lambda;
    }

    // Check if it's wrapped in casts
    if (auto* mat_temp = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(last_arg)) {
        if (auto* lambda = llvm::dyn_cast<clang::LambdaExpr>(
                mat_temp->getSubExpr()->IgnoreImplicit())) {
            return lambda;
        }
    }

    return nullptr;
}

void ParallaxCollectorVisitor::extractIterators(
    clang::CallExpr* call,
    clang::Expr*& first,
    clang::Expr*& last) {

    // For std::for_each(policy, first, last, lambda)
    // first is arg 1, last is arg 2
    if (call->getNumArgs() >= 4) {
        first = call->getArg(1);
        last = call->getArg(2);
    } else if (call->getNumArgs() == 3) {
        // Some overloads might not have policy
        first = call->getArg(0);
        last = call->getArg(1);
    }
}

std::string ParallaxCollectorVisitor::generateKernelName(const TransformInfo& info) {
    // Generate unique kernel name based on source location
    std::string loc = info.call_expr->getBeginLoc().printToString(
        context_.getSourceManager()
    );

    // Extract line number
    size_t line_start = loc.rfind(':');
    std::string line_num = "0";
    if (line_start != std::string::npos) {
        line_num = loc.substr(line_start + 1);
        // Remove column number
        size_t col_start = line_num.find(':');
        if (col_start != std::string::npos) {
            line_num = line_num.substr(0, col_start);
        }
    }

    return "__parallax_kernel_" + info.algorithm_name + "_" + line_num;
}

/**
 * Updated AST Consumer
 */
class ParallaxASTConsumerV2 : public clang::ASTConsumer {
public:
    explicit ParallaxASTConsumerV2(clang::CompilerInstance& CI)
        : CI_(CI),
          rewriter_(CI.getSourceManager(), CI.getLangOpts(), CI) {}

    void HandleTranslationUnit(clang::ASTContext& context) override {
        llvm::errs() << "[Parallax] Phase 1: Collecting transformations...\n";

        // Phase 1: Collect transformations
        ParallaxCollectorVisitor collector(context, CI_, rewriter_);
        collector.TraverseDecl(context.getTranslationUnitDecl());

        llvm::errs() << "[Parallax] Phase 2: Applying transformations...\n";

        // Phase 2: Apply transformations
        rewriter_.applyAllTransformations();

        llvm::errs() << "[Parallax] Phase 3: Writing rewritten files...\n";

        // Phase 3: Output rewritten source
        if (!rewriter_.writeRewrittenFiles()) {
            llvm::errs() << "[Parallax] Successfully rewrote files\n";
        } else {
            llvm::errs() << "[Parallax] Failed to rewrite files\n";
        }
    }

private:
    clang::CompilerInstance& CI_;
    ParallaxRewriter rewriter_;
};

} // namespace parallax

// Factory function to create the V2 consumer (accessible from ParallaxPluginV2.cpp)
std::unique_ptr<clang::ASTConsumer> createParallaxASTConsumerV2(clang::CompilerInstance& CI) {
    return std::make_unique<parallax::ParallaxASTConsumerV2>(CI);
}
