#include "ParallaxPlugin.h"
#include "parallax/lambda_ir_generator.hpp"
#include "parallax/spirv_generator.hpp"
#include "parallax/class_context_extractor.hpp"
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Lex/Lexer.h>
#include <clang/AST/Type.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/TemplateBase.h>
#include <clang/AST/ExprCXX.h>
#include <llvm/Support/raw_ostream.h>
#include <sstream>
#include <iomanip>
#include <set>

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

    /**
     * Mark a container as needing allocator injection
     */
    void markContainerForAllocation(const clang::VarDecl* var_decl) {
        if (var_decl) {
            containers_needing_allocator_.insert(var_decl);
        }
    }

    /**
     * Apply allocator injections to all marked containers
     */
    void applyAllocatorInjections();

private:
    clang::Rewriter rewriter_;
    clang::CompilerInstance& CI_;
    clang::SourceManager& SM_;
    std::vector<TransformInfo> transforms_;

    // Container tracking for allocator injection
    std::set<const clang::VarDecl*> containers_needing_allocator_;
    std::set<const clang::VarDecl*> rewritten_containers_;
    bool allocator_header_included_ = false;

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

    /**
     * Rewrite a container type to inject parallax::allocator
     */
    void rewriteContainerType(const clang::VarDecl* var_decl);

    /**
     * Check if a container can be rewritten
     */
    bool canRewriteContainer(const clang::VarDecl* var_decl);

    /**
     * Ensure allocator header is included
     */
    void ensureAllocatorHeader();
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

void ParallaxRewriter::applyAllocatorInjections() {
    llvm::errs() << "[ParallaxRewriter] Injecting allocators into "
                 << containers_needing_allocator_.size() << " containers\n";

    if (!containers_needing_allocator_.empty()) {
        ensureAllocatorHeader();
    }

    for (const clang::VarDecl* var_decl : containers_needing_allocator_) {
        if (rewritten_containers_.count(var_decl)) {
            continue;  // Already rewritten
        }

        rewriteContainerType(var_decl);
        rewritten_containers_.insert(var_decl);
    }
}

void ParallaxRewriter::rewriteContainerType(const clang::VarDecl* var_decl) {
    if (!canRewriteContainer(var_decl)) {
        return;
    }

    clang::QualType original_type = var_decl->getType();
    std::string type_str = original_type.getAsString();

    llvm::errs() << "[ParallaxRewriter] Rewriting type: " << type_str << "\n";

    // Get the source range for the type
    clang::TypeSourceInfo* type_src_info = var_decl->getTypeSourceInfo();
    if (!type_src_info) {
        llvm::errs() << "[ParallaxRewriter] Warning: No type source info for "
                     << var_decl->getNameAsString() << "\n";
        return;
    }

    clang::SourceRange type_range = type_src_info->getTypeLoc().getSourceRange();

    // Handle 'auto' types specially
    if (original_type->isUndeducedType()) {
        llvm::errs() << "[ParallaxRewriter] Warning: Cannot rewrite 'auto' types\n";
        clang::DiagnosticsEngine& diag = CI_.getDiagnostics();
        unsigned diag_id = diag.getCustomDiagID(
            clang::DiagnosticsEngine::Warning,
            "Cannot inject allocator into 'auto' type. Please use explicit type "
            "std::vector<T, parallax::allocator<T>>"
        );
        diag.Report(var_decl->getLocation(), diag_id);
        return;
    }

    // Extract element type and container template
    clang::QualType element_type;
    std::string container_template;

    // Get non-reference type for analysis
    clang::QualType base_type = original_type.getNonReferenceType();

    if (const auto* template_spec =
            base_type->getAs<clang::TemplateSpecializationType>()) {

        // Get template name (std::vector, std::array, etc.)
        if (auto* template_decl = template_spec->getTemplateName().getAsTemplateDecl()) {
            container_template = template_decl->getQualifiedNameAsString();
        }

        llvm::ArrayRef<clang::TemplateArgument> args = template_spec->template_arguments();
        if (args.size() > 0) {
            const auto& arg = args[0];
            if (arg.getKind() == clang::TemplateArgument::Type) {
                element_type = arg.getAsType();
            }
        }
    }

    if (element_type.isNull()) {
        llvm::errs() << "[ParallaxRewriter] Warning: Could not extract element type\n";
        return;
    }

    // Build new type string with allocator
    std::string element_type_str = element_type.getAsString();
    std::string new_type;

    if (container_template == "std::vector") {
        new_type = "std::vector<" + element_type_str +
                   ", parallax::allocator<" + element_type_str + ">>";
    } else if (container_template == "std::deque") {
        new_type = "std::deque<" + element_type_str +
                   ", parallax::allocator<" + element_type_str + ">>";
    } else {
        llvm::errs() << "[ParallaxRewriter] Warning: Unsupported container type: "
                     << container_template << "\n";
        return;
    }

    llvm::errs() << "[ParallaxRewriter] New type: " << new_type << "\n";

    // Replace the type
    rewriter_.ReplaceText(type_range, new_type);
}

bool ParallaxRewriter::canRewriteContainer(const clang::VarDecl* var_decl) {
    // Don't rewrite function parameters
    if (llvm::isa<clang::ParmVarDecl>(var_decl)) {
        llvm::errs() << "[ParallaxRewriter] Skipping function parameter\n";
        return false;
    }

    // Don't rewrite global variables (complex initialization issues)
    if (var_decl->hasGlobalStorage() && !var_decl->isStaticLocal()) {
        llvm::errs() << "[ParallaxRewriter] Skipping global variable\n";
        return false;
    }

    return true;
}

void ParallaxRewriter::ensureAllocatorHeader() {
    if (allocator_header_included_) return;

    // Find the first location in the main file
    clang::SourceLocation insert_loc = SM_.getLocForStartOfFile(
        SM_.getMainFileID()
    );

    // Insert the header at the top of the file
    rewriter_.InsertTextBefore(insert_loc,
        "#include <parallax/allocator.hpp>\n");

    llvm::errs() << "[ParallaxRewriter] Injected allocator header\n";

    allocator_header_included_ = true;
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

        // NEW: Trace iterators to containers and mark for allocator injection
        const clang::VarDecl* first_container = nullptr;
        const clang::VarDecl* last_container = nullptr;

        if (info.first_iterator) {
            first_container = traceIteratorToContainer(info.first_iterator);
        }
        if (info.last_iterator) {
            last_container = traceIteratorToContainer(info.last_iterator);
        }

        // Validate that both iterators come from the same container
        if (first_container && last_container && first_container == last_container) {
            llvm::errs() << "[ParallaxCollector] Found container: "
                         << first_container->getNameAsString() << "\n";

            // Check if container already has parallax::allocator
            clang::QualType container_type = first_container->getType();
            if (!hasParallaxAllocator(container_type)) {
                llvm::errs() << "[ParallaxCollector] Marking for allocator injection\n";
                rewriter_.markContainerForAllocation(first_container);
            } else {
                llvm::errs() << "[ParallaxCollector] Already has parallax::allocator\n";
            }
        } else if (first_container || last_container) {
            llvm::errs() << "[ParallaxCollector] Warning: Iterators from different "
                         << "containers or one iterator not traceable\n";
        }

        // For transform, also check output iterator
        if (info.algorithm_name == "transform" && info.output_iterator) {
            const clang::VarDecl* output_container =
                traceIteratorToContainer(info.output_iterator);
            if (output_container) {
                clang::QualType container_type = output_container->getType();
                if (!hasParallaxAllocator(container_type)) {
                    llvm::errs() << "[ParallaxCollector] Marking output container for allocator injection\n";
                    rewriter_.markContainerForAllocation(output_container);
                }
            }
        }

        info.kernel_name = generateKernelName(info);

        // Try to extract lambda first
        if (!info.lambda) {
            llvm::errs() << "[ParallaxCollector V2] No lambda found, trying function object...\n";

            // Try function object extraction
            clang::CXXRecordDecl* functor = extractFunctionObject(info.call_expr);
            if (functor) {
                clang::CXXMethodDecl* op_call = getFunctionCallOperator(functor);
                if (op_call && op_call->hasBody()) {
                    llvm::errs() << "[ParallaxCollector V2] Using function object: "
                                << functor->getNameAsString() << "\n";

                    // NEW V2: Extract full class context
                    ClassContext class_ctx = class_extractor_.extract(op_call, context_);

                    llvm::errs() << "[V2] Class has "
                                << class_ctx.member_variables.size()
                                << " members\n";

                    // Generate IR with CodeGen
                    auto module = ir_generator_.generateIR(op_call, context_);

                    if (!module) {
                        llvm::errs() << "[ParallaxCollector V2] Error: Failed to generate IR from functor\n";
                        return true;
                    }

                    // Find the kernel function in the module
                    llvm::Function* kernel_func = nullptr;
                    for (auto& f : *module) {
                        if (f.getName().startswith("kernel_")) {
                            kernel_func = &f;
                            break;
                        }
                    }

                    if (!kernel_func) {
                        // Fallback to first non-declaration function
                        for (auto& f : *module) {
                            if (!f.isDeclaration()) {
                                kernel_func = &f;
                                break;
                            }
                        }
                    }

                    if (!kernel_func) {
                        llvm::errs() << "[ParallaxCollector V2] Error: No function in module\n";
                        return true;
                    }

                    // Generate SPIR-V
                    SPIRVGenerator spirv_gen;
                    spirv_gen.set_target_vulkan_version(1, 2);

                    // NEW V2: Extract parameter types including captured members
                    std::vector<std::string> param_types;
                    if (info.algorithm_name == "for_each") {
                        param_types.push_back("float&");
                    } else if (info.algorithm_name == "transform") {
                        param_types.push_back("float");
                        param_types.push_back("float&");
                    }

                    info.spirv = spirv_gen.generate_from_lambda(kernel_func, param_types);

                    if (info.spirv.empty()) {
                        llvm::errs() << "[ParallaxCollector V2] Error: SPIR-V generation failed\n";
                        return true;
                    }

                    llvm::errs() << "[ParallaxCollector V2] Generated " << info.spirv.size()
                                << " SPIR-V words\n";

                    // Add to transformations
                    rewriter_.addTransform(info);
                    return true;
                }
            }

            llvm::errs() << "[ParallaxCollector V2] Warning: Could not extract lambda or function object\n";
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
    ClassContextExtractor class_extractor_;

    bool isParallelAlgorithm(clang::CallExpr* call);
    std::string extractAlgorithmName(clang::CallExpr* call);
    clang::LambdaExpr* extractLambda(clang::CallExpr* call);
    void extractIterators(clang::CallExpr* call, clang::Expr*& first, clang::Expr*& last);
    std::string generateKernelName(const TransformInfo& info);

    // NEW: Function object support
    clang::CXXRecordDecl* extractFunctionObject(clang::CallExpr* call);
    clang::CXXMethodDecl* getFunctionCallOperator(clang::CXXRecordDecl* record);

    // NEW: Container tracking methods
    const clang::VarDecl* traceIteratorToContainer(clang::Expr* iterator_expr);
    bool isStandardContainer(clang::QualType type);
    clang::QualType getContainerElementType(clang::QualType container_type);
    bool hasParallaxAllocator(clang::QualType type);
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
            policy_type.find("parallel_unsequenced_policy") != std::string::npos ||
            policy_type.find("par_unseq") != std::string::npos ||
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

clang::CXXRecordDecl* ParallaxCollectorVisitor::extractFunctionObject(clang::CallExpr* call) {
    // Function object is typically the last argument
    if (call->getNumArgs() < 3) return nullptr;

    clang::Expr* last_arg = call->getArg(call->getNumArgs() - 1)->IgnoreImplicit();

    // Get the type of the argument
    clang::QualType arg_type = last_arg->getType();

    // Remove references
    arg_type = arg_type.getNonReferenceType();

    // Get the record type (struct/class)
    if (const auto* record_type = arg_type->getAsCXXRecordDecl()) {
        // Check if it has operator()
        if (getFunctionCallOperator(const_cast<clang::CXXRecordDecl*>(record_type))) {
            llvm::errs() << "[ParallaxCollector] Found function object: "
                        << record_type->getNameAsString() << "\n";
            return const_cast<clang::CXXRecordDecl*>(record_type);
        }
    }

    return nullptr;
}

clang::CXXMethodDecl* ParallaxCollectorVisitor::getFunctionCallOperator(clang::CXXRecordDecl* record) {
    if (!record) return nullptr;

    // Complete the definition if needed
    if (!record->hasDefinition()) {
        return nullptr;
    }

    record = record->getDefinition();

    // Look for operator()
    for (auto* method : record->methods()) {
        if (method->isOverloadedOperator() &&
            method->getOverloadedOperator() == clang::OO_Call) {
            llvm::errs() << "[ParallaxCollector] Found operator() in "
                        << record->getNameAsString() << "\n";
            return method;
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

const clang::VarDecl* ParallaxCollectorVisitor::traceIteratorToContainer(clang::Expr* iterator_expr) {
    if (!iterator_expr) return nullptr;

    // Remove implicit casts and temporary materializations
    clang::Expr* expr = iterator_expr->IgnoreImplicit();

    // Pattern 1: container.begin() or container.end()
    if (auto* member_call = llvm::dyn_cast<clang::CXXMemberCallExpr>(expr)) {
        clang::Expr* object_expr = member_call->getImplicitObjectArgument();
        if (!object_expr) return nullptr;

        // Remove more implicit nodes
        object_expr = object_expr->IgnoreImplicit();

        // Pattern 1a: Direct variable reference (data.begin())
        if (auto* decl_ref = llvm::dyn_cast<clang::DeclRefExpr>(object_expr)) {
            if (auto* var_decl = llvm::dyn_cast<clang::VarDecl>(decl_ref->getDecl())) {
                return var_decl;
            }
        }

        // Pattern 1b: Array subscript (arrays[i].begin())
        if (auto* subscript = llvm::dyn_cast<clang::ArraySubscriptExpr>(object_expr)) {
            // Recurse on the base array
            return traceIteratorToContainer(subscript->getBase());
        }
    }

    // Pattern 2: std::begin(container) or std::end(container)
    if (auto* call_expr = llvm::dyn_cast<clang::CallExpr>(expr)) {
        if (auto* func_decl = call_expr->getDirectCallee()) {
            std::string func_name = func_decl->getQualifiedNameAsString();

            if ((func_name == "std::begin" || func_name == "std::end") &&
                call_expr->getNumArgs() >= 1) {

                clang::Expr* container_arg = call_expr->getArg(0)->IgnoreImplicit();

                if (auto* decl_ref = llvm::dyn_cast<clang::DeclRefExpr>(container_arg)) {
                    if (auto* var_decl = llvm::dyn_cast<clang::VarDecl>(decl_ref->getDecl())) {
                        return var_decl;
                    }
                }
            }
        }
    }

    // Pattern 3: Direct DeclRefExpr (rare but possible)
    if (auto* decl_ref = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
        if (auto* var_decl = llvm::dyn_cast<clang::VarDecl>(decl_ref->getDecl())) {
            return var_decl;
        }
    }

    return nullptr;
}

bool ParallaxCollectorVisitor::isStandardContainer(clang::QualType type) {
    // Remove cv-qualifiers and references
    type = type.getNonReferenceType().getUnqualifiedType();
    std::string type_str = type.getAsString();

    return (type_str.find("std::vector") == 0 ||
            type_str.find("std::array") == 0 ||
            type_str.find("std::deque") == 0 ||
            type_str.find("vector") == 0 ||
            type_str.find("array") == 0 ||
            type_str.find("deque") == 0);
}

clang::QualType ParallaxCollectorVisitor::getContainerElementType(clang::QualType container_type) {
    // Get the template specialization
    if (const auto* template_spec =
            container_type->getAs<clang::TemplateSpecializationType>()) {

        // First template argument is the element type
        llvm::ArrayRef<clang::TemplateArgument> args = template_spec->template_arguments();
        if (args.size() > 0) {
            const auto& arg = args[0];
            if (arg.getKind() == clang::TemplateArgument::Type) {
                return arg.getAsType();
            }
        }
    }

    // For elaborated types, recurse
    if (const auto* elaborated =
            container_type->getAs<clang::ElaboratedType>()) {
        return getContainerElementType(elaborated->getNamedType());
    }

    return clang::QualType();
}

bool ParallaxCollectorVisitor::hasParallaxAllocator(clang::QualType type) {
    std::string type_str = type.getAsString();
    return type_str.find("parallax::allocator") != std::string::npos;
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

        llvm::errs() << "[Parallax] Phase 1.5: Injecting allocators...\n";

        // Phase 1.5: Inject allocators into containers
        rewriter_.applyAllocatorInjections();

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
