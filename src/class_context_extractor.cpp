#include "parallax/class_context_extractor.hpp"
#include <clang/AST/RecursiveASTVisitor.h>
#include <llvm/Support/raw_ostream.h>
#include <set>

namespace parallax {

/**
 * Visitor to find all member function calls within a method
 */
class MemberCallVisitor : public clang::RecursiveASTVisitor<MemberCallVisitor> {
public:
    std::set<clang::CXXMethodDecl*> called_methods;

    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* call) {
        if (clang::CXXMethodDecl* method = call->getMethodDecl()) {
            called_methods.insert(method);
        }
        return true;
    }
};

/**
 * Extract class context from function objects
 */
ClassContext ClassContextExtractor::extract(
    clang::CXXMethodDecl* call_operator,
    clang::ASTContext& context) {

    ClassContext ctx;
    ctx.call_operator = call_operator;
    ctx.record = call_operator->getParent();
    llvm::errs() << "[ClassContextExtractor] Extracting context for: "
                  << ctx.record->getNameAsString() << "\n";

    // Collect member variables
    collectMemberVariables(ctx.record, ctx.member_variables);

    llvm::errs() << "[ClassContextExtractor] Found "
                  << ctx.member_variables.size() << " member variables\n";

    // Collect called member functions
    collectCalledMemberFunctions(call_operator, ctx.member_functions);

    llvm::errs() << "[ClassContextExtractor] Found "
                  << ctx.member_functions.size() << " member functions\n";

    // Collect base classes
    collectBaseClasses(ctx.record, ctx.base_classes);

    llvm::errs() << "[ClassContextExtractor] Found "
                  << ctx.base_classes.size() << " base classes\n";

    return ctx;
}

void ClassContextExtractor::collectMemberVariables(
    clang::CXXRecordDecl* record,
    std::vector<clang::FieldDecl*>& members) {

    if (!record || !record->hasDefinition()) return;

    // First, recursively collect members from base classes
    for (const auto& base : record->bases()) {
        clang::QualType base_type = base.getType();
        if (const auto* base_record = base_type->getAsCXXRecordDecl()) {
            if (base_record->hasDefinition()) {
                llvm::errs() << "[ClassContextExtractor]   Collecting from base class: "
                              << base_record->getNameAsString() << "\n";
                collectMemberVariables(const_cast<clang::CXXRecordDecl*>(base_record), members);
            }
        }
    }

    // Then, get direct members of this class
    for (auto* field : record->fields()) {
        members.push_back(field);
        llvm::errs() << "[ClassContextExtractor]   Member: "
                      << field->getNameAsString() << " : "
                      << field->getType().getAsString() << "\n";
    }
}

void ClassContextExtractor::collectCalledMemberFunctions(
    clang::CXXMethodDecl* method,
    std::vector<clang::CXXMethodDecl*>& functions) {

    if (!method || !method->hasBody()) return;

    MemberCallVisitor visitor;
    visitor.TraverseStmt(method->getBody());
    for (auto* func : visitor.called_methods) {
        functions.push_back(func);
    }
}

void ClassContextExtractor::collectBaseClasses(
    clang::CXXRecordDecl* record,
    std::vector<clang::CXXRecordDecl*>& bases) {

    if (!record || !record->hasDefinition()) return;

    for (const auto& base : record->bases()) {
        clang::QualType base_type = base.getType();
        if (const auto* base_record = base_type->getAsCXXRecordDecl()) {
            bases.push_back(const_cast<clang::CXXRecordDecl*>(base_record));
        }
    }
}

} // namespace parallax
