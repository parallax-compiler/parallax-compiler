#include "parallax/class_context_extractor.hpp"
#include <clang/AST/RecursiveASTVisitor.h>
#include <llvm/Support/raw_ostream.h>

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

    record = record->getDefinition();

    // Get direct members
    for (auto* field : record->fields()) {
        members.push_back(field);
        llvm::errs() << "[ClassContextExtractor]   Member: "
                     << field->getNameAsString() << " : "
                     << field->getType().getAsString() << "\n";
    }

    // Recursively collect from base classes
    for (const auto& base : record->bases()) {
        if (clang::CXXRecordDecl* base_record =
            base.getType()->getAsCXXRecordDecl()) {
            collectMemberVariables(base_record, members);
        }
    }
}

void ClassContextExtractor::collectCalledMemberFunctions(
    clang::CXXMethodDecl* method,
    std::vector<clang::CXXMethodDecl*>& functions) {

    if (!method || !method->hasBody()) return;

    MemberCallVisitor visitor;
    visitor.TraverseStmt(method->getBody());

    functions.insert(functions.end(),
                    visitor.called_methods.begin(),
                    visitor.called_methods.end());

    // Recursively collect from called functions
    for (clang::CXXMethodDecl* called : visitor.called_methods) {
        std::vector<clang::CXXMethodDecl*> transitive;
        collectCalledMemberFunctions(called, transitive);
        functions.insert(functions.end(), transitive.begin(), transitive.end());
    }
}

void ClassContextExtractor::collectBaseClasses(
    clang::CXXRecordDecl* record,
    std::vector<clang::CXXRecordDecl*>& bases) {

    if (!record || !record->hasDefinition()) return;

    for (const auto& base : record->bases()) {
        if (clang::CXXRecordDecl* base_record =
            base.getType()->getAsCXXRecordDecl()) {
            bases.push_back(base_record);
            llvm::errs() << "[ClassContextExtractor]   Base: "
                         << base_record->getNameAsString() << "\n";
            collectBaseClasses(base_record, bases);
        }
    }
}

} // namespace parallax
