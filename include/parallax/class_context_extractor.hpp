#ifndef PARALLAX_CLASS_CONTEXT_EXTRACTOR_HPP
#define PARALLAX_CLASS_CONTEXT_EXTRACTOR_HPP

#include <clang/AST/Decl.h>
#include <clang/AST/DeclCXX.h>
#include <vector>

namespace parallax {

/**
 * Complete context of a function object class
 *
 * Captures all information needed to generate GPU kernel from a functor:
 * - The class definition
 * - The operator() method
 * - All member variables (including from base classes)
 * - All member functions that might be called
 * - Base class hierarchy
 */
struct ClassContext {
    clang::CXXRecordDecl* record;           ///< The class/struct definition
    clang::CXXMethodDecl* call_operator;     ///< The operator() method
    std::vector<clang::FieldDecl*> member_variables;  ///< All member variables
    std::vector<clang::CXXMethodDecl*> member_functions;  ///< Called member functions
    std::vector<clang::CXXRecordDecl*> base_classes;  ///< Base class hierarchy
};

/**
 * Extracts complete class context for function objects
 *
 * This class analyzes a function object's operator() method and extracts:
 * 1. The containing class definition
 * 2. All member variables (including inherited)
 * 3. All member functions called by operator()
 * 4. The base class hierarchy
 */
class ClassContextExtractor {
public:
    /**
     * Extract complete class context for a function object
     * @param call_operator The operator() method
     * @param context AST context
     * @return Full class context including members and bases
     */
    ClassContext extract(
        clang::CXXMethodDecl* call_operator,
        clang::ASTContext& context
    );

private:
    // Collect all member variables (including from base classes)
    void collectMemberVariables(
        clang::CXXRecordDecl* record,
        std::vector<clang::FieldDecl*>& members
    );

    // Collect all member functions that might be called
    void collectCalledMemberFunctions(
        clang::CXXMethodDecl* method,
        std::vector<clang::CXXMethodDecl*>& functions
    );

    // Collect base classes
    void collectBaseClasses(
        clang::CXXRecordDecl* record,
        std::vector<clang::CXXRecordDecl*>& bases
    );
};

} // namespace parallax

#endif // PARALLAX_CLASS_CONTEXT_EXTRACTOR_HPP
