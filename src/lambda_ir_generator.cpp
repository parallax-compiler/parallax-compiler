#include "parallax/lambda_ir_generator.hpp"
#include <clang/AST/Type.h>
#include <clang/AST/OperationKinds.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include <iostream>

namespace parallax {

LambdaIRGenerator::LambdaIRGenerator(clang::CompilerInstance& CI)
    : CI_(CI), llvm_context_(std::make_unique<llvm::LLVMContext>()) {}

LambdaIRGenerator::~LambdaIRGenerator() = default;

clang::CXXMethodDecl* LambdaIRGenerator::getLambdaCallOperator(clang::LambdaExpr* lambda) {
    return lambda->getCallOperator();
}

std::vector<LambdaIRGenerator::CaptureInfo>
LambdaIRGenerator::extractCaptures(clang::LambdaExpr* lambda) {
    std::vector<CaptureInfo> captures;

    for (const auto& capture : lambda->captures()) {
        if (capture.capturesVariable()) {
            CaptureInfo info;
            info.name = capture.getCapturedVar()->getNameAsString();
            info.type = capture.getCapturedVar()->getType();
            info.is_by_reference = (capture.getCaptureKind() == clang::LCK_ByRef);
            captures.push_back(info);
        }
    }

    return captures;
}

llvm::Type* LambdaIRGenerator::convertType(clang::QualType clang_type) {
    // Remove qualifiers and get canonical type
    clang::QualType canonical = clang_type.getCanonicalType();

    // Handle references - convert to pointers for LLVM
    if (canonical->isReferenceType()) {
        canonical = canonical->getPointeeType();
        return llvm::PointerType::get(*llvm_context_, 0); // Opaque pointer
    }

    // Handle pointers
    if (canonical->isPointerType()) {
        return llvm::PointerType::get(*llvm_context_, 0);
    }

    // Handle builtin types
    if (const auto* builtin = llvm::dyn_cast<clang::BuiltinType>(canonical)) {
        switch (builtin->getKind()) {
            case clang::BuiltinType::Void:
                return llvm::Type::getVoidTy(*llvm_context_);
            case clang::BuiltinType::Bool:
                return llvm::Type::getInt1Ty(*llvm_context_);
            case clang::BuiltinType::Char_S:
            case clang::BuiltinType::Char_U:
            case clang::BuiltinType::SChar:
            case clang::BuiltinType::UChar:
                return llvm::Type::getInt8Ty(*llvm_context_);
            case clang::BuiltinType::Short:
            case clang::BuiltinType::UShort:
                return llvm::Type::getInt16Ty(*llvm_context_);
            case clang::BuiltinType::Int:
            case clang::BuiltinType::UInt:
                return llvm::Type::getInt32Ty(*llvm_context_);
            case clang::BuiltinType::Long:
            case clang::BuiltinType::ULong:
            case clang::BuiltinType::LongLong:
            case clang::BuiltinType::ULongLong:
                return llvm::Type::getInt64Ty(*llvm_context_);
            case clang::BuiltinType::Float:
                return llvm::Type::getFloatTy(*llvm_context_);
            case clang::BuiltinType::Double:
                return llvm::Type::getDoubleTy(*llvm_context_);
            default:
                llvm::errs() << "Warning: Unknown builtin type, defaulting to i32\n";
                return llvm::Type::getInt32Ty(*llvm_context_);
        }
    }

    // Default fallback
    llvm::errs() << "Warning: Unhandled type, defaulting to i32\n";
    return llvm::Type::getInt32Ty(*llvm_context_);
}

llvm::Value* LambdaIRGenerator::translateExpr(
    clang::Expr* expr,
    llvm::IRBuilder<>& builder,
    clang::ASTContext& context,
    std::map<const clang::VarDecl*, llvm::Value*>& var_map) {

    if (!expr) return nullptr;

    // Handle different expression types
    if (auto* binary_op = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
        llvm::errs() << "[translateExpr] BinaryOperator: " << binary_op->getOpcodeStr() << "\n";
        llvm::errs() << "[translateExpr] LHS type: " << binary_op->getLHS()->getType().getAsString() << "\n";
        llvm::errs() << "[translateExpr] RHS type: " << binary_op->getRHS()->getType().getAsString() << "\n";

        llvm::Value* lhs = translateExpr(binary_op->getLHS(), builder, context, var_map);
        llvm::Value* rhs = translateExpr(binary_op->getRHS(), builder, context, var_map);

        if (!lhs) {
            llvm::errs() << "[translateExpr] LHS is null!\n";
            return nullptr;
        }
        if (!rhs) {
            llvm::errs() << "[translateExpr] RHS is null!\n";
            return nullptr;
        }

        llvm::errs() << "[translateExpr] LHS LLVM type: ";
        lhs->getType()->print(llvm::errs());
        llvm::errs() << "\n";
        llvm::errs() << "[translateExpr] RHS LLVM type: ";
        rhs->getType()->print(llvm::errs());
        llvm::errs() << "\n";

        switch (binary_op->getOpcode()) {
            case clang::BO_Mul:
                // For arithmetic operators, operands must be values, not pointers
                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(builder.getFloatTy(), lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(builder.getFloatTy(), rhs, "rhs_load");
                }
                return builder.CreateFMul(lhs, rhs, "mul");
            case clang::BO_Add:
                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(builder.getFloatTy(), lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(builder.getFloatTy(), rhs, "rhs_load");
                }
                return builder.CreateFAdd(lhs, rhs, "add");
            case clang::BO_Sub:
                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(builder.getFloatTy(), lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(builder.getFloatTy(), rhs, "rhs_load");
                }
                return builder.CreateFSub(lhs, rhs, "sub");
            case clang::BO_Div:
                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(builder.getFloatTy(), lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(builder.getFloatTy(), rhs, "rhs_load");
                }
                return builder.CreateFDiv(lhs, rhs, "div");
            case clang::BO_Assign: {
                llvm::errs() << "[translateExpr] Handling assignment\n";
                // LHS should be an lvalue (pointer/reference)
                // RHS should be a value (not pointer) - load if needed
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(builder.getFloatTy(), rhs, "rhs_load");
                }
                if (lhs->getType()->isPointerTy()) {
                    builder.CreateStore(rhs, lhs);
                    return rhs;
                }
                return nullptr;
            }
            case clang::BO_MulAssign: {
                llvm::errs() << "[translateExpr] Handling *=\n";
                if (lhs->getType()->isPointerTy()) {
                    llvm::errs() << "[translateExpr] LHS is pointer, loading value\n";
                    llvm::Value* loaded = builder.CreateLoad(builder.getFloatTy(), lhs, "tmp");
                    llvm::errs() << "[translateExpr] Creating multiply\n";
                    llvm::Value* result = builder.CreateFMul(loaded, rhs, "mul");
                    llvm::errs() << "[translateExpr] Storing result\n";
                    builder.CreateStore(result, lhs);
                    llvm::errs() << "[translateExpr] MulAssign complete\n";
                    return result;
                } else {
                    llvm::errs() << "[translateExpr] ERROR: LHS is not pointer!\n";
                }
                return nullptr;
            }
            case clang::BO_AddAssign: {
                if (lhs->getType()->isPointerTy()) {
                    llvm::Value* loaded = builder.CreateLoad(builder.getFloatTy(), lhs, "tmp");
                    llvm::Value* result = builder.CreateFAdd(loaded, rhs, "add");
                    builder.CreateStore(result, lhs);
                    return result;
                }
                return nullptr;
            }
            default:
                llvm::errs() << "Warning: Unhandled binary operator: " << binary_op->getOpcodeStr() << "\n";
                return nullptr;
        }
    }

    // Handle unary operators
    if (auto* unary_op = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
        llvm::Value* operand = translateExpr(unary_op->getSubExpr(), builder, context, var_map);

        switch (unary_op->getOpcode()) {
            case clang::UO_Deref:
                // Dereference: load from pointer
                if (operand && operand->getType()->isPointerTy()) {
                    return builder.CreateLoad(builder.getFloatTy(), operand, "deref");
                }
                return operand;
            default:
                return operand;
        }
    }

    // Handle floating point literals
    if (auto* float_literal = llvm::dyn_cast<clang::FloatingLiteral>(expr)) {
        double value = float_literal->getValueAsApproximateDouble();
        return llvm::ConstantFP::get(builder.getFloatTy(), value);
    }

    // Handle integer literals
    if (auto* int_literal = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
        int64_t value = int_literal->getValue().getSExtValue();
        return llvm::ConstantInt::get(builder.getInt32Ty(), value);
    }

    // Handle variable references (parameters, captures)
    if (auto* decl_ref = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
        llvm::errs() << "[translateExpr] DeclRefExpr: " << decl_ref->getDecl()->getNameAsString() << "\n";
        llvm::errs() << "[translateExpr] Type: " << decl_ref->getType().getAsString() << "\n";
        llvm::errs() << "[translateExpr] Is LValue: " << decl_ref->isLValue() << "\n";

        if (auto* var_decl = llvm::dyn_cast<clang::VarDecl>(decl_ref->getDecl())) {
            auto it = var_map.find(var_decl);
            if (it != var_map.end()) {
                llvm::Value* var = it->second;
                llvm::errs() << "[translateExpr] Found variable in map, type: ";
                var->getType()->print(llvm::errs());
                llvm::errs() << "\n";

                // For reference types or lvalues, return the pointer directly
                // Don't auto-load for LHS of assignments
                if (decl_ref->getType()->isReferenceType() || decl_ref->isLValue()) {
                    llvm::errs() << "[translateExpr] Returning pointer for lvalue/reference\n";
                    return var;
                }

                // For rvalue uses, load the value
                if (var->getType()->isPointerTy()) {
                    llvm::errs() << "[translateExpr] Loading rvalue\n";
                    return builder.CreateLoad(builder.getFloatTy(), var, var_decl->getNameAsString());
                }
                return var;
            }
        }

        // Handle function parameters
        if (auto* parm_decl = llvm::dyn_cast<clang::ParmVarDecl>(decl_ref->getDecl())) {
            auto it = var_map.find(parm_decl);
            if (it != var_map.end()) {
                llvm::Value* var = it->second;
                llvm::errs() << "[translateExpr] Found parameter in map\n";
                // Parameters are pointers for reference types
                if (decl_ref->getType()->isReferenceType() || decl_ref->isLValue()) {
                    llvm::errs() << "[translateExpr] Returning parameter pointer\n";
                    return var;
                }
                return var;
            }
        }
    }

    // Handle implicit casts
    if (auto* implicit_cast = llvm::dyn_cast<clang::ImplicitCastExpr>(expr)) {
        return translateExpr(implicit_cast->getSubExpr(), builder, context, var_map);
    }

    // Handle parenthesized expressions
    if (auto* paren = llvm::dyn_cast<clang::ParenExpr>(expr)) {
        return translateExpr(paren->getSubExpr(), builder, context, var_map);
    }

    llvm::errs() << "Warning: Unhandled expression type: ";
    expr->dump();
    return nullptr;
}

void LambdaIRGenerator::translateStmt(
    clang::Stmt* stmt,
    llvm::IRBuilder<>& builder,
    clang::ASTContext& context,
    std::map<const clang::VarDecl*, llvm::Value*>& var_map) {

    if (!stmt) return;

    llvm::errs() << "[translateStmt] Statement type: " << stmt->getStmtClassName() << "\n";

    // Handle compound statements (blocks)
    if (auto* compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
        llvm::errs() << "[translateStmt] CompoundStmt with " << compound->size() << " children\n";
        for (auto* child : compound->body()) {
            translateStmt(child, builder, context, var_map);
        }
        return;
    }

    // Handle ExprWithCleanups (wrapper for expressions with destructors)
    if (auto* expr_cleanups = llvm::dyn_cast<clang::ExprWithCleanups>(stmt)) {
        llvm::errs() << "[translateStmt] Unwrapping ExprWithCleanups\n";
        translateStmt(expr_cleanups->getSubExpr(), builder, context, var_map);
        return;
    }

    // Handle expression statements
    if (auto* expr_stmt = llvm::dyn_cast<clang::Expr>(stmt)) {
        llvm::errs() << "[translateStmt] Expression: ";
        expr_stmt->dump();
        translateExpr(expr_stmt, builder, context, var_map);
        return;
    }

    // Handle return statements
    if (auto* ret_stmt = llvm::dyn_cast<clang::ReturnStmt>(stmt)) {
        if (ret_stmt->getRetValue()) {
            llvm::Value* ret_val = translateExpr(ret_stmt->getRetValue(), builder, context, var_map);
            if (ret_val) {
                builder.CreateRet(ret_val);
                return;
            }
        }
        builder.CreateRetVoid();
        return;
    }

    llvm::errs() << "Warning: Unhandled statement type: " << stmt->getStmtClassName() << "\n";
    stmt->dump();
}

std::unique_ptr<llvm::Module> LambdaIRGenerator::generateIRManual(
    clang::LambdaExpr* lambda,
    clang::ASTContext& context) {

    std::string module_name = "lambda_" + std::to_string(reinterpret_cast<uintptr_t>(lambda));
    auto module = std::make_unique<llvm::Module>(module_name, *llvm_context_);

    // Get lambda call operator
    clang::CXXMethodDecl* call_op = getLambdaCallOperator(lambda);

    // Build function signature
    std::vector<llvm::Type*> param_types;
    std::map<const clang::VarDecl*, llvm::Value*> var_map;

    for (auto* param : call_op->parameters()) {
        llvm::Type* param_type = convertType(param->getType());
        param_types.push_back(param_type);
    }

    llvm::Type* return_type = convertType(call_op->getReturnType());
    llvm::FunctionType* func_type = llvm::FunctionType::get(return_type, param_types, false);

    // Create function
    llvm::Function* func = llvm::Function::Create(
        func_type,
        llvm::Function::ExternalLinkage,
        "lambda_kernel",
        module.get()
    );

    // Map parameters to LLVM arguments
    auto arg_it = func->arg_begin();
    for (auto* param : call_op->parameters()) {
        llvm::Argument* arg = &(*arg_it++);
        arg->setName(param->getNameAsString());
        var_map[param] = arg;
    }

    // Create entry basic block
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(*llvm_context_, "entry", func);
    llvm::IRBuilder<> builder(entry);

    // Translate lambda body
    if (clang::Stmt* body = call_op->getBody()) {
        translateStmt(body, builder, context, var_map);
    }

    // Add return if not already present
    if (!entry->getTerminator()) {
        if (return_type->isVoidTy()) {
            builder.CreateRetVoid();
        }
    }

    // Verify function
    if (llvm::verifyFunction(*func, &llvm::errs())) {
        llvm::errs() << "ERROR: Generated LLVM IR is invalid!\n";
        func->print(llvm::errs());
        return nullptr;
    }

    std::cerr << "\n[LambdaIRGenerator] Generated LLVM IR:\n";
    func->print(llvm::errs());
    std::cerr << "\n";

    return module;
}

std::unique_ptr<llvm::Module> LambdaIRGenerator::generateIR(
    clang::LambdaExpr* lambda,
    clang::ASTContext& context) {

    // For now, use manual generation
    // TODO: Implement using Clang's CodeGen for more complex cases
    return generateIRManual(lambda, context);
}

} // namespace parallax
