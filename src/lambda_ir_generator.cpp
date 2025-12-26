#include "parallax/lambda_ir_generator.hpp"
#include "parallax/class_context_extractor.hpp"
#include "parallax/kernel_wrapper.hpp"
#include <clang/AST/Type.h>
#include <clang/AST/OperationKinds.h>
#include <clang/CodeGen/ModuleBuilder.h>
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

            // Comparison operators
            case clang::BO_LT:
                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(builder.getInt32Ty(), lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(builder.getInt32Ty(), rhs, "rhs_load");
                }
                if (lhs->getType()->isFloatingPointTy()) {
                    return builder.CreateFCmpOLT(lhs, rhs, "cmp_lt");
                } else {
                    return builder.CreateICmpSLT(lhs, rhs, "cmp_lt");
                }
            case clang::BO_LE:
                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(builder.getInt32Ty(), lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(builder.getInt32Ty(), rhs, "rhs_load");
                }
                if (lhs->getType()->isFloatingPointTy()) {
                    return builder.CreateFCmpOLE(lhs, rhs, "cmp_le");
                } else {
                    return builder.CreateICmpSLE(lhs, rhs, "cmp_le");
                }
            case clang::BO_GT:
                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(builder.getInt32Ty(), lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(builder.getInt32Ty(), rhs, "rhs_load");
                }
                if (lhs->getType()->isFloatingPointTy()) {
                    return builder.CreateFCmpOGT(lhs, rhs, "cmp_gt");
                } else {
                    return builder.CreateICmpSGT(lhs, rhs, "cmp_gt");
                }
            case clang::BO_GE:
                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(builder.getInt32Ty(), lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(builder.getInt32Ty(), rhs, "rhs_load");
                }
                if (lhs->getType()->isFloatingPointTy()) {
                    return builder.CreateFCmpOGE(lhs, rhs, "cmp_ge");
                } else {
                    return builder.CreateICmpSGE(lhs, rhs, "cmp_ge");
                }
            case clang::BO_EQ:
                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(builder.getInt32Ty(), lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(builder.getInt32Ty(), rhs, "rhs_load");
                }
                if (lhs->getType()->isFloatingPointTy()) {
                    return builder.CreateFCmpOEQ(lhs, rhs, "cmp_eq");
                } else {
                    return builder.CreateICmpEQ(lhs, rhs, "cmp_eq");
                }
            case clang::BO_NE:
                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(builder.getInt32Ty(), lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(builder.getInt32Ty(), rhs, "rhs_load");
                }
                if (lhs->getType()->isFloatingPointTy()) {
                    return builder.CreateFCmpONE(lhs, rhs, "cmp_ne");
                } else {
                    return builder.CreateICmpNE(lhs, rhs, "cmp_ne");
                }

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
            case clang::BO_SubAssign: {
                if (lhs->getType()->isPointerTy()) {
                    llvm::Value* loaded = builder.CreateLoad(builder.getFloatTy(), lhs, "tmp");
                    llvm::Value* result = builder.CreateFSub(loaded, rhs, "sub");
                    builder.CreateStore(result, lhs);
                    return result;
                }
                return nullptr;
            }
            case clang::BO_DivAssign: {
                if (lhs->getType()->isPointerTy()) {
                    llvm::Value* loaded = builder.CreateLoad(builder.getFloatTy(), lhs, "tmp");
                    llvm::Value* result = builder.CreateFDiv(loaded, rhs, "div");
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
        if (!operand) return nullptr;

        switch (unary_op->getOpcode()) {
            case clang::UO_Deref:
                // Dereference: load from pointer
                if (operand && operand->getType()->isPointerTy()) {
                    return builder.CreateLoad(builder.getFloatTy(), operand, "deref");
                }
                return operand;

            case clang::UO_Minus:
                if (operand->getType()->isPointerTy()) {
                    operand = builder.CreateLoad(builder.getFloatTy(), operand, "load");
                }
                if (operand->getType()->isFloatingPointTy()) {
                    return builder.CreateFNeg(operand, "neg");
                } else {
                    return builder.CreateNeg(operand, "neg");
                }

            case clang::UO_PreInc: {
                // ++i: load, add 1, store, return new value
                llvm::Type* elem_type = operand->getType()->isPointerTy() ?
                    builder.getInt32Ty() : operand->getType();
                llvm::Value* val = operand->getType()->isPointerTy() ?
                    builder.CreateLoad(elem_type, operand, "preinc_load") : operand;
                llvm::Value* inc = elem_type->isFloatingPointTy() ?
                    builder.CreateFAdd(val, llvm::ConstantFP::get(elem_type, 1.0), "preinc") :
                    builder.CreateAdd(val, llvm::ConstantInt::get(elem_type, 1), "preinc");
                if (operand->getType()->isPointerTy()) {
                    builder.CreateStore(inc, operand);
                }
                return inc;
            }

            case clang::UO_PostInc: {
                // i++: load, save old, add 1, store, return old value
                llvm::Type* elem_type = operand->getType()->isPointerTy() ?
                    builder.getInt32Ty() : operand->getType();
                llvm::Value* old_val = operand->getType()->isPointerTy() ?
                    builder.CreateLoad(elem_type, operand, "postinc_load") : operand;
                llvm::Value* inc = elem_type->isFloatingPointTy() ?
                    builder.CreateFAdd(old_val, llvm::ConstantFP::get(elem_type, 1.0), "postinc") :
                    builder.CreateAdd(old_val, llvm::ConstantInt::get(elem_type, 1), "postinc");
                if (operand->getType()->isPointerTy()) {
                    builder.CreateStore(inc, operand);
                }
                return old_val;  // Return the old value
            }

            case clang::UO_PreDec: {
                // --i: load, sub 1, store, return new value
                llvm::Type* elem_type = operand->getType()->isPointerTy() ?
                    builder.getInt32Ty() : operand->getType();
                llvm::Value* val = operand->getType()->isPointerTy() ?
                    builder.CreateLoad(elem_type, operand, "predec_load") : operand;
                llvm::Value* dec = elem_type->isFloatingPointTy() ?
                    builder.CreateFSub(val, llvm::ConstantFP::get(elem_type, 1.0), "predec") :
                    builder.CreateSub(val, llvm::ConstantInt::get(elem_type, 1), "predec");
                if (operand->getType()->isPointerTy()) {
                    builder.CreateStore(dec, operand);
                }
                return dec;
            }

            case clang::UO_PostDec: {
                // i--: load, save old, sub 1, store, return old value
                llvm::Type* elem_type = operand->getType()->isPointerTy() ?
                    builder.getInt32Ty() : operand->getType();
                llvm::Value* old_val = operand->getType()->isPointerTy() ?
                    builder.CreateLoad(elem_type, operand, "postdec_load") : operand;
                llvm::Value* dec = elem_type->isFloatingPointTy() ?
                    builder.CreateFSub(old_val, llvm::ConstantFP::get(elem_type, 1.0), "postdec") :
                    builder.CreateSub(old_val, llvm::ConstantInt::get(elem_type, 1), "postdec");
                if (operand->getType()->isPointerTy()) {
                    builder.CreateStore(dec, operand);
                }
                return old_val;  // Return the old value
            }

            default:
                llvm::errs() << "Warning: Unhandled unary operator: " << unary_op->getOpcodeStr(unary_op->getOpcode()) << "\n";
                return operand;
        }
    }

    // Handle array subscript expressions
    if (auto* array_sub = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
        llvm::errs() << "[translateExpr] ArraySubscriptExpr\n";

        llvm::Value* base = translateExpr(array_sub->getBase(), builder, context, var_map);
        llvm::Value* idx = translateExpr(array_sub->getIdx(), builder, context, var_map);

        if (!base || !idx) {
            llvm::errs() << "[translateExpr] Failed to translate array subscript base or index\n";
            return nullptr;
        }

        // If index is a pointer, load it
        if (idx->getType()->isPointerTy()) {
            idx = builder.CreateLoad(builder.getInt32Ty(), idx, "idx_load");
        }

        llvm::errs() << "[translateExpr] Array base type: ";
        base->getType()->print(llvm::errs());
        llvm::errs() << "\n";

        // GEP for array indexing
        if (base->getType()->isPointerTy()) {
            // For opaque pointers in LLVM 21, we need to track the pointee type separately
            // Try to get it from the var_map by looking up the base declaration

            // Simple approach: assume float type for now
            // In a more complete implementation, we'd need to track the actual element type
            llvm::Type* element_type = builder.getFloatTy();

            // Check if base is from a DeclRefExpr to get actual type
            auto* base_expr = array_sub->getBase()->IgnoreImplicit();
            if (auto* decl_ref = llvm::dyn_cast<clang::DeclRefExpr>(base_expr)) {
                if (auto* var_decl = llvm::dyn_cast<clang::VarDecl>(decl_ref->getDecl())) {
                    clang::QualType qt = var_decl->getType();
                    if (qt->isConstantArrayType()) {
                        const clang::ConstantArrayType* arr_type = context.getAsConstantArrayType(qt);
                        element_type = convertType(arr_type->getElementType());
                    } else if (qt->isPointerType()) {
                        element_type = convertType(qt->getPointeeType());
                    }
                }
            }

            // For arrays: GEP with two indices [0, idx]
            // For pointers: GEP with one index [idx]
            clang::QualType base_type = array_sub->getBase()->IgnoreImplicit()->getType();
            if (base_type->isConstantArrayType()) {
                const clang::ConstantArrayType* arr_type = context.getAsConstantArrayType(base_type);
                uint64_t arr_size = arr_type->getSize().getZExtValue();
                llvm::ArrayType* llvm_arr_type = llvm::ArrayType::get(element_type, arr_size);

                llvm::Value* indices[] = {
                    llvm::ConstantInt::get(builder.getInt32Ty(), 0),
                    idx
                };
                llvm::Value* elem_ptr = builder.CreateGEP(llvm_arr_type, base, indices, "arrayidx");
                llvm::errs() << "[translateExpr] Created GEP for array element\n";
                return elem_ptr;
            } else {
                // Pointer indexing
                llvm::Value* elem_ptr = builder.CreateGEP(element_type, base, idx, "arrayidx");
                llvm::errs() << "[translateExpr] Created GEP for pointer element\n";
                return elem_ptr;
            }
        }

        llvm::errs() << "[translateExpr] Warning: Unhandled array subscript pattern\n";
        return nullptr;
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

    // Handle variable declarations (DeclStmt)
    if (auto* decl_stmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
        llvm::errs() << "[translateStmt] DeclStmt with " << (*decl_stmt->decl_begin())->getDeclKindName() << "\n";

        for (auto* decl : decl_stmt->decls()) {
            if (auto* var_decl = llvm::dyn_cast<clang::VarDecl>(decl)) {
                llvm::errs() << "[translateStmt] Variable declaration: " << var_decl->getNameAsString() << "\n";
                llvm::errs() << "[translateStmt] Type: " << var_decl->getType().getAsString() << "\n";

                clang::QualType var_type = var_decl->getType();

                // Get the entry block for alloca insertion (SPIR-V requirement)
                llvm::Function* func = builder.GetInsertBlock()->getParent();
                llvm::BasicBlock& entry_block = func->getEntryBlock();
                llvm::IRBuilder<>::InsertPoint saved_ip = builder.saveIP();

                // Find the first non-alloca instruction to insert before it
                llvm::Instruction* first_non_alloca = nullptr;
                for (auto& inst : entry_block) {
                    if (!llvm::isa<llvm::AllocaInst>(inst)) {
                        first_non_alloca = &inst;
                        break;
                    }
                }

                if (first_non_alloca) {
                    builder.SetInsertPoint(first_non_alloca);
                } else {
                    builder.SetInsertPoint(&entry_block);
                }

                // Handle array types
                if (var_type->isConstantArrayType()) {
                    const clang::ConstantArrayType* array_type =
                        context.getAsConstantArrayType(var_type);
                    uint64_t array_size = array_type->getSize().getZExtValue();
                    clang::QualType element_type = array_type->getElementType();

                    llvm::Type* llvm_element_type = convertType(element_type);
                    llvm::ArrayType* llvm_array_type = llvm::ArrayType::get(llvm_element_type, array_size);

                    llvm::Value* alloca = builder.CreateAlloca(llvm_array_type, nullptr,
                                                               var_decl->getNameAsString());
                    var_map[var_decl] = alloca;

                    // Restore insertion point for initialization
                    builder.restoreIP(saved_ip);

                    llvm::errs() << "[translateStmt] Created array alloca: " << var_decl->getNameAsString()
                               << " [" << array_size << "]\n";

                    // Handle array initialization if present
                    if (var_decl->hasInit()) {
                        if (auto* init_list = llvm::dyn_cast<clang::InitListExpr>(var_decl->getInit())) {
                            llvm::errs() << "[translateStmt] Initializing array with "
                                       << init_list->getNumInits() << " elements\n";

                            for (unsigned i = 0; i < init_list->getNumInits(); i++) {
                                llvm::Value* init_val = translateExpr(init_list->getInit(i),
                                                                     builder, context, var_map);
                                if (init_val) {
                                    // GEP to get pointer to array element
                                    llvm::Value* indices[] = {
                                        llvm::ConstantInt::get(builder.getInt32Ty(), 0),
                                        llvm::ConstantInt::get(builder.getInt32Ty(), i)
                                    };
                                    llvm::Value* elem_ptr = builder.CreateGEP(llvm_array_type, alloca,
                                                                             indices, "arrayinit");
                                    builder.CreateStore(init_val, elem_ptr);
                                }
                            }
                        }
                    }
                } else {
                    // Handle scalar types
                    llvm::Type* llvm_var_type = convertType(var_type);
                    llvm::Value* alloca = builder.CreateAlloca(llvm_var_type, nullptr,
                                                               var_decl->getNameAsString());
                    var_map[var_decl] = alloca;

                    // Restore insertion point for initialization
                    builder.restoreIP(saved_ip);

                    llvm::errs() << "[translateStmt] Created scalar alloca: " << var_decl->getNameAsString() << "\n";

                    // Handle initialization
                    if (var_decl->hasInit()) {
                        llvm::Value* init_val = translateExpr(var_decl->getInit(), builder, context, var_map);
                        if (init_val) {
                            // If init_val is a pointer and we're initializing a value, load it
                            if (init_val->getType()->isPointerTy() && !llvm_var_type->isPointerTy()) {
                                init_val = builder.CreateLoad(llvm_var_type, init_val, "init_load");
                            }
                            builder.CreateStore(init_val, alloca);
                            llvm::errs() << "[translateStmt] Initialized variable\n";
                        }
                    }
                }
            }
        }
        return;
    }

    // Handle if statements
    if (auto* if_stmt = llvm::dyn_cast<clang::IfStmt>(stmt)) {
        llvm::errs() << "[translateStmt] IfStmt\n";

        // Get the parent function
        llvm::Function* func = builder.GetInsertBlock()->getParent();

        // Translate condition
        llvm::Value* cond = translateExpr(if_stmt->getCond(), builder, context, var_map);
        if (!cond) {
            llvm::errs() << "[translateStmt] Failed to translate if condition\n";
            return;
        }

        // If condition is a pointer (i.e., a boolean variable), load it
        if (cond->getType()->isPointerTy()) {
            cond = builder.CreateLoad(builder.getInt1Ty(), cond, "cond_load");
        }

        // Convert float comparisons to boolean if needed
        if (cond->getType()->isFloatingPointTy()) {
            cond = builder.CreateFCmpONE(cond, llvm::ConstantFP::get(cond->getType(), 0.0), "tobool");
        } else if (cond->getType()->isIntegerTy() && !cond->getType()->isIntegerTy(1)) {
            cond = builder.CreateICmpNE(cond, llvm::ConstantInt::get(cond->getType(), 0), "tobool");
        }

        // Create basic blocks
        llvm::BasicBlock* then_bb = llvm::BasicBlock::Create(*llvm_context_, "if.then", func);
        llvm::BasicBlock* else_bb = if_stmt->getElse() ?
            llvm::BasicBlock::Create(*llvm_context_, "if.else", func) : nullptr;
        llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(*llvm_context_, "if.end", func);

        // Conditional branch
        if (else_bb) {
            builder.CreateCondBr(cond, then_bb, else_bb);
        } else {
            builder.CreateCondBr(cond, then_bb, merge_bb);
        }

        // Then block
        builder.SetInsertPoint(then_bb);
        translateStmt(if_stmt->getThen(), builder, context, var_map);
        if (!then_bb->getTerminator()) {
            builder.CreateBr(merge_bb);
        }

        // Else block (if present)
        if (else_bb) {
            builder.SetInsertPoint(else_bb);
            translateStmt(if_stmt->getElse(), builder, context, var_map);
            if (!else_bb->getTerminator()) {
                builder.CreateBr(merge_bb);
            }
        }

        // Merge block
        builder.SetInsertPoint(merge_bb);
        return;
    }

    // Handle for loops
    if (auto* for_stmt = llvm::dyn_cast<clang::ForStmt>(stmt)) {
        llvm::errs() << "[translateStmt] ForStmt\n";

        llvm::Function* func = builder.GetInsertBlock()->getParent();

        // Initialization
        if (for_stmt->getInit()) {
            translateStmt(for_stmt->getInit(), builder, context, var_map);
        }

        // Create basic blocks
        llvm::BasicBlock* cond_bb = llvm::BasicBlock::Create(*llvm_context_, "for.cond", func);
        llvm::BasicBlock* body_bb = llvm::BasicBlock::Create(*llvm_context_, "for.body", func);
        llvm::BasicBlock* inc_bb = llvm::BasicBlock::Create(*llvm_context_, "for.inc", func);
        llvm::BasicBlock* end_bb = llvm::BasicBlock::Create(*llvm_context_, "for.end", func);

        // Branch to condition
        builder.CreateBr(cond_bb);

        // Condition block
        builder.SetInsertPoint(cond_bb);
        if (for_stmt->getCond()) {
            llvm::Value* cond = translateExpr(for_stmt->getCond(), builder, context, var_map);
            if (cond) {
                // Convert to boolean if needed
                if (cond->getType()->isPointerTy()) {
                    cond = builder.CreateLoad(builder.getInt1Ty(), cond, "cond_load");
                }
                if (cond->getType()->isFloatingPointTy()) {
                    cond = builder.CreateFCmpONE(cond, llvm::ConstantFP::get(cond->getType(), 0.0), "tobool");
                } else if (cond->getType()->isIntegerTy() && !cond->getType()->isIntegerTy(1)) {
                    cond = builder.CreateICmpNE(cond, llvm::ConstantInt::get(cond->getType(), 0), "tobool");
                }
                builder.CreateCondBr(cond, body_bb, end_bb);
            } else {
                builder.CreateBr(body_bb);
            }
        } else {
            // No condition means infinite loop (or until break)
            builder.CreateBr(body_bb);
        }

        // Body block
        builder.SetInsertPoint(body_bb);
        if (for_stmt->getBody()) {
            translateStmt(for_stmt->getBody(), builder, context, var_map);
        }
        if (!body_bb->getTerminator()) {
            builder.CreateBr(inc_bb);
        }

        // Increment block
        builder.SetInsertPoint(inc_bb);
        if (for_stmt->getInc()) {
            translateExpr(for_stmt->getInc(), builder, context, var_map);
        }
        builder.CreateBr(cond_bb);

        // End block
        builder.SetInsertPoint(end_bb);
        return;
    }

    // Handle while loops
    if (auto* while_stmt = llvm::dyn_cast<clang::WhileStmt>(stmt)) {
        llvm::errs() << "[translateStmt] WhileStmt\n";

        llvm::Function* func = builder.GetInsertBlock()->getParent();

        llvm::BasicBlock* cond_bb = llvm::BasicBlock::Create(*llvm_context_, "while.cond", func);
        llvm::BasicBlock* body_bb = llvm::BasicBlock::Create(*llvm_context_, "while.body", func);
        llvm::BasicBlock* end_bb = llvm::BasicBlock::Create(*llvm_context_, "while.end", func);

        builder.CreateBr(cond_bb);

        // Condition
        builder.SetInsertPoint(cond_bb);
        llvm::Value* cond = translateExpr(while_stmt->getCond(), builder, context, var_map);
        if (cond) {
            if (cond->getType()->isPointerTy()) {
                cond = builder.CreateLoad(builder.getInt1Ty(), cond, "cond_load");
            }
            if (cond->getType()->isFloatingPointTy()) {
                cond = builder.CreateFCmpONE(cond, llvm::ConstantFP::get(cond->getType(), 0.0), "tobool");
            } else if (cond->getType()->isIntegerTy() && !cond->getType()->isIntegerTy(1)) {
                cond = builder.CreateICmpNE(cond, llvm::ConstantInt::get(cond->getType(), 0), "tobool");
            }
            builder.CreateCondBr(cond, body_bb, end_bb);
        }

        // Body
        builder.SetInsertPoint(body_bb);
        translateStmt(while_stmt->getBody(), builder, context, var_map);
        if (!body_bb->getTerminator()) {
            builder.CreateBr(cond_bb);
        }

        builder.SetInsertPoint(end_bb);
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

std::unique_ptr<llvm::Module> LambdaIRGenerator::generateIR(
    clang::CXXMethodDecl* method,
    clang::ASTContext& context) {

    llvm::errs() << "[LambdaIRGenerator V2] Generating IR with Clang CodeGen\n";

    // Extract full class context
    ClassContext class_ctx = class_extractor_.extract(method, context);

    llvm::errs() << "[V2] Found " << class_ctx.member_variables.size()
                 << " member variables\n";
    llvm::errs() << "[V2] Found " << class_ctx.member_functions.size()
                 << " member functions\n";

    // Generate IR using Clang CodeGen
    auto module = generateWithCodeGen(method, class_ctx, context);
    
    if (!module) {
        llvm::errs() << "[V2] ERROR: CodeGen failed, falling back to manual IR\n";
        return generateIRManualFallback(method, context);
    }
    
    return module;
}

std::unique_ptr<llvm::Module> LambdaIRGenerator::generateWithCodeGen(
    clang::CXXMethodDecl* method,
    const ClassContext& context,
    clang::ASTContext& ast_context) {

    // Step 1: Create CodeGenerator
    std::unique_ptr<clang::CodeGenerator> codegen(
        clang::CreateLLVMCodeGen(
            CI_.getDiagnostics(),
            "parallax_kernel",
            llvm::vfs::getRealFileSystem(),
            CI_.getHeaderSearchOpts(),
            CI_.getPreprocessorOpts(),
            CI_.getCodeGenOpts(),
            *llvm_context_
        )
    );

    if (!codegen) {
        llvm::errs() << "[V2] ERROR: Failed to create CodeGenerator\n";
        return nullptr;
    }

    // Step 2: Initialize CodeGen
    codegen->Initialize(ast_context);

    // Step 3: Generate IR for the class definition
    llvm::errs() << "[V2] Generating IR for class: "
                 << context.record->getNameAsString() << "\n";

    codegen->HandleTopLevelDecl(
        clang::DeclGroupRef(context.record)
    );

    // Step 4: Generate IR for operator() and called member functions
    for (clang::CXXMethodDecl* func : context.member_functions) {
        llvm::errs() << "[V2] Generating IR for member function: "
                     << func->getNameAsString() << "\n";
        codegen->HandleTopLevelDecl(clang::DeclGroupRef(func));
    }

    codegen->HandleTopLevelDecl(clang::DeclGroupRef(method));

    // Step 5: Finalize
    codegen->HandleTranslationUnit(ast_context);

    // Step 6: Extract module
    std::unique_ptr<llvm::Module> module(codegen->ReleaseModule());

    if (!module) {
        llvm::errs() << "[V2] ERROR: CodeGen returned null module\n";
        return nullptr;
    }

    // Step 7: Verify IR
    std::string verify_errors;
    llvm::raw_string_ostream error_stream(verify_errors);
    if (llvm::verifyModule(*module, &error_stream)) {
        llvm::errs() << "[V2] ERROR: Module verification failed:\n"
                     << verify_errors << "\n";
        module->print(llvm::errs(), nullptr);
        return nullptr;
    }

    llvm::errs() << "[V2] Successfully generated " << module->size()
                 << " functions\n";

    // Step 8: Generate GPU kernel wrapper
    KernelWrapper wrapper(*llvm_context_);
    llvm::Function* kernel = wrapper.generateWrapper(context, module.get());

    if (!kernel) {
        llvm::errs() << "[V2] ERROR: Failed to generate kernel wrapper\n";
        return nullptr;
    }

    llvm::errs() << "[V2] Generated GPU kernel: " << kernel->getName().str() << "\n";

    return module;
}

std::unique_ptr<llvm::Module> LambdaIRGenerator::generateIRManualFallback(
    clang::CXXMethodDecl* method,
    clang::ASTContext& context) {

    llvm::errs() << "[LambdaIRGenerator] Using manual IR fallback\n";

    std::string module_name = "functor_manual_" + method->getQualifiedNameAsString();
    auto module = std::make_unique<llvm::Module>(module_name, *llvm_context_);

    // Build function signature
    std::vector<llvm::Type*> param_types;
    std::map<const clang::VarDecl*, llvm::Value*> var_map;

    for (auto* param : method->parameters()) {
        llvm::Type* param_type = convertType(param->getType());
        param_types.push_back(param_type);
    }

    llvm::Type* return_type = convertType(method->getReturnType());
    llvm::FunctionType* func_type = llvm::FunctionType::get(return_type, param_types, false);

    // Create function
    llvm::Function* func = llvm::Function::Create(
        func_type,
        llvm::Function::ExternalLinkage,
        "lambda_kernel_fallback",
        module.get()
    );

    // Map parameters to LLVM arguments
    auto arg_it = func->arg_begin();
    for (auto* param : method->parameters()) {
        llvm::Argument* arg = &(*arg_it++);
        arg->setName(param->getNameAsString());
        var_map[param] = arg;
    }

    // Create entry basic block
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(*llvm_context_, "entry", func);
    llvm::IRBuilder<> builder(entry);

    // Translate method body
    if (clang::Stmt* body = method->getBody()) {
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

    llvm::errs() << "\n[LambdaIRGenerator] Generated fallback LLVM IR:\n";
    func->print(llvm::errs());
    llvm::errs() << "\n";

    return module;
}

} // namespace parallax
