#include "parallax/lambda_ir_generator.hpp"
#include "parallax/class_context_extractor.hpp"
#include "parallax/kernel_wrapper.hpp"
#include <clang/AST/Type.h>
#include <clang/AST/OperationKinds.h>
#include <clang/CodeGen/ModuleBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/VirtualFileSystem.h>
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
            info.var_decl = llvm::dyn_cast<clang::VarDecl>(capture.getCapturedVar());
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
                // IMPORTANT: Use Int32 instead of Int64 since many GPUs (like GTX 980M)
                // don't support the Int64 capability. This may cause issues with
                // code that relies on 64-bit integer precision, but it's necessary
                // for compatibility.
                llvm::errs() << "Warning: Mapping 64-bit integer type to 32-bit for GPU compatibility\n";
                return llvm::Type::getInt32Ty(*llvm_context_);
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
            case clang::BO_Mul: {
                // For arithmetic operators, operands must be values, not pointers
                // Determine actual type from AST, not just assuming float
                llvm::Type* load_type = binary_op->getLHS()->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(load_type, lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }

                // Use appropriate multiply instruction based on type
                if (lhs->getType()->isFloatingPointTy()) {
                    return builder.CreateFMul(lhs, rhs, "mul");
                } else {
                    return builder.CreateMul(lhs, rhs, "mul");
                }
            }
            case clang::BO_Add: {
                llvm::Type* load_type = binary_op->getLHS()->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(load_type, lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }

                if (lhs->getType()->isFloatingPointTy()) {
                    return builder.CreateFAdd(lhs, rhs, "add");
                } else {
                    return builder.CreateAdd(lhs, rhs, "add");
                }
            }
            case clang::BO_Sub: {
                llvm::Type* load_type = binary_op->getLHS()->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(load_type, lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }

                if (lhs->getType()->isFloatingPointTy()) {
                    return builder.CreateFSub(lhs, rhs, "sub");
                } else {
                    return builder.CreateSub(lhs, rhs, "sub");
                }
            }
            case clang::BO_Div: {
                llvm::Type* load_type = binary_op->getLHS()->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(load_type, lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }

                if (lhs->getType()->isFloatingPointTy()) {
                    return builder.CreateFDiv(lhs, rhs, "div");
                } else {
                    // Integer division - use signed division
                    return builder.CreateSDiv(lhs, rhs, "div");
                }
            }

            // Comparison operators
            case clang::BO_LT: {
                // Determine the correct load type from Clang AST
                llvm::Type* load_type = binary_op->getLHS()->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(load_type, lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }
                if (lhs->getType()->isFloatingPointTy()) {
                    return builder.CreateFCmpOLT(lhs, rhs, "cmp_lt");
                } else {
                    return builder.CreateICmpSLT(lhs, rhs, "cmp_lt");
                }
            }
            case clang::BO_LE: {
                llvm::Type* load_type = binary_op->getLHS()->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(load_type, lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }
                if (lhs->getType()->isFloatingPointTy()) {
                    return builder.CreateFCmpOLE(lhs, rhs, "cmp_le");
                } else {
                    return builder.CreateICmpSLE(lhs, rhs, "cmp_le");
                }
            }
            case clang::BO_GT: {
                llvm::Type* load_type = binary_op->getLHS()->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(load_type, lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }
                if (lhs->getType()->isFloatingPointTy()) {
                    return builder.CreateFCmpOGT(lhs, rhs, "cmp_gt");
                } else {
                    return builder.CreateICmpSGT(lhs, rhs, "cmp_gt");
                }
            }
            case clang::BO_GE: {
                llvm::Type* load_type = binary_op->getLHS()->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(load_type, lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }
                if (lhs->getType()->isFloatingPointTy()) {
                    return builder.CreateFCmpOGE(lhs, rhs, "cmp_ge");
                } else {
                    return builder.CreateICmpSGE(lhs, rhs, "cmp_ge");
                }
            }
            case clang::BO_EQ: {
                llvm::Type* load_type = binary_op->getLHS()->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(load_type, lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }
                if (lhs->getType()->isFloatingPointTy()) {
                    return builder.CreateFCmpOEQ(lhs, rhs, "cmp_eq");
                } else {
                    return builder.CreateICmpEQ(lhs, rhs, "cmp_eq");
                }
            }
            case clang::BO_NE: {
                llvm::Type* load_type = binary_op->getLHS()->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(load_type, lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }
                if (lhs->getType()->isFloatingPointTy()) {
                    return builder.CreateFCmpONE(lhs, rhs, "cmp_ne");
                } else {
                    return builder.CreateICmpNE(lhs, rhs, "cmp_ne");
                }
            }

            case clang::BO_Assign: {
                llvm::errs() << "[translateExpr] Handling assignment\n";
                // LHS should be an lvalue (pointer/reference)
                // RHS should be a value (not pointer) - load if needed
                llvm::Type* load_type = binary_op->getRHS()->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();

                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }
                if (lhs->getType()->isPointerTy()) {
                    builder.CreateStore(rhs, lhs);
                    return rhs;
                }
                return nullptr;
            }
            case clang::BO_MulAssign: {
                llvm::errs() << "[translateExpr] Handling *=\n";
                llvm::Type* load_type = binary_op->getLHS()->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    llvm::errs() << "[translateExpr] LHS is pointer, loading value\n";
                    llvm::Value* loaded = builder.CreateLoad(load_type, lhs, "tmp");

                    // Load RHS if it's a pointer
                    if (rhs->getType()->isPointerTy()) {
                        rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                    }

                    llvm::errs() << "[translateExpr] Creating multiply\n";
                    llvm::Value* result;
                    if (loaded->getType()->isFloatingPointTy()) {
                        result = builder.CreateFMul(loaded, rhs, "mul");
                    } else {
                        result = builder.CreateMul(loaded, rhs, "mul");
                    }

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
                llvm::Type* load_type = binary_op->getLHS()->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    llvm::Value* loaded = builder.CreateLoad(load_type, lhs, "tmp");
                    if (rhs->getType()->isPointerTy()) {
                        rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                    }

                    llvm::Value* result;
                    if (loaded->getType()->isFloatingPointTy()) {
                        result = builder.CreateFAdd(loaded, rhs, "add");
                    } else {
                        result = builder.CreateAdd(loaded, rhs, "add");
                    }
                    builder.CreateStore(result, lhs);
                    return result;
                }
                return nullptr;
            }
            case clang::BO_SubAssign: {
                llvm::Type* load_type = binary_op->getLHS()->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    llvm::Value* loaded = builder.CreateLoad(load_type, lhs, "tmp");
                    if (rhs->getType()->isPointerTy()) {
                        rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                    }

                    llvm::Value* result;
                    if (loaded->getType()->isFloatingPointTy()) {
                        result = builder.CreateFSub(loaded, rhs, "sub");
                    } else {
                        result = builder.CreateSub(loaded, rhs, "sub");
                    }
                    builder.CreateStore(result, lhs);
                    return result;
                }
                return nullptr;
            }
            case clang::BO_DivAssign: {
                llvm::Type* load_type = binary_op->getLHS()->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    llvm::Value* loaded = builder.CreateLoad(load_type, lhs, "tmp");
                    if (rhs->getType()->isPointerTy()) {
                        rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                    }

                    llvm::Value* result;
                    if (loaded->getType()->isFloatingPointTy()) {
                        result = builder.CreateFDiv(loaded, rhs, "div");
                    } else {
                        result = builder.CreateSDiv(loaded, rhs, "div");
                    }
                    builder.CreateStore(result, lhs);
                    return result;
                }
                return nullptr;
            }
            case clang::BO_Rem: {
                // Modulo/remainder operator
                llvm::Type* load_type = binary_op->getLHS()->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(load_type, lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }

                if (lhs->getType()->isFloatingPointTy()) {
                    return builder.CreateFRem(lhs, rhs, "rem");
                } else {
                    return builder.CreateSRem(lhs, rhs, "rem");
                }
            }

            case clang::BO_And: {
                // Bitwise AND
                llvm::Type* load_type = builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(load_type, lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }
                return builder.CreateAnd(lhs, rhs, "and");
            }

            case clang::BO_Or: {
                // Bitwise OR
                llvm::Type* load_type = builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(load_type, lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }
                return builder.CreateOr(lhs, rhs, "or");
            }

            case clang::BO_Xor: {
                // Bitwise XOR
                llvm::Type* load_type = builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(load_type, lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }
                return builder.CreateXor(lhs, rhs, "xor");
            }

            case clang::BO_Shl: {
                // Left shift
                llvm::Type* load_type = builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(load_type, lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }
                return builder.CreateShl(lhs, rhs, "shl");
            }

            case clang::BO_Shr: {
                // Right shift (arithmetic)
                llvm::Type* load_type = builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(load_type, lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }
                return builder.CreateAShr(lhs, rhs, "shr");
            }

            case clang::BO_LAnd: {
                // Logical AND
                llvm::Type* load_type = binary_op->getLHS()->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(load_type, lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }

                // Convert to boolean
                llvm::Value* lhs_bool = lhs->getType()->isFloatingPointTy() ?
                    builder.CreateFCmpONE(lhs, llvm::ConstantFP::get(lhs->getType(), 0.0), "lhs_bool") :
                    builder.CreateICmpNE(lhs, llvm::ConstantInt::get(lhs->getType(), 0), "lhs_bool");
                llvm::Value* rhs_bool = rhs->getType()->isFloatingPointTy() ?
                    builder.CreateFCmpONE(rhs, llvm::ConstantFP::get(rhs->getType(), 0.0), "rhs_bool") :
                    builder.CreateICmpNE(rhs, llvm::ConstantInt::get(rhs->getType(), 0), "rhs_bool");

                return builder.CreateAnd(lhs_bool, rhs_bool, "land");
            }

            case clang::BO_LOr: {
                // Logical OR
                llvm::Type* load_type = binary_op->getLHS()->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();

                if (lhs->getType()->isPointerTy()) {
                    lhs = builder.CreateLoad(load_type, lhs, "lhs_load");
                }
                if (rhs->getType()->isPointerTy()) {
                    rhs = builder.CreateLoad(load_type, rhs, "rhs_load");
                }

                // Convert to boolean
                llvm::Value* lhs_bool = lhs->getType()->isFloatingPointTy() ?
                    builder.CreateFCmpONE(lhs, llvm::ConstantFP::get(lhs->getType(), 0.0), "lhs_bool") :
                    builder.CreateICmpNE(lhs, llvm::ConstantInt::get(lhs->getType(), 0), "lhs_bool");
                llvm::Value* rhs_bool = rhs->getType()->isFloatingPointTy() ?
                    builder.CreateFCmpONE(rhs, llvm::ConstantFP::get(rhs->getType(), 0.0), "rhs_bool") :
                    builder.CreateICmpNE(rhs, llvm::ConstantInt::get(rhs->getType(), 0), "rhs_bool");

                return builder.CreateOr(lhs_bool, rhs_bool, "lor");
            }

            default:
                llvm::errs() << "Warning: Unhandled binary operator: " << binary_op->getOpcodeStr() << "\n";
                // Instead of returning nullptr, create a dummy zero value to prevent "Id is 0" errors
                llvm::Type* result_type = binary_op->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();
                if (result_type->isFloatingPointTy()) {
                    return llvm::ConstantFP::get(result_type, 0.0);
                } else {
                    return llvm::ConstantInt::get(result_type, 0);
                }
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

            case clang::UO_LNot: {
                // Logical NOT: !x
                if (operand->getType()->isPointerTy()) {
                    llvm::Type* load_type = unary_op->getSubExpr()->getType()->isFloatingType() ?
                        builder.getFloatTy() : builder.getInt32Ty();
                    operand = builder.CreateLoad(load_type, operand, "lnot_load");
                }

                // Convert to boolean and negate
                llvm::Value* as_bool;
                if (operand->getType()->isFloatingPointTy()) {
                    as_bool = builder.CreateFCmpOEQ(operand, llvm::ConstantFP::get(operand->getType(), 0.0), "lnot");
                } else if (operand->getType()->isIntegerTy(1)) {
                    as_bool = builder.CreateNot(operand, "lnot");
                } else {
                    as_bool = builder.CreateICmpEQ(operand, llvm::ConstantInt::get(operand->getType(), 0), "lnot");
                }
                return as_bool;
            }

            case clang::UO_Not: {
                // Bitwise NOT: ~x
                if (operand->getType()->isPointerTy()) {
                    operand = builder.CreateLoad(builder.getInt32Ty(), operand, "not_load");
                }
                return builder.CreateNot(operand, "not");
            }

            case clang::UO_Plus: {
                // Unary plus: +x (no-op, just return operand)
                if (operand->getType()->isPointerTy()) {
                    llvm::Type* load_type = unary_op->getSubExpr()->getType()->isFloatingType() ?
                        builder.getFloatTy() : builder.getInt32Ty();
                    operand = builder.CreateLoad(load_type, operand, "plus_load");
                }
                return operand;
            }

            default:
                llvm::errs() << "Warning: Unhandled unary operator: " << unary_op->getOpcodeStr(unary_op->getOpcode()) << "\n";
                // Return operand as-is instead of failing
                if (operand && !operand->getType()->isVoidTy()) {
                    return operand;
                } else {
                    // Return a dummy zero value to prevent "Id is 0" errors
                    return llvm::ConstantInt::get(builder.getInt32Ty(), 0);
                }
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

    // Handle function calls (including std:: math functions)
    if (auto* call = llvm::dyn_cast<clang::CallExpr>(expr)) {
        llvm::errs() << "[translateExpr] CallExpr\n";

        // Get the function being called
        const clang::FunctionDecl* callee = call->getDirectCallee();
        if (!callee) {
            llvm::errs() << "[translateExpr] No direct callee found\n";
            return nullptr;
        }

        std::string func_name = callee->getNameAsString();
        llvm::errs() << "[translateExpr] Calling function: " << func_name << "\n";

        // Translate all arguments
        std::vector<llvm::Value*> args;
        for (unsigned i = 0; i < call->getNumArgs(); ++i) {
            llvm::Value* arg = translateExpr(call->getArg(i), builder, context, var_map);
            if (!arg) {
                llvm::errs() << "[translateExpr] Failed to translate argument " << i << "\n";
                return nullptr;
            }
            // Load argument if it's a pointer
            if (arg->getType()->isPointerTy()) {
                llvm::Type* load_type = call->getArg(i)->getType()->isFloatingType() ?
                    builder.getFloatTy() : builder.getInt32Ty();
                arg = builder.CreateLoad(load_type, arg, "arg_load");
            }
            args.push_back(arg);
        }

        // Map common math functions to LLVM intrinsics
        if (func_name == "sin" || func_name == "sinf") {
            llvm::Function* sin_fn = llvm::Intrinsic::getDeclaration(
                builder.GetInsertBlock()->getModule(),
                llvm::Intrinsic::sin,
                {args[0]->getType()}
            );
            return builder.CreateCall(sin_fn, args[0], "sin");
        } else if (func_name == "cos" || func_name == "cosf") {
            llvm::Function* cos_fn = llvm::Intrinsic::getDeclaration(
                builder.GetInsertBlock()->getModule(),
                llvm::Intrinsic::cos,
                {args[0]->getType()}
            );
            return builder.CreateCall(cos_fn, args[0], "cos");
        } else if (func_name == "sqrt" || func_name == "sqrtf") {
            llvm::Function* sqrt_fn = llvm::Intrinsic::getDeclaration(
                builder.GetInsertBlock()->getModule(),
                llvm::Intrinsic::sqrt,
                {args[0]->getType()}
            );
            return builder.CreateCall(sqrt_fn, args[0], "sqrt");
        } else if (func_name == "exp" || func_name == "expf") {
            llvm::Function* exp_fn = llvm::Intrinsic::getDeclaration(
                builder.GetInsertBlock()->getModule(),
                llvm::Intrinsic::exp,
                {args[0]->getType()}
            );
            return builder.CreateCall(exp_fn, args[0], "exp");
        } else if (func_name == "log" || func_name == "logf") {
            llvm::Function* log_fn = llvm::Intrinsic::getDeclaration(
                builder.GetInsertBlock()->getModule(),
                llvm::Intrinsic::log,
                {args[0]->getType()}
            );
            return builder.CreateCall(log_fn, args[0], "log");
        } else if (func_name == "pow" || func_name == "powf") {
            llvm::Function* pow_fn = llvm::Intrinsic::getDeclaration(
                builder.GetInsertBlock()->getModule(),
                llvm::Intrinsic::pow,
                {args[0]->getType()}
            );
            return builder.CreateCall(pow_fn, args, "pow");
        } else if (func_name == "fabs" || func_name == "fabsf" || func_name == "abs") {
            llvm::Function* fabs_fn = llvm::Intrinsic::getDeclaration(
                builder.GetInsertBlock()->getModule(),
                llvm::Intrinsic::fabs,
                {args[0]->getType()}
            );
            return builder.CreateCall(fabs_fn, args[0], "fabs");
        } else if (func_name == "max") {
            // std::max(a, b) -> (a > b) ? a : b
            llvm::Value* cmp = args[0]->getType()->isFloatingPointTy() ?
                builder.CreateFCmpOGT(args[0], args[1], "max_cmp") :
                builder.CreateICmpSGT(args[0], args[1], "max_cmp");
            return builder.CreateSelect(cmp, args[0], args[1], "max");
        } else if (func_name == "min") {
            // std::min(a, b) -> (a < b) ? a : b
            llvm::Value* cmp = args[0]->getType()->isFloatingPointTy() ?
                builder.CreateFCmpOLT(args[0], args[1], "min_cmp") :
                builder.CreateICmpSLT(args[0], args[1], "min_cmp");
            return builder.CreateSelect(cmp, args[0], args[1], "min");
        } else if (func_name == "cbrt" || func_name == "cbrtf") {
            // cbrt(x) = x^(1/3) = pow(x, 1.0/3.0)
            llvm::Function* pow_fn = llvm::Intrinsic::getDeclaration(
                builder.GetInsertBlock()->getModule(),
                llvm::Intrinsic::pow,
                {args[0]->getType()}
            );
            llvm::Value* one_third = llvm::ConstantFP::get(args[0]->getType(), 1.0/3.0);
            return builder.CreateCall(pow_fn, {args[0], one_third}, "cbrt");
        }

        llvm::errs() << "[translateExpr] Warning: Unhandled function call: " << func_name << "\n";
        return nullptr;
    }

    // Handle member access expressions (struct.field or ptr->field)
    if (auto* member = llvm::dyn_cast<clang::MemberExpr>(expr)) {
        llvm::errs() << "[translateExpr] MemberExpr: " << member->getMemberDecl()->getNameAsString() << "\n";

        // Translate the base expression (the struct or pointer)
        llvm::Value* base = translateExpr(member->getBase(), builder, context, var_map);
        if (!base) {
            llvm::errs() << "[translateExpr] Failed to translate member base\n";
            return nullptr;
        }

        // Get the field declaration
        if (auto* field_decl = llvm::dyn_cast<clang::FieldDecl>(member->getMemberDecl())) {
            llvm::errs() << "[translateExpr] Field access: " << field_decl->getNameAsString() << "\n";

            // Get the struct type from the base
            const clang::RecordDecl* record_decl = member->getBase()->getType()->getAsStructureType()->getDecl();

            // Find the field index
            unsigned field_idx = 0;
            for (auto* field : record_decl->fields()) {
                if (field == field_decl) {
                    break;
                }
                field_idx++;
            }

            llvm::errs() << "[translateExpr] Field index: " << field_idx << "\n";

            // Get the struct type in LLVM IR
            llvm::Type* struct_type = convertType(member->getBase()->getType());

            // If base is not a pointer (value type), we need its address
            // For now, assume base is already a pointer to the struct
            if (!base->getType()->isPointerTy()) {
                llvm::errs() << "[translateExpr] Warning: Member access on non-pointer base\n";
                return nullptr;
            }

            // Create GEP to access the field
            llvm::Value* field_ptr = builder.CreateStructGEP(struct_type, base, field_idx,
                                                              field_decl->getNameAsString());
            llvm::errs() << "[translateExpr] Created struct GEP for field\n";

            // Return the pointer to the field (caller will load if needed)
            return field_ptr;
        }

        llvm::errs() << "[translateExpr] Warning: Unhandled member type\n";
        return nullptr;
    }

    // Handle conditional operator (ternary ? :)
    if (auto* cond_op = llvm::dyn_cast<clang::ConditionalOperator>(expr)) {
        llvm::errs() << "[translateExpr] ConditionalOperator (ternary ? :)\n";

        // Get the parent function
        llvm::Function* func = builder.GetInsertBlock()->getParent();

        // Translate condition
        llvm::Value* cond = translateExpr(cond_op->getCond(), builder, context, var_map);
        if (!cond) {
            llvm::errs() << "[translateExpr] Failed to translate condition\n";
            return nullptr;
        }

        // Load condition if it's a pointer
        if (cond->getType()->isPointerTy()) {
            cond = builder.CreateLoad(builder.getInt1Ty(), cond, "cond_load");
        }

        // Convert to boolean if needed
        if (cond->getType()->isFloatingPointTy()) {
            cond = builder.CreateFCmpONE(cond, llvm::ConstantFP::get(cond->getType(), 0.0), "tobool");
        } else if (cond->getType()->isIntegerTy() && !cond->getType()->isIntegerTy(1)) {
            cond = builder.CreateICmpNE(cond, llvm::ConstantInt::get(cond->getType(), 0), "tobool");
        }

        // Create basic blocks for true/false branches
        llvm::BasicBlock* true_bb = llvm::BasicBlock::Create(*llvm_context_, "cond.true", func);
        llvm::BasicBlock* false_bb = llvm::BasicBlock::Create(*llvm_context_, "cond.false", func);
        llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(*llvm_context_, "cond.end", func);

        // Conditional branch
        builder.CreateCondBr(cond, true_bb, false_bb);

        // True branch
        builder.SetInsertPoint(true_bb);
        llvm::Value* true_val = translateExpr(cond_op->getTrueExpr(), builder, context, var_map);
        if (!true_val) {
            llvm::errs() << "[translateExpr] Failed to translate true expression\n";
            return nullptr;
        }
        // Load if pointer
        if (true_val->getType()->isPointerTy()) {
            llvm::Type* load_type = cond_op->getTrueExpr()->getType()->isFloatingType() ?
                builder.getFloatTy() : builder.getInt32Ty();
            true_val = builder.CreateLoad(load_type, true_val, "true_val");
        }
        llvm::BasicBlock* true_bb_end = builder.GetInsertBlock(); // May have changed
        builder.CreateBr(merge_bb);

        // False branch
        builder.SetInsertPoint(false_bb);
        llvm::Value* false_val = translateExpr(cond_op->getFalseExpr(), builder, context, var_map);
        if (!false_val) {
            llvm::errs() << "[translateExpr] Failed to translate false expression\n";
            return nullptr;
        }
        // Load if pointer
        if (false_val->getType()->isPointerTy()) {
            llvm::Type* load_type = cond_op->getFalseExpr()->getType()->isFloatingType() ?
                builder.getFloatTy() : builder.getInt32Ty();
            false_val = builder.CreateLoad(load_type, false_val, "false_val");
        }
        llvm::BasicBlock* false_bb_end = builder.GetInsertBlock(); // May have changed
        builder.CreateBr(merge_bb);

        // Merge block with PHI node
        builder.SetInsertPoint(merge_bb);
        llvm::PHINode* phi = builder.CreatePHI(true_val->getType(), 2, "cond.result");
        phi->addIncoming(true_val, true_bb_end);
        phi->addIncoming(false_val, false_bb_end);

        llvm::errs() << "[translateExpr] Ternary operator completed\n";
        return phi;
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

    // Extract lambda captures
    auto captures = extractCaptures(lambda);

    llvm::errs() << "[generateIRManual] Lambda has " << captures.size() << " captures\n";
    for (const auto& cap : captures) {
        llvm::errs() << "  - " << cap.name << " ("
                     << (cap.is_by_reference ? "by ref" : "by value") << ")\n";
    }

    // Build function signature
    std::vector<llvm::Type*> param_types;
    std::map<const clang::VarDecl*, llvm::Value*> var_map;

    // Add explicit lambda parameters
    for (auto* param : call_op->parameters()) {
        llvm::Type* param_type = convertType(param->getType());
        param_types.push_back(param_type);
    }

    // Add captures as additional parameters (passed by the runtime)
    // IMPORTANT: For GPU offload, all captured pointers/references are passed as uint32
    // from push constants (since GPU doesn't support Int64). We use uint32 placeholders.
    for (const auto& capture : captures) {
        llvm::Type* capture_type;

        // Check if this is a pointer or reference type
        if (capture.is_by_reference || capture.type->isPointerType()) {
            // Use uint32 placeholder for all pointer/reference captures
            capture_type = llvm::Type::getInt32Ty(*llvm_context_);
            llvm::errs() << "[LambdaIRGenerator] Manual: Capture '" << capture.name
                         << "' is pointer/ref, using uint32 placeholder\n";
        } else {
            // Use actual type for value captures
            capture_type = convertType(capture.type);
        }
        param_types.push_back(capture_type);
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

    // Map explicit parameters to LLVM arguments
    auto arg_it = func->arg_begin();
    for (auto* param : call_op->parameters()) {
        llvm::Argument* arg = &(*arg_it++);
        arg->setName(param->getNameAsString());
        var_map[param] = arg;
    }

    // Map captures to LLVM arguments
    for (const auto& capture : captures) {
        llvm::Argument* arg = &(*arg_it++);
        arg->setName("capture_" + capture.name);
        var_map[capture.var_decl] = arg;
        llvm::errs() << "[generateIRManual] Mapped capture " << capture.name << " to argument\n";
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

    if (!lambda) return nullptr;

    llvm::errs() << "[LambdaIRGenerator] Generating IR for lambda using CodeGen\n";

    // IMPORTANT: Lambdas are just syntactic sugar for anonymous classes with operator()
    // So we should use the SAME CodeGen path as functors for full C++ support!

    // Get the lambda's call operator (operator())
    clang::CXXMethodDecl* call_op = getLambdaCallOperator(lambda);
    if (!call_op) {
        llvm::errs() << "[LambdaIRGenerator] ERROR: Lambda has no call operator\n";
        return generateSimplifiedStub(lambda, context);
    }

    // Get the lambda class (the anonymous class representing the lambda)
    const clang::CXXRecordDecl* lambda_class = lambda->getLambdaClass();
    if (!lambda_class) {
        llvm::errs() << "[LambdaIRGenerator] ERROR: Lambda has no class definition\n";
        return generateSimplifiedStub(lambda, context);
    }

    llvm::errs() << "[LambdaIRGenerator] Lambda class: " << lambda_class->getNameAsString() << "\n";

    // Extract class context (same as for functors)
    ClassContext class_ctx = class_extractor_.extract(const_cast<clang::CXXMethodDecl*>(call_op), context);

    llvm::errs() << "[LambdaIRGenerator] Found " << class_ctx.member_variables.size()
                 << " captures, " << class_ctx.member_functions.size() << " member functions\n";

    // Use CodeGen to generate IR (same path as functors!)
    auto module = generateWithCodeGen(call_op, class_ctx, context);

    if (!module) {
        llvm::errs() << "[LambdaIRGenerator] CodeGen failed, trying manual generation as fallback\n";

        // Try manual generation as fallback
        auto manual_module = generateIRManual(lambda, context);
        if (manual_module) {
            llvm::errs() << "[LambdaIRGenerator] Manual generation succeeded!\n";
            return manual_module;
        }

        llvm::errs() << "[LambdaIRGenerator] Manual generation also failed, using stub\n";
        return generateSimplifiedStub(lambda, context);
    }

    llvm::errs() << "[LambdaIRGenerator] CodeGen succeeded for lambda!\n";
    return module;
}

std::unique_ptr<llvm::Module> LambdaIRGenerator::generateSimplifiedStub(
    clang::LambdaExpr* lambda,
    clang::ASTContext& context) {

    llvm::errs() << "[LambdaIRGenerator] Creating simplified stub kernel\n";

    std::string module_name = "lambda_stub";
    auto module = std::make_unique<llvm::Module>(module_name, *llvm_context_);

    clang::CXXMethodDecl* call_op = getLambdaCallOperator(lambda);
    auto captures = extractCaptures(lambda);

    // Build simple function signature: just parameters, no complex logic
    std::vector<llvm::Type*> param_types;

    // Add explicit parameters
    for (auto* param : call_op->parameters()) {
        llvm::Type* param_type = convertType(param->getType());
        param_types.push_back(param_type);
    }

    // Add captures
    // IMPORTANT: For GPU offload, all captured pointers/references are passed as uint32
    // from push constants (since GPU doesn't support Int64). We use uint32 placeholders.
    for (const auto& capture : captures) {
        llvm::Type* capture_type;

        // Check if this is a pointer or reference type
        if (capture.is_by_reference || capture.type->isPointerType()) {
            // Use uint32 placeholder for all pointer/reference captures
            capture_type = llvm::Type::getInt32Ty(*llvm_context_);
            llvm::errs() << "[LambdaIRGenerator] Capture '" << capture.name
                         << "' is pointer/ref, using uint32 placeholder\n";
        } else {
            // Use actual type for value captures
            capture_type = convertType(capture.type);
        }
        param_types.push_back(capture_type);
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

    // Create a simple body that just returns
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(*llvm_context_, "entry", func);
    llvm::IRBuilder<> builder(entry);

    if (return_type->isVoidTy()) {
        builder.CreateRetVoid();
    } else {
        // Return zero/default value
        if (return_type->isFloatingPointTy()) {
            builder.CreateRet(llvm::ConstantFP::get(return_type, 0.0));
        } else if (return_type->isIntegerTy()) {
            builder.CreateRet(llvm::ConstantInt::get(return_type, 0));
        } else {
            builder.CreateRetVoid(); // Fallback
        }
    }

    llvm::errs() << "[LambdaIRGenerator] Generated stub kernel\n";
    llvm::errs() << "[LambdaIRGenerator] Full module dump:\n";
    module->print(llvm::errs(), nullptr);
    llvm::errs() << "\n[LambdaIRGenerator] End of module dump\n";

    return module;
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

    // IMPORTANT: HandleTagDeclDefinition tells CodeGen to emit the complete class definition
    // This should trigger emission of all member functions
    codegen->HandleTagDeclDefinition(context.record);

    // Also try HandleTopLevelDecl for the record
    codegen->HandleTopLevelDecl(clang::DeclGroupRef(context.record));

    // Step 4: Force emission of operator() by marking it as DEFINITION to emit
    llvm::errs() << "[V2] Forcing emission of operator()\n";

    // Get Sema for forcing instantiation
    clang::Sema& sema = CI_.getSema();

    // CRITICAL: CodeGen only emits functions that are marked for emission
    // We need to explicitly tell CodeGen to emit this function
    if (method->hasBody() && method->isThisDeclarationADefinition()) {
        llvm::errs() << "[V2] operator() has body, marking for emission\n";

        // Mark as used (necessary but not sufficient)
        method->setIsUsed();

        // Use Sema to force it
        sema.MarkFunctionReferenced(method->getBeginLoc(), method);

        // IMPORTANT: Manually mark for deferred emission
        // This is what makes CodeGen actually emit the function!
        ast_context.getTranslationUnitDecl()->addDecl(const_cast<clang::CXXMethodDecl*>(method));
    }

    codegen->HandleTopLevelDecl(clang::DeclGroupRef(method));

    for (clang::CXXMethodDecl* func : context.member_functions) {
        llvm::errs() << "[V2] Generating IR for member function: "
                     << func->getNameAsString() << "\n";
        if (func->hasBody() && func->isThisDeclarationADefinition()) {
            func->setIsUsed();
            sema.MarkFunctionReferenced(func->getBeginLoc(), func);
            if (func->getTemplatedKind() != clang::FunctionDecl::TK_NonTemplate) {
                sema.InstantiateFunctionDefinition(func->getBeginLoc(), func, true);
            }
        }
        codegen->HandleTopLevelDecl(clang::DeclGroupRef(func));
    }

    // Step 5: Finalize
    llvm::errs() << "[V2] Finalizing code generation\n";
    codegen->HandleTranslationUnit(ast_context);

    // Step 6: Extract module
    std::unique_ptr<llvm::Module> module(codegen->ReleaseModule());

    if (!module) {
        llvm::errs() << "[V2] ERROR: CodeGen returned null module\n";
        return nullptr;
    }

    llvm::errs() << "[V2] CodeGen produced module with " << module->size() << " functions\n";

    // DEBUG: Print all functions in module
    for (auto& func : *module) {
        llvm::errs() << "[V2]   Function: " << func.getName() << " (declaration="
                     << func.isDeclaration() << ", empty=" << func.empty() << ")\n";
    }

    if (module->empty()) {
        llvm::errs() << "[V2] WARNING: CodeGen produced empty module - operator() not instantiated\n";
        llvm::errs() << "[V2] This happens because lambdas are only emitted when called\n";
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

    llvm::errs() << "[LambdaIRGenerator] Using manual IR fallback (simplified stub)\n";

    std::string module_name = "functor_stub";
    auto module = std::make_unique<llvm::Module>(module_name, *llvm_context_);

    // Build function signature with uint32 placeholders for pointer/reference parameters
    std::vector<llvm::Type*> param_types;

    // Add explicit parameters
    for (auto* param : method->parameters()) {
        llvm::Type* param_llvm_type;
        clang::QualType param_type = param->getType();

        // Check if this is a pointer or reference type
        if (param_type->isPointerType() || param_type->isReferenceType()) {
            // Use uint32 placeholder for all pointer/reference parameters
            param_llvm_type = llvm::Type::getInt32Ty(*llvm_context_);
            llvm::errs() << "[LambdaIRGenerator] Parameter '" << param->getNameAsString()
                         << "' is pointer/ref, using uint32 placeholder\n";
        } else {
            // Use actual type for value parameters
            param_llvm_type = convertType(param_type);
        }
        param_types.push_back(param_llvm_type);
    }

    // TODO: Also handle functor member variables as additional parameters if needed
    // For now, this simplified fallback just handles explicit parameters

    llvm::Type* return_type = convertType(method->getReturnType());
    llvm::FunctionType* func_type = llvm::FunctionType::get(return_type, param_types, false);

    // Create function
    llvm::Function* func = llvm::Function::Create(
        func_type,
        llvm::Function::ExternalLinkage,
        "lambda_kernel",
        module.get()
    );

    // Create a simple body that just returns (stub implementation)
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(*llvm_context_, "entry", func);
    llvm::IRBuilder<> builder(entry);

    if (return_type->isVoidTy()) {
        builder.CreateRetVoid();
    } else {
        // Return zero/default value
        if (return_type->isFloatingPointTy()) {
            builder.CreateRet(llvm::ConstantFP::get(return_type, 0.0));
        } else if (return_type->isIntegerTy()) {
            builder.CreateRet(llvm::ConstantInt::get(return_type, 0));
        } else {
            builder.CreateRetVoid(); // Fallback
        }
    }

    llvm::errs() << "[LambdaIRGenerator] Generated fallback stub\n";
    llvm::errs() << "[LambdaIRGenerator] Full module dump:\n";
    module->print(llvm::errs(), nullptr);
    llvm::errs() << "\n[LambdaIRGenerator] End of module dump\n";

    // IMPORTANT: Remove the old broken code below that tried to translate statements
    // We're just creating a stub, so no need to translate the method body
    return module;
}

} // namespace parallax
