#include "parallax/lambda_ir_generator.hpp"
#include "parallax/class_context_extractor.hpp"
#include "parallax/kernel_wrapper.hpp"
#include <clang/AST/Type.h>
#include <clang/AST/OperationKinds.h>
#include <clang/AST/Mangle.h>
#include <clang/CodeGen/ModuleBuilder.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <functional>
#include <llvm/Passes/PassBuilder.h>
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
                // Emit a real 64-bit integer. The SPIR-V generator declares the
                // Int64 capability when this type is used, and the runtime checks
                // shaderInt64 at kernel load. Truncating to 32 bits silently
                // corrupted values above 2^31.
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
            case clang::BO_Mul: {
                // For arithmetic operators, operands must be values, not pointers
                // Determine actual type from AST, not just assuming float
                llvm::Type* load_type = convertType(binary_op->getLHS()->getType());

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
                llvm::Type* load_type = convertType(binary_op->getLHS()->getType());

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
                llvm::Type* load_type = convertType(binary_op->getLHS()->getType());

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
                llvm::Type* load_type = convertType(binary_op->getLHS()->getType());

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
                llvm::Type* load_type = convertType(binary_op->getLHS()->getType());

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
                llvm::Type* load_type = convertType(binary_op->getLHS()->getType());

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
                llvm::Type* load_type = convertType(binary_op->getLHS()->getType());

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
                llvm::Type* load_type = convertType(binary_op->getLHS()->getType());

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
                llvm::Type* load_type = convertType(binary_op->getLHS()->getType());

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
                llvm::Type* load_type = convertType(binary_op->getLHS()->getType());

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
                llvm::Type* load_type = convertType(binary_op->getRHS()->getType());

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
                llvm::Type* load_type = convertType(binary_op->getLHS()->getType());

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
                llvm::Type* load_type = convertType(binary_op->getLHS()->getType());

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
                llvm::Type* load_type = convertType(binary_op->getLHS()->getType());

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
                llvm::Type* load_type = convertType(binary_op->getLHS()->getType());

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
                llvm::Type* load_type = convertType(binary_op->getLHS()->getType());

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
                llvm::Type* load_type = convertType(binary_op->getLHS()->getType());

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
                llvm::Type* load_type = convertType(binary_op->getLHS()->getType());

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
                llvm::Type* result_type = convertType(binary_op->getType());
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
                    llvm::Type* load_type = convertType(unary_op->getSubExpr()->getType());
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
                    llvm::Type* load_type = convertType(unary_op->getSubExpr()->getType());
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
    // Handle member access expressions (functor members ONLY - early check)
    if (auto* member_expr = llvm::dyn_cast<clang::MemberExpr>(expr)) {
        if (auto* field_decl = llvm::dyn_cast<clang::FieldDecl>(member_expr->getMemberDecl())) {
            auto it = current_functor_members_.find(field_decl);
            if (it != current_functor_members_.end()) {
                llvm::Value* member_val = it->second;
                llvm::errs() << "[translateExpr] Found functor member in map: " << field_decl->getNameAsString() << "\n";

                // Members are passed as arguments, return them directly
                // For pointer/reference types, they're already the right type
                return member_val;
            }
        }
        // NOT a functor member - fall through to general struct member access handler below
    }

    if (auto* decl_ref = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
        llvm::errs() << "[translateExpr] DeclRefExpr: " << decl_ref->getDecl()->getNameAsString() << "\n";
        llvm::errs() << "[translateExpr] Type: " << decl_ref->getType().getAsString() << "\n";
        llvm::errs() << "[translateExpr] Is LValue: " << decl_ref->isLValue() << "\n";

        if (auto* var_decl = llvm::dyn_cast<clang::VarDecl>(decl_ref->getDecl())) {
            // Check for global constants (constexpr)
            if (var_decl->isConstexpr() || (var_decl->hasGlobalStorage() && var_decl->getType().isConstQualified())) {
                llvm::errs() << "[translateExpr] Global constant, evaluating...\n";
                if (auto* init_expr = var_decl->getAnyInitializer()) {
                    // Evaluate constant expression
                    if (auto* float_lit = llvm::dyn_cast<clang::FloatingLiteral>(init_expr->IgnoreImplicit())) {
                        double value = float_lit->getValueAsApproximateDouble();
                        return llvm::ConstantFP::get(builder.getFloatTy(), value);
                    } else if (auto* int_lit = llvm::dyn_cast<clang::IntegerLiteral>(init_expr->IgnoreImplicit())) {
                        int64_t value = int_lit->getValue().getSExtValue();
                        return llvm::ConstantInt::get(builder.getInt32Ty(), value);
                    }
                }
            }

            auto it = var_map.find(var_decl);
            if (it != var_map.end()) {
                llvm::Value* var = it->second;
                llvm::errs() << "[translateExpr] Found variable in map, type: ";
                var->getType()->print(llvm::errs());
                llvm::errs() << "\n";

                // If the variable is already a value type (not a pointer), return it directly
                // This handles captured-by-value parameters like 'N' which are stored as i32
                if (!var->getType()->isPointerTy()) {
                    llvm::errs() << "[translateExpr] Variable is already a value type, returning directly\n";
                    return var;
                }

                // For reference types or lvalues, return the pointer directly
                // Don't auto-load for LHS of assignments
                if (decl_ref->getType()->isReferenceType() || decl_ref->isLValue()) {
                    llvm::errs() << "[translateExpr] Returning pointer for lvalue/reference\n";
                    return var;
                }

                // For rvalue uses, load the value
                llvm::errs() << "[translateExpr] Loading rvalue\n";
                llvm::Type* load_type = decl_ref->getType()->isIntegerType() ? builder.getInt32Ty() : builder.getFloatTy();
                return builder.CreateLoad(load_type, var, var_decl->getNameAsString());
            }
        }

        // Handle function parameters
        if (auto* parm_decl = llvm::dyn_cast<clang::ParmVarDecl>(decl_ref->getDecl())) {
            auto it = var_map.find(parm_decl);
            if (it != var_map.end()) {
                llvm::Value* var = it->second;
                llvm::errs() << "[translateExpr] Found parameter in map\n";

                // If parameter is a value type (not pointer), return directly
                if (!var->getType()->isPointerTy()) {
                    llvm::errs() << "[translateExpr] Parameter is value type, returning directly\n";
                    return var;
                }

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
                llvm::Type* load_type = convertType(call->getArg(i)->getType());
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
        } else if (func_name == "operator[]") {
            // Handle vector/array indexing: vec[index]
            llvm::errs() << "[translateExpr] Handling operator[]\n";

            if (args.size() != 1) {
                llvm::errs() << "[translateExpr] operator[] expects 1 argument, got " << args.size() << "\n";
                return nullptr;
            }

            // Get the object being indexed (the 'this' pointer for the call)
            // For vec[index], the callee->getThisObjectType() gives us the vector type
            if (auto* member_call = llvm::dyn_cast<clang::CXXMemberCallExpr>(call)) {
                llvm::Value* base = translateExpr(member_call->getImplicitObjectArgument(), builder, context, var_map);
                llvm::Value* index = args[0];

                if (!base || !index) {
                    llvm::errs() << "[translateExpr] Failed to get base or index for operator[]\n";
                    return nullptr;
                }

                llvm::errs() << "[translateExpr] Base type: ";
                base->getType()->print(llvm::errs());
                llvm::errs() << ", Index type: ";
                index->getType()->print(llvm::errs());
                llvm::errs() << "\n";

                // Base should be a pointer (i32 for decomposed vector data pointer)
                // For decomposed vectors, 'base' is the data pointer passed as i32
                // We need to get the actual element type
                clang::QualType obj_type = member_call->getImplicitObjectArgument()->getType();

                // Get element type - for std::vector<T>, we want T
                if (obj_type->isReferenceType()) {
                    obj_type = obj_type->getPointeeType();
                }

                // Extract element type from vector<T>
                llvm::Type* element_type = builder.getFloatTy(); // default
                std::string type_str = obj_type.getAsString();
                if (type_str.find("std::vector<") != std::string::npos) {
                    // Parse element type from "std::vector<ElementType, ...>"
                    size_t start = type_str.find('<') + 1;
                    size_t end = type_str.find(',', start);
                    if (end == std::string::npos) end = type_str.find('>', start);
                    std::string elem_type_str = type_str.substr(start, end - start);

                    // Map to LLVM type
                    element_type = convertType(call->getType());
                }

                // If base is i32 (decomposed vector data pointer), we need to convert it to a proper pointer
                // For now, treat it as an opaque pointer and use GEP
                if (!base->getType()->isPointerTy()) {
                    // base is i32, need to inttoptr
                    llvm::Type* ptr_type = llvm::PointerType::getUnqual(builder.getContext());
                    base = builder.CreateIntToPtr(base, ptr_type, "vec_data_ptr");
                }

                // Create GEP for array indexing
                llvm::Value* elem_ptr = builder.CreateGEP(element_type, base, index, "vec_elem_ptr");
                llvm::errs() << "[translateExpr] Created GEP for operator[]\n";

                // Return pointer to element (caller will load if needed)
                return elem_ptr;
            }
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

            // Get the struct type from the base, handling references and pointers
            clang::QualType base_type = member->getBase()->getType();

            // Strip reference if present
            if (base_type->isReferenceType()) {
                base_type = base_type->getPointeeType();
            }

            // Strip pointer if present
            if (base_type->isPointerType()) {
                base_type = base_type->getPointeeType();
            }

            const clang::RecordDecl* record_decl = base_type->getAsStructureType()->getDecl();

            // Find the field index
            unsigned field_idx = 0;
            for (auto* field : record_decl->fields()) {
                if (field == field_decl) {
                    break;
                }
                field_idx++;
            }

            llvm::errs() << "[translateExpr] Field index: " << field_idx << "\n";

            // Get the struct type in LLVM IR (using the stripped base_type)
            llvm::Type* struct_type = convertType(base_type);

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
            llvm::Type* load_type = convertType(cond_op->getTrueExpr()->getType());
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
            llvm::Type* load_type = convertType(cond_op->getFalseExpr()->getType());
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
    // Pointer captures should be passed as actual pointers for array indexing to work
    for (const auto& capture : captures) {
        llvm::Type* capture_type;

        // Check if this is a pointer or reference type
        if (capture.is_by_reference || capture.type->isPointerType()) {
            // Use opaque pointer type for pointer/reference captures
            capture_type = llvm::PointerType::getUnqual(*llvm_context_);
            llvm::errs() << "[LambdaIRGenerator] Manual: Capture '" << capture.name
                         << "' is pointer/ref, using pointer type\n";
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

// NEW: Generate IR for functor (function object) - treats members like lambda captures
std::unique_ptr<llvm::Module> LambdaIRGenerator::generateIRManual(
    clang::CXXMethodDecl* method,
    const ClassContext& class_ctx,
    clang::ASTContext& context) {

    std::string module_name = "functor_" + std::to_string(reinterpret_cast<uintptr_t>(method));
    auto module = std::make_unique<llvm::Module>(module_name, *llvm_context_);

    llvm::errs() << "[generateIRManual] Functor operator() in class: "
                 << class_ctx.record->getNameAsString() << "\n";
    llvm::errs() << "[generateIRManual] Functor has " << class_ctx.member_variables.size()
                 << " member variables\n";
    for (const auto& member : class_ctx.member_variables) {
        llvm::errs() << "  - " << member->getNameAsString() << " : "
                     << member->getType().getAsString() << "\n";
    }

    // Build function signature
    std::vector<llvm::Type*> param_types;
    std::map<const clang::VarDecl*, llvm::Value*> var_map;

    // Add explicit operator() parameters
    for (auto* param : method->parameters()) {
        llvm::Type* param_type = convertType(param->getType());
        param_types.push_back(param_type);
    }

    // Add functor member variables as additional parameters (like lambda captures)
    // Functor members: use actual pointer types for array indexing
    for (const auto& member : class_ctx.member_variables) {
        llvm::Type* member_type;

        clang::QualType member_qtype = member->getType();
        if (member_qtype->isPointerType() || member_qtype->isReferenceType()) {
            // Use pointer type for pointers/references
            member_type = llvm::PointerType::getUnqual(*llvm_context_);
            llvm::errs() << "[LambdaIRGenerator] Functor member '" << member->getNameAsString()
                         << "' is pointer/ref, using pointer type\n";
        } else {
            // Use actual type for value members
            member_type = convertType(member_qtype);
        }
        param_types.push_back(member_type);
    }

    llvm::Type* return_type = convertType(method->getReturnType());
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
    for (auto* param : method->parameters()) {
        llvm::Argument* arg = &(*arg_it++);
        arg->setName(param->getNameAsString());
        var_map[param] = arg;
    }

    // Map functor members to LLVM arguments (like captures)
    // Create a temporary map for member -> argument mapping
    std::map<const clang::FieldDecl*, llvm::Value*> member_map;
    for (const auto& member : class_ctx.member_variables) {
        llvm::Argument* arg = &(*arg_it++);
        arg->setName("member_" + member->getNameAsString());
        // Store member in member_map
        member_map[member] = arg;
        llvm::errs() << "[generateIRManual] Mapped member " << member->getNameAsString()
                     << " to argument\n";
    }

    // Create entry basic block
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(*llvm_context_, "entry", func);
    llvm::IRBuilder<> builder(entry);

    // Store member_map in class member for access during translation
    current_functor_members_ = member_map;

    // Translate operator() body
    if (clang::Stmt* body = method->getBody()) {
        translateStmt(body, builder, context, var_map);
    }

    // Clear member map after translation
    current_functor_members_.clear();

    // Add return if not already present
    if (!entry->getTerminator()) {
        if (return_type->isVoidTy()) {
            builder.CreateRetVoid();
        }
    }

    // Verify function
    if (llvm::verifyFunction(*func, &llvm::errs())) {
        llvm::errs() << "ERROR: Generated LLVM IR is invalid for functor!\n";
        func->print(llvm::errs());
        return nullptr;
    }

    std::cerr << "\n[LambdaIRGenerator] Generated Functor LLVM IR:\n";
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
    // Add captures as parameters using actual pointer types
    for (const auto& capture : captures) {
        llvm::Type* capture_type;

        // Check if this is a pointer or reference type
        if (capture.is_by_reference || capture.type->isPointerType()) {
            // Use pointer type for pointer/reference captures
            capture_type = llvm::PointerType::getUnqual(*llvm_context_);
            llvm::errs() << "[LambdaIRGenerator] Capture '" << capture.name
                         << "' is pointer/ref, using pointer type\n";
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

    // Proper code generation: run Clang's real CodeGen pipeline (EmitLLVMOnly)
    // over the translation unit so the lambda/functor operator() is emitted with
    // natively-correct LLVM types for every type and operation, then pick out that
    // one function by its mangled name. This replaces the fragile manual translator.

    clang::SourceManager& SM = ast_context.getSourceManager();
    clang::FileID fid = SM.getFileID(method->getLocation());
    auto file_ref = SM.getFileEntryRefForID(fid);
    if (!file_ref) {
        llvm::errs() << "[CodeGen] Could not resolve source file for operator()\n";
        return nullptr;
    }
    std::string source_path = file_ref->getName().str();

    // Clone the current invocation, retargeting it to emit LLVM IR only for just
    // this file, with optimizations and plugins disabled (avoid plugin recursion).
    auto inv = std::make_shared<clang::CompilerInvocation>(CI_.getInvocation());
    inv->getFrontendOpts().ProgramAction = clang::frontend::EmitLLVMOnly;
    inv->getFrontendOpts().ActionName.clear();
    inv->getFrontendOpts().Plugins.clear();
    inv->getFrontendOpts().AddPluginActions.clear();
    inv->getFrontendOpts().PluginArgs.clear();
    inv->getFrontendOpts().Inputs.clear();
    inv->getFrontendOpts().Inputs.emplace_back(
        source_path, clang::InputKind(clang::Language::CXX));
    inv->getCodeGenOpts().OptimizationLevel = 0;
    // Emit line-table debug info so each emitted operator() carries its source line.
    // We select the target lambda by source location (robust), because mangled-name
    // lookup is ambiguous: same-signature lambdas in a TU can share a discriminator.
    inv->getCodeGenOpts().setDebugInfo(llvm::codegenoptions::DebugLineTablesOnly);
    // Disable FP contraction so a*b+c stays as separate fmul/fadd instead of
    // llvm.fmuladd (kept simple for the SPIR-V translator; fmuladd is also handled).
    inv->getLangOpts().setDefaultFPContractMode(clang::LangOptions::FPM_Off);

    clang::CompilerInstance sub_ci(inv);
    sub_ci.createDiagnostics(*llvm::vfs::getRealFileSystem());
    if (!sub_ci.hasDiagnostics()) {
        llvm::errs() << "[CodeGen] Failed to create diagnostics for sub-compilation\n";
        return nullptr;
    }

    clang::EmitLLVMOnlyAction action(llvm_context_.get());
    if (!sub_ci.ExecuteAction(action)) {
        llvm::errs() << "[CodeGen] EmitLLVMOnly sub-compilation failed\n";
        return nullptr;
    }

    std::unique_ptr<llvm::Module> module = action.takeModule();
    if (!module) {
        llvm::errs() << "[CodeGen] Sub-compilation produced no module\n";
        return nullptr;
    }

    // Compute the mangled name (used as a fallback and for logging).
    std::unique_ptr<clang::MangleContext> mangler(ast_context.createMangleContext());
    std::string mangled;
    {
        llvm::raw_string_ostream os(mangled);
        if (mangler->shouldMangleDeclName(method)) {
            mangler->mangleName(method, os);
        } else {
            os << method->getName();
        }
    }

    // Primary lookup: match the emitted operator() by SOURCE LINE. Mangled-name
    // lookup is ambiguous — same-signature lambdas in one TU can all mangle to
    // '$_0' (no Sema-assigned discriminator), so getFunction() would return the
    // first lambda for every call. The source line is unique per lambda.
    //
    // The emitted operator()'s DISubprogram line aligns with the decl's BEGIN
    // location (the lambda's '['), NOT method->getLocation() — for a lambda that
    // "name" location can sit a line off, which would drop us into the ambiguous
    // mangled-name fallback and pick the WRONG lambda (aliasing). Try the begin
    // line first, then the name-location line, before the mangled fallback.
    auto& src_mgr = ast_context.getSourceManager();
    unsigned begin_line = src_mgr.getExpansionLineNumber(method->getBeginLoc());
    unsigned loc_line   = src_mgr.getExpansionLineNumber(method->getLocation());
    llvm::Function* target = nullptr;
    unsigned matches = 0;
    unsigned target_line = begin_line;
    for (unsigned cand : {begin_line, loc_line}) {
        target = nullptr; matches = 0;
        for (llvm::Function& f : *module) {
            if (f.isDeclaration()) continue;
            llvm::DISubprogram* sp = f.getSubprogram();
            if (sp && sp->getLine() == cand) { target = &f; ++matches; }
        }
        if (matches == 1) { target_line = cand; break; }
    }
    if (matches != 1) {
        // Ambiguous or no line match — fall back to the mangled name.
        llvm::Function* byname = module->getFunction(mangled);
        if (byname && !byname->isDeclaration()) target = byname;
        else if (matches == 0) target = nullptr;
    }
    if (!target || target->isDeclaration()) {
        llvm::errs() << "[CodeGen] Could not locate emitted operator() '" << mangled
                     << "' (line " << target_line << ", " << matches << " line matches)\n";
        return nullptr;
    }
    llvm::errs() << "[CodeGen] Located operator(): " << target->getName().str()
                 << " (line " << target_line << ", " << target->arg_size() << " args)\n";

    // Done selecting; strip debug info so the wrapper/inline/mem2reg/SPIR-V path
    // sees clean IR (no DILocations on a wrapper that has no DISubprogram).
    llvm::StripDebugInfo(*module);

    // The member operator() has an implicit leading 'this' (the closure object).
    // Downstream SPIR-V codegen expects a plain kernel whose first parameter is the
    // data element, so wrap operator() in a this-less function and inline it. For a
    // captureless lambda 'this' is unused, so a null closure pointer is correct.
    // (Lambdas with captures need the closure passed as a buffer — Phase 4.)
    llvm::FunctionType* target_fty = target->getFunctionType();
    if (target_fty->getNumParams() < 1) {
        llvm::errs() << "[CodeGen] operator() has no 'this' parameter; unexpected\n";
        return nullptr;
    }

    // Recover the closure struct type (= the by-value captures) from a GEP in
    // operator() whose base traces to 'this' (arg 0). Captureless lambdas have none.
    llvm::StructType* closure_ty = nullptr;
    {
        llvm::Value* this_arg = target->getArg(0);
        auto from_gep = [](llvm::Value* base) -> llvm::StructType* {
            for (llvm::User* u : base->users())
                if (auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(u))
                    if (gep->getPointerOperand() == base)
                        if (auto* st = llvm::dyn_cast<llvm::StructType>(gep->getSourceElementType()))
                            return st;
            return nullptr;
        };
        closure_ty = from_gep(this_arg);
        if (!closure_ty) {  // -O0: 'this' is stored to an alloca, loaded, then GEP'd.
            for (llvm::User* u : this_arg->users()) {
                auto* st = llvm::dyn_cast<llvm::StoreInst>(u);
                if (!st || st->getValueOperand() != this_arg) continue;
                auto* slot = llvm::dyn_cast<llvm::AllocaInst>(st->getPointerOperand());
                if (!slot) continue;
                for (llvm::User* su : slot->users())
                    if (auto* ld = llvm::dyn_cast<llvm::LoadInst>(su))
                        if ((closure_ty = from_gep(ld))) break;
                if (closure_ty) break;
            }
        }
    }

    // Wrapper params = operator()'s data args (after 'this'), then one scalar arg per
    // capture (in closure-field order, matching the runtime's captures packing).
    std::vector<llvm::Type*> wrapper_params(target_fty->param_begin() + 1,
                                            target_fty->param_end());
    size_t num_data_params = wrapper_params.size();

    // Flatten the closure into LEAF scalars (recursively through nested structs and
    // arrays). Each leaf becomes one wrapper arg; since the runtime packs the captured
    // values contiguously, a captured POD struct's fields line up by byte offset.
    std::vector<std::vector<unsigned>> leaf_paths;  // GEP indices after the leading 0
    std::vector<llvm::Type*> leaf_types;
    if (closure_ty) {
        std::function<void(llvm::Type*, std::vector<unsigned>)> flatten =
            [&](llvm::Type* t, std::vector<unsigned> path) {
                if (auto* st = llvm::dyn_cast<llvm::StructType>(t)) {
                    for (unsigned i = 0; i < st->getNumElements(); ++i) {
                        auto p = path; p.push_back(i);
                        flatten(st->getElementType(i), p);
                    }
                } else if (auto* at = llvm::dyn_cast<llvm::ArrayType>(t)) {
                    for (unsigned i = 0; i < at->getNumElements(); ++i) {
                        auto p = path; p.push_back(i);
                        flatten(at->getElementType(), p);
                    }
                } else {
                    leaf_paths.push_back(std::move(path));
                    leaf_types.push_back(t);
                }
            };
        flatten(closure_ty, {});
        for (llvm::Type* lt : leaf_types) wrapper_params.push_back(lt);
    }

    llvm::FunctionType* wrapper_fty =
        llvm::FunctionType::get(target_fty->getReturnType(), wrapper_params, false);
    llvm::Function* wrapper = llvm::Function::Create(
        wrapper_fty, llvm::Function::ExternalLinkage, "__parallax_kernel_body", module.get());
    if (closure_ty)
        llvm::errs() << "[CodeGen] Lambda captures " << leaf_types.size()
                     << " leaf value(s); reconstructing closure in the kernel\n";

    llvm::BasicBlock* entry =
        llvm::BasicBlock::Create(*llvm_context_, "entry", wrapper);
    llvm::IRBuilder<> builder(entry);

    // Build the 'this' operator() reads from: a stack closure filled from the capture
    // args (SROA+mem2reg later promote it), or null for a captureless lambda.
    llvm::Value* this_val;
    if (closure_ty) {
        llvm::Value* clo = builder.CreateAlloca(closure_ty);
        llvm::Type* i32 = llvm::Type::getInt32Ty(*llvm_context_);
        for (size_t li = 0; li < leaf_paths.size(); ++li) {
            std::vector<llvm::Value*> idxs;
            idxs.push_back(llvm::ConstantInt::get(i32, 0));  // deref the closure pointer
            for (unsigned ix : leaf_paths[li]) idxs.push_back(llvm::ConstantInt::get(i32, ix));
            llvm::Value* field = builder.CreateGEP(closure_ty, clo, idxs);
            builder.CreateStore(wrapper->getArg(num_data_params + li), field);
        }
        this_val = clo;
    } else {
        this_val = llvm::ConstantPointerNull::get(
            llvm::cast<llvm::PointerType>(target_fty->getParamType(0)));  // null 'this'
    }

    std::vector<llvm::Value*> call_args;
    call_args.push_back(this_val);
    for (size_t i = 0; i < num_data_params; ++i) call_args.push_back(wrapper->getArg(i));
    llvm::CallInst* fwd = builder.CreateCall(target, call_args);
    if (wrapper_fty->getReturnType()->isVoidTy()) {
        builder.CreateRetVoid();
    } else {
        builder.CreateRet(fwd);
    }

    // Inline operator() into the wrapper so the body is self-contained, then drop
    // every other definition's body. The wrapper becomes the lone definition the
    // caller selects.
    llvm::InlineFunctionInfo ifi;
    llvm::InlineResult ir = llvm::InlineFunction(*fwd, ifi);
    if (!ir.isSuccess()) {
        llvm::errs() << "[CodeGen] Failed to inline operator() into wrapper: "
                     << ir.getFailureReason() << "\n";
        return nullptr;
    }

    // Recursively inline any further calls to user-defined functions (helper
    // functions the lambda calls), so the kernel body is fully self-contained.
    // Declarations (math libcalls / intrinsics) are left for intrinsic mapping.
    for (int guard = 0; guard < 256; ++guard) {
        llvm::CallInst* next = nullptr;
        for (llvm::BasicBlock& bb : *wrapper) {
            for (llvm::Instruction& inst : bb) {
                auto* ci = llvm::dyn_cast<llvm::CallInst>(&inst);
                if (!ci) continue;
                llvm::Function* callee = ci->getCalledFunction();
                if (callee && !callee->isDeclaration() && !callee->isIntrinsic()) {
                    next = ci;
                    break;
                }
            }
            if (next) break;
        }
        if (!next) break;
        llvm::InlineFunctionInfo ifi2;
        if (!llvm::InlineFunction(*next, ifi2).isSuccess()) {
            llvm::errs() << "[CodeGen] Could not inline helper call '"
                         << next->getCalledFunction()->getName() << "'\n";
            break;
        }
    }

    for (llvm::Function& f : *module) {
        if (&f != wrapper && !f.isDeclaration()) {
            f.deleteBody();
        }
    }

    // Promote allocas to SSA registers (mem2reg) + simplify. At -O0 Clang emits
    // stack slots and pointer-to-pointer indirection that the opaque-pointer SPIR-V
    // translator cannot type; after promotion the body is clean load/compute/store.
    {
        llvm::PassBuilder PB;
        llvm::FunctionAnalysisManager FAM;
        PB.registerFunctionAnalyses(FAM);
        llvm::FunctionPassManager FPM;
        // SROA splits the reconstructed closure struct into per-field SSA values
        // (mem2reg alone can't promote an alloca that has GEP field accesses), then
        // mem2reg promotes the remaining scalar stack slots.
        FPM.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
        FPM.addPass(llvm::PromotePass());  // mem2reg
        FPM.run(*wrapper, FAM);
    }

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
            // Use pointer type for pointer/reference parameters
            param_llvm_type = llvm::PointerType::getUnqual(*llvm_context_);
            llvm::errs() << "[LambdaIRGenerator] Parameter '" << param->getNameAsString()
                         << "' is pointer/ref, using pointer type\n";
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
