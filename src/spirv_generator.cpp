#include "parallax/spirv_generator.hpp"
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <unordered_map>
#include <stdexcept>

namespace parallax {

// SPIR-V opcodes
enum class SPIRVOp : uint32_t {
    OpNop = 0,
    OpSource = 3,
    OpName = 5,
    OpMemberName = 6,
    OpString = 7,
    OpLine = 8,
    OpExtension = 10,
    OpExtInstImport = 11,
    OpExtInst = 12,
    OpMemoryModel = 14,
    OpEntryPoint = 15,
    OpExecutionMode = 16,
    OpCapability = 17,
    OpTypeVoid = 19,
    OpTypeBool = 20,
    OpTypeInt = 21,
    OpTypeFloat = 22,
    OpTypeVector = 23,
    OpTypeMatrix = 24,
    OpTypeImage = 25,
    OpTypeSampler = 26,
    OpTypeSampledImage = 27,
    OpTypeArray = 28,
    OpTypeRuntimeArray = 29,
    OpTypeStruct = 30,
    OpTypePointer = 32,
    OpTypeFunction = 33,
    OpConstantTrue = 41,
    OpConstantFalse = 42,
    OpConstant = 43,
    OpConstantComposite = 44,
    OpFunction = 54,
    OpFunctionParameter = 55,
    OpFunctionEnd = 56,
    OpFunctionCall = 57,
    OpVariable = 59,
    OpLoad = 61,
    OpStore = 62,
    OpAccessChain = 65,
    OpDecorate = 71,
    OpMemberDecorate = 72,
    OpVectorExtractDynamic = 77,
    OpVectorInsertDynamic = 78,
    OpVectorShuffle = 79,
    OpCompositeConstruct = 80,
    OpCompositeExtract = 81,
    OpCompositeInsert = 82,
    OpCopyObject = 83,
    OpTranspose = 84,
    OpSNegate = 126,
    OpFNegate = 127,
    OpIAdd = 128,
    OpFAdd = 129,
    OpISub = 130,
    OpFSub = 131,
    OpIMul = 132,
    OpFMul = 133,
    OpUDiv = 134,
    OpSDiv = 135,
    OpFDiv = 136,
    OpUMod = 137,
    OpSRem = 138,
    OpSMod = 139,
    OpFRem = 140,
    OpFMod = 141,
    OpVectorTimesScalar = 142,
    OpMatrixTimesScalar = 143,
    OpVectorTimesMatrix = 144,
    OpMatrixTimesVector = 145,
    OpMatrixTimesMatrix = 146,
    OpLogicalEqual = 164,
    OpLogicalNotEqual = 165,
    OpLogicalOr = 166,
    OpLogicalAnd = 167,
    OpLogicalNot = 168,
    OpSelect = 169,
    OpIEqual = 170,
    OpINotEqual = 171,
    OpUGreaterThan = 172,
    OpSGreaterThan = 173,
    OpUGreaterThanEqual = 174,
    OpSGreaterThanEqual = 175,
    OpULessThan = 176,
    OpSLessThan = 177,
    OpULessThanEqual = 178,
    OpSLessThanEqual = 179,
    OpFOrdEqual = 180,
    OpFUnordEqual = 181,
    OpFOrdNotEqual = 182,
    OpFUnordNotEqual = 183,
    OpFOrdLessThan = 184,
    OpFUnordLessThan = 185,
    OpFOrdGreaterThan = 186,
    OpFUnordGreaterThan = 187,
    OpFOrdLessThanEqual = 188,
    OpFUnordLessThanEqual = 189,
    OpFOrdGreaterThanEqual = 190,
    OpFUnordGreaterThanEqual = 191,
    OpShiftRightLogical = 194,
    OpShiftRightArithmetic = 195,
    OpShiftLeftLogical = 196,
    OpBitwiseOr = 197,
    OpBitwiseXor = 198,
    OpBitwiseAnd = 199,
    OpNot = 200,
    OpBitFieldInsert = 201,
    OpBitFieldSExtract = 202,
    OpBitFieldUExtract = 203,
    OpBitReverse = 204,
    OpBitCount = 205,
    OpPhi = 245,
    OpLabel = 248,
    OpBranch = 249,
    OpBranchConditional = 250,
    OpSwitch = 251,
    OpReturn = 253,
    OpReturnValue = 254,
};

class SPIRVBuilder {
public:
    SPIRVBuilder() : next_id_(1) {}
    
    uint32_t get_next_id() { return next_id_++; }
    
    void emit_word(uint32_t word) {
        spirv_.push_back(word);
    }
    
    void emit_op(SPIRVOp op, const std::vector<uint32_t>& operands) {
        uint32_t word_count = 1 + operands.size();
        emit_word((word_count << 16) | static_cast<uint32_t>(op));
        for (uint32_t operand : operands) {
            emit_word(operand);
        }
    }
    
    void emit_string(const std::string& str) {
        const char* data = str.c_str();
        size_t len = str.length();
        for (size_t i = 0; i <= len; i += 4) {
            uint32_t word = 0;
            for (size_t j = 0; j < 4 && (i + j) <= len; j++) {
                word |= static_cast<uint32_t>(data[i + j]) << (j * 8);
            }
            emit_word(word);
        }
    }
    
    std::vector<uint32_t> get_spirv() const { return spirv_; }
    
private:
    std::vector<uint32_t> spirv_;
    uint32_t next_id_;
};

SPIRVGenerator::SPIRVGenerator()
    : vulkan_major_(1), vulkan_minor_(3) {}

SPIRVGenerator::~SPIRVGenerator() = default;

void SPIRVGenerator::set_target_vulkan_version(uint32_t major, uint32_t minor) {
    vulkan_major_ = major;
    vulkan_minor_ = minor;
}

std::vector<uint32_t> SPIRVGenerator::generate(llvm::Module* module) {
    SPIRVBuilder builder;
    
    // Emit header
    builder.emit_word(0x07230203); // Magic number
    builder.emit_word(0x00010600); // Version 1.6
    builder.emit_word(0x000d000b); // Generator
    uint32_t bound_id = builder.get_next_id();
    builder.emit_word(bound_id);   // Bound (will update)
    builder.emit_word(0x00000000); // Schema
    
    // Emit capabilities
    builder.emit_op(SPIRVOp::OpCapability, {1}); // Shader
    
    // Emit memory model
    builder.emit_op(SPIRVOp::OpMemoryModel, {0, 1}); // Logical GLSL450
    
    // Find entry point
    for (auto& func : module->functions()) {
        if (!func.isDeclaration()) {
            uint32_t func_id = builder.get_next_id();
            
            // Emit entry point
            std::vector<uint32_t> entry_operands = {5, func_id}; // GLCompute
            builder.emit_op(SPIRVOp::OpEntryPoint, entry_operands);
            builder.emit_string(func.getName().str());
            
            // Emit execution mode
            builder.emit_op(SPIRVOp::OpExecutionMode, {func_id, 17, 256, 1, 1}); // LocalSize
            
            // Translate function
            translate_function(builder, &func, func_id);
            break;
        }
    }
    
    return builder.get_spirv();
}

void SPIRVGenerator::translate_function(SPIRVBuilder& builder, llvm::Function* func, uint32_t func_id) {
    std::unordered_map<llvm::Value*, uint32_t> value_map;
    
    // Emit type declarations
    uint32_t void_type = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpTypeVoid, {void_type});
    
    uint32_t func_type = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpTypeFunction, {func_type, void_type});
    
    // Emit function
    builder.emit_op(SPIRVOp::OpFunction, {void_type, func_id, 0, func_type});
    
    // Handle arguments
    for (auto& arg : func->args()) {
        uint32_t arg_id = builder.get_next_id();
        uint32_t arg_type_id = get_type_id(builder, arg.getType());
        builder.emit_op(SPIRVOp::OpFunctionParameter, {arg_type_id, arg_id});
        value_map[&arg] = arg_id;
    }
    
    // Translate basic blocks
    for (auto& bb : *func) {
        // ... (label handling)
        uint32_t label_id = builder.get_next_id();
        value_map[&bb] = label_id;
        builder.emit_op(SPIRVOp::OpLabel, {label_id});
        
        // Translate instructions
        for (auto& inst : bb) {
            translate_instruction(builder, &inst, value_map);
        }
    }
    
    builder.emit_op(SPIRVOp::OpFunctionEnd, {});
}

void SPIRVGenerator::translate_instruction(SPIRVBuilder& builder, llvm::Instruction* inst,
                                           std::unordered_map<llvm::Value*, uint32_t>& value_map) {
    uint32_t result_id = builder.get_next_id();
    value_map[inst] = result_id;
    
    switch (inst->getOpcode()) {
        case llvm::Instruction::Add:
            if (inst->getType()->isIntegerTy()) {
                uint32_t op1 = value_map[inst->getOperand(0)];
                uint32_t op2 = value_map[inst->getOperand(1)];
                builder.emit_op(SPIRVOp::OpIAdd, {get_type_id(builder, inst->getType()), result_id, op1, op2});
            } else {
                uint32_t op1 = value_map[inst->getOperand(0)];
                uint32_t op2 = value_map[inst->getOperand(1)];
                builder.emit_op(SPIRVOp::OpFAdd, {get_type_id(builder, inst->getType()), result_id, op1, op2});
            }
            break;
            
        case llvm::Instruction::Sub:
            if (inst->getType()->isIntegerTy()) {
                uint32_t op1 = value_map[inst->getOperand(0)];
                uint32_t op2 = value_map[inst->getOperand(1)];
                builder.emit_op(SPIRVOp::OpISub, {get_type_id(builder, inst->getType()), result_id, op1, op2});
            } else {
                uint32_t op1 = value_map[inst->getOperand(0)];
                uint32_t op2 = value_map[inst->getOperand(1)];
                builder.emit_op(SPIRVOp::OpFSub, {get_type_id(builder, inst->getType()), result_id, op1, op2});
            }
            break;
            
        case llvm::Instruction::Mul:
            if (inst->getType()->isIntegerTy()) {
                uint32_t op1 = value_map[inst->getOperand(0)];
                uint32_t op2 = value_map[inst->getOperand(1)];
                builder.emit_op(SPIRVOp::OpIMul, {get_type_id(builder, inst->getType()), result_id, op1, op2});
            } else {
                uint32_t op1 = value_map[inst->getOperand(0)];
                uint32_t op2 = value_map[inst->getOperand(1)];
                builder.emit_op(SPIRVOp::OpFMul, {get_type_id(builder, inst->getType()), result_id, op1, op2});
            }
            break;
            
        case llvm::Instruction::FDiv: {
            uint32_t op1 = value_map[inst->getOperand(0)];
            uint32_t op2 = value_map[inst->getOperand(1)];
            builder.emit_op(SPIRVOp::OpFDiv, {get_type_id(builder, inst->getType()), result_id, op1, op2});
            break;
        }
        
        case llvm::Instruction::Load: {
            uint32_t ptr = value_map[inst->getOperand(0)];
            builder.emit_op(SPIRVOp::OpLoad, {get_type_id(builder, inst->getType()), result_id, ptr});
            break;
        }
        
        case llvm::Instruction::Store: {
            uint32_t value = value_map[inst->getOperand(0)];
            uint32_t ptr = value_map[inst->getOperand(1)];
            builder.emit_op(SPIRVOp::OpStore, {ptr, value});
            break;
        }
        
        case llvm::Instruction::Ret:
            builder.emit_op(SPIRVOp::OpReturn, {});
            break;
            
        case llvm::Instruction::Br: {
            auto* br = llvm::cast<llvm::BranchInst>(inst);
            if (br->isUnconditional()) {
                uint32_t target = value_map[br->getSuccessor(0)];
                builder.emit_op(SPIRVOp::OpBranch, {target});
            } else {
                uint32_t cond = value_map[br->getCondition()];
                uint32_t true_label = value_map[br->getSuccessor(0)];
                uint32_t false_label = value_map[br->getSuccessor(1)];
                builder.emit_op(SPIRVOp::OpBranchConditional, {cond, true_label, false_label});
            }
            break;
        }
        
        case llvm::Instruction::ICmp: {
            auto* cmp = llvm::cast<llvm::ICmpInst>(inst);
            uint32_t op1 = value_map[cmp->getOperand(0)];
            uint32_t op2 = value_map[cmp->getOperand(1)];
            
            SPIRVOp op;
            switch (cmp->getPredicate()) {
                case llvm::CmpInst::ICMP_EQ:  op = SPIRVOp::OpIEqual; break;
                case llvm::CmpInst::ICMP_NE:  op = SPIRVOp::OpINotEqual; break;
                case llvm::CmpInst::ICMP_UGT: op = SPIRVOp::OpUGreaterThan; break;
                case llvm::CmpInst::ICMP_UGE: op = SPIRVOp::OpUGreaterThanEqual; break;
                case llvm::CmpInst::ICMP_ULT: op = SPIRVOp::OpULessThan; break;
                case llvm::CmpInst::ICMP_ULE: op = SPIRVOp::OpULessThanEqual; break;
                case llvm::CmpInst::ICMP_SGT: op = SPIRVOp::OpSGreaterThan; break;
                case llvm::CmpInst::ICMP_SGE: op = SPIRVOp::OpSGreaterThanEqual; break;
                case llvm::CmpInst::ICMP_SLT: op = SPIRVOp::OpSLessThan; break;
                case llvm::CmpInst::ICMP_SLE: op = SPIRVOp::OpSLessThanEqual; break;
                default: op = SPIRVOp::OpIEqual;
            }
            
            builder.emit_op(op, {get_type_id(builder, inst->getType()), result_id, op1, op2});
            break;
        }
        
        case llvm::Instruction::FCmp: {
            auto* cmp = llvm::cast<llvm::FCmpInst>(inst);
            uint32_t op1 = value_map[cmp->getOperand(0)];
            uint32_t op2 = value_map[cmp->getOperand(1)];
            
            SPIRVOp op;
            switch (cmp->getPredicate()) {
                case llvm::CmpInst::FCMP_OEQ: op = SPIRVOp::OpFOrdEqual; break;
                case llvm::CmpInst::FCMP_ONE: op = SPIRVOp::OpFOrdNotEqual; break;
                case llvm::CmpInst::FCMP_OGT: op = SPIRVOp::OpFOrdGreaterThan; break;
                case llvm::CmpInst::FCMP_OGE: op = SPIRVOp::OpFOrdGreaterThanEqual; break;
                case llvm::CmpInst::FCMP_OLT: op = SPIRVOp::OpFOrdLessThan; break;
                case llvm::CmpInst::FCMP_OLE: op = SPIRVOp::OpFOrdLessThanEqual; break;
                default: op = SPIRVOp::OpFOrdEqual;
            }
            
            builder.emit_op(op, {get_type_id(builder, inst->getType()), result_id, op1, op2});
            break;
        }
        
        default:
            // Handle other instructions as needed
            break;
    }
}

uint32_t SPIRVGenerator::get_type_id(SPIRVBuilder& builder, llvm::Type* type) {
    if (type_cache_.count(type)) {
        return type_cache_[type];
    }
    
    uint32_t type_id = builder.get_next_id();
    
    if (type->isVoidTy()) {
        builder.emit_op(SPIRVOp::OpTypeVoid, {type_id});
    } else if (type->isIntegerTy(32)) {
        builder.emit_op(SPIRVOp::OpTypeInt, {type_id, 32, 0}); // 0 = unsigned? No, 0=unsigned, 1=signed usually. Wait, OpTypeInt width signedness. signedness: 0 indicates unsigned handling, 1 indicates signed handling
        // LLVM i32 is signless. We default to 0 (unsigned) or 1 (signed)?
        // For general arithmetic, 0 is safer? Or 1?
        // Let's use 0 (unsigned) for now as typical for bitwise, but Add can be standard.
        // Wait, OpIAdd works on both.
        // Let's use 0.
    } else if (type->isFloatTy()) {
        builder.emit_op(SPIRVOp::OpTypeFloat, {type_id, 32});
    } else if (type->isDoubleTy()) {
        builder.emit_op(SPIRVOp::OpTypeFloat, {type_id, 64});
    } else if (type->isPointerTy()) {
        // Assume default storage class (Function=7) or Uniform=2 or generic?
        // LLVM pointers are generic. In SPIR-V we need storage class.
        // For kernel args, we likely need CrossWorkgroup(5) or Uniform(2).
        // For local vars, Function(7).
        // Default to Function(7) for now? Or CrossWorkgroup(5) for buffer pointers?
        // This is tricky. Let's assume Function(7) for internal pointers, but for Arguments we might need explicit handling.
        // Let's use CrossWorkgroup(5) for now as it's generic global memory.
        
        uint32_t element_type_id;
        // LLVM 15+ opaque pointers don't have element type.
        // We assume float* for now if opaque?
        // Or we rely on typed pointers if LLVM version is older.
        // LLVM 21 is Opaque.
        // We assume 'float' element type as default for this MVP.
        // TODO: Handle types properly.
        llvm::Type* float_ty = llvm::Type::getFloatTy(type->getContext());
        element_type_id = get_type_id(builder, float_ty);
        
        builder.emit_op(SPIRVOp::OpTypePointer, {type_id, 5 /* CrossWorkgroup */, element_type_id});
    } else {
        // Fallback for unknown
        builder.emit_op(SPIRVOp::OpTypeInt, {type_id, 32, 0});
    }
    
    type_cache_[type] = type_id;
    return type_id;
}

std::vector<uint32_t> SPIRVGenerator::generate_from_lambda(
    llvm::Function* lambda_func,
    const std::vector<std::string>& param_types) {
    
    return generate(lambda_func->getParent());
}

void SPIRVGenerator::emit_header(std::vector<uint32_t>& spirv) {
    spirv.push_back(0x07230203);
    spirv.push_back(0x00010600);
    spirv.push_back(0x000d000b);
    spirv.push_back(0x00000100);
    spirv.push_back(0x00000000);
}

void SPIRVGenerator::emit_capabilities(std::vector<uint32_t>& spirv) {
    spirv.push_back(0x00020011);
    spirv.push_back(0x00000001);
}

void SPIRVGenerator::emit_extensions(std::vector<uint32_t>& spirv) {}
void SPIRVGenerator::emit_memory_model(std::vector<uint32_t>& spirv) {
    spirv.push_back(0x0003000e);
    spirv.push_back(0x00000000);
    spirv.push_back(0x00000001);
}

void SPIRVGenerator::emit_entry_point(std::vector<uint32_t>& spirv, const std::string& name) {}
void SPIRVGenerator::emit_execution_mode(std::vector<uint32_t>& spirv) {}
void SPIRVGenerator::emit_decorations(std::vector<uint32_t>& spirv) {}
void SPIRVGenerator::emit_types(std::vector<uint32_t>& spirv) {}
void SPIRVGenerator::emit_function(std::vector<uint32_t>& spirv, llvm::Function* func) {}

} // namespace parallax
