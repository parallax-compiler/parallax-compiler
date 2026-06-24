#include "parallax/spirv_generator.hpp"
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <iostream>
#include <unordered_map>
#include <set>
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
    OpBitcast = 124,
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
    OpSelectionMerge = 247,
    OpLabel = 248,
    OpBranch = 249,
    OpBranchConditional = 250,
    OpSwitch = 251,
    OpReturn = 253,
    OpReturnValue = 254,
};

class SPIRVBuilder {
public:
    enum class Section {
        Header,
        Capabilities, // OpCapability only (must precede everything else)
        Preamble,     // Extensions, ExtInstImport, MemoryModel
        EntryPoints,
        Decorations,
        Types,        // Types, Constants, Global Variables
        Code
    };

    SPIRVBuilder() : next_id_(1), current_section_(Section::Code) {}
    
    uint32_t get_next_id() { return next_id_++; }
    
    void set_section(Section section) { current_section_ = section; }
    Section get_current_section() const { return current_section_; }
    
    void emit_word(uint32_t word) {
        switch (current_section_) {
            case Section::Header: header_.push_back(word); break;
            case Section::Capabilities: capabilities_.push_back(word); break;
            case Section::Preamble: preamble_.push_back(word); break;
            case Section::EntryPoints: entry_points_.push_back(word); break;
            case Section::Decorations: decorations_.push_back(word); break;
            case Section::Types: types_.push_back(word); break;
            case Section::Code: code_.push_back(word); break;
        }
    }
    
    void emit_op(SPIRVOp op, const std::vector<uint32_t>& operands) {
        std::cerr << "[SPIRVBuilder] section " << (int)current_section_ << " emit_op " << (uint32_t)op << " operands: ";
        for(auto o : operands) std::cerr << o << " "; std::cerr << std::endl;
        
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
    
    std::vector<uint32_t> get_spirv() const {
        std::vector<uint32_t> combined;
        combined.insert(combined.end(), header_.begin(), header_.end());
        combined.insert(combined.end(), capabilities_.begin(), capabilities_.end());
        combined.insert(combined.end(), preamble_.begin(), preamble_.end());
        combined.insert(combined.end(), entry_points_.begin(), entry_points_.end());
        combined.insert(combined.end(), decorations_.begin(), decorations_.end());
        combined.insert(combined.end(), types_.begin(), types_.end());
        combined.insert(combined.end(), code_.begin(), code_.end());
        return combined;
    }
    
    // Explicit access for header generation
    std::vector<uint32_t>& get_header() { return header_; }
    
private:
    uint32_t next_id_;
    Section current_section_;
    std::vector<uint32_t> header_;
    std::vector<uint32_t> capabilities_;
    std::vector<uint32_t> preamble_;
    std::vector<uint32_t> entry_points_;
    std::vector<uint32_t> decorations_;
    std::vector<uint32_t> types_;
    std::vector<uint32_t> code_;
};

SPIRVGenerator::SPIRVGenerator()
    : vulkan_major_(1), vulkan_minor_(2), glsl_std_id_(0) {}

SPIRVGenerator::~SPIRVGenerator() = default;

void SPIRVGenerator::set_target_vulkan_version(uint32_t major, uint32_t minor) {
    vulkan_major_ = major;
    vulkan_minor_ = minor;
}

void SPIRVGenerator::require_capability(SPIRVBuilder& builder, uint32_t capability) {
    if (!emitted_capabilities_.insert(capability).second) {
        return;  // already declared
    }
    SPIRVBuilder::Section prev = builder.get_current_section();
    builder.set_section(SPIRVBuilder::Section::Capabilities);
    builder.emit_op(SPIRVOp::OpCapability, {capability});
    builder.set_section(prev);
}

std::vector<uint32_t> SPIRVGenerator::generate(llvm::Module* module) {
    std::cerr << "[SPIRVGenerator] generate(Module) called" << std::endl;
    
    // Unify: If there is exactly one non-declaration function, use robust path
    llvm::Function* main_func = nullptr;
    int func_count = 0;
    for (auto& func : module->functions()) {
        if (!func.isDeclaration()) {
            main_func = &func;
            func_count++;
        }
    }
    
    if (func_count == 1) {
        std::cerr << "[SPIRVGenerator] Redirecting to generate_from_lambda" << std::endl;
        return generate_from_lambda(main_func, {"float&"});
    }

    std::cerr << "[SPIRVGenerator] Falling back to manual generate" << std::endl;
    SPIRVBuilder builder;
    emit_header(builder.get_header());

    builder.set_section(SPIRVBuilder::Section::Preamble);
    llvm::errs() << "[SPIRVGenerator] FALLBACK PATH: Emitting capabilities\n";
    builder.emit_op(SPIRVOp::OpCapability, {1}); // Shader
    llvm::errs() << "[SPIRVGenerator] FALLBACK: NOT emitting Int64 - GPU doesn't support it\n";
    // DO NOT emit Int64 capability - GPU doesn't support it
    // builder.emit_op(SPIRVOp::OpCapability, {11}); // Int64
    builder.emit_op(SPIRVOp::OpMemoryModel, {0, 1}); // Logical GLSL450
    
    builder.set_section(SPIRVBuilder::Section::Types);
    
    for (auto& func : module->functions()) {
        if (!func.isDeclaration()) {
            uint32_t func_id = builder.get_next_id();
            
            builder.set_section(SPIRVBuilder::Section::EntryPoints);
            std::string func_name = func.getName().str();
            size_t name_words = (func_name.length() + 4) / 4;
            uint32_t ep_wc = 1 + 1 + 1 + name_words; 
            builder.emit_word((ep_wc << 16) | static_cast<uint32_t>(SPIRVOp::OpEntryPoint));
            builder.emit_word(5); // GLCompute
            builder.emit_word(func_id);
            builder.emit_string(func_name);
            
            builder.emit_op(SPIRVOp::OpExecutionMode, {func_id, 17, 256, 1, 1});
            
            builder.set_section(SPIRVBuilder::Section::Code);
            translate_function(builder, &func, func_id);
            break;
        }
    }
    
    builder.get_header()[3] = builder.get_next_id();
    return builder.get_spirv();
}

void SPIRVGenerator::translate_function(SPIRVBuilder& builder, llvm::Function* func, uint32_t func_id,
                                        const std::set<size_t>& buffer_param_indices) {
    std::unordered_map<llvm::Value*, uint32_t> value_map;

    // Get runtime array type for buffer parameters
    llvm::Type* float_ty = llvm::Type::getFloatTy(func->getContext());
    uint32_t float_id = get_type_id(builder, float_ty);

    builder.set_section(SPIRVBuilder::Section::Types);
    uint32_t rarray_id_local = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpTypeRuntimeArray, {rarray_id_local, float_id});
    builder.set_section(SPIRVBuilder::Section::Decorations);
    builder.emit_op(SPIRVOp::OpDecorate, {rarray_id_local, 6 /* ArrayStride */, 4});

    uint32_t ptr_rarray_sb = get_pointer_type_id(builder, rarray_id_local, 12 /* StorageBuffer */);

    // Emit type declarations
    builder.set_section(SPIRVBuilder::Section::Types);
    uint32_t void_type = get_type_id(builder, llvm::Type::getVoidTy(func->getContext()));
    uint32_t return_type = get_type_id(builder, func->getReturnType());

    std::vector<uint32_t> func_type_operands;
    func_type_operands.push_back(return_type); // Return type (use actual, not always void)
    for (auto& arg : func->args()) {
        size_t arg_no = arg.getArgNo();
        bool is_buffer_param = buffer_param_indices.count(arg_no) > 0;
        uint32_t arg_type_id = is_buffer_param ? ptr_rarray_sb : get_type_id(builder, arg.getType());
        func_type_operands.push_back(arg_type_id);
    }

    uint32_t func_type = builder.get_next_id();
    std::vector<uint32_t> all_func_ops = {func_type};
    all_func_ops.insert(all_func_ops.end(), func_type_operands.begin(), func_type_operands.end());
    builder.emit_op(SPIRVOp::OpTypeFunction, all_func_ops);

    // Emit function
    builder.set_section(SPIRVBuilder::Section::Code);
    builder.emit_op(SPIRVOp::OpFunction, {return_type, func_id, 0, func_type});

    // Handle arguments
    for (auto& arg : func->args()) {
        uint32_t arg_id = builder.get_next_id();
        llvm::Type* arg_llvm_type = arg.getType();
        size_t arg_no = arg.getArgNo();
        bool is_buffer_param = buffer_param_indices.count(arg_no) > 0;

        llvm::errs() << "[SPIRVGenerator] Function arg " << arg_no
                     << " LLVM type: " << *arg_llvm_type;
        if (arg_llvm_type->isPointerTy()) {
            llvm::errs() << " (POINTER TYPE!)";
            if (is_buffer_param) {
                llvm::errs() << " -> BUFFER (RuntimeArray)";
            }
        } else if (arg_llvm_type->isIntegerTy()) {
            llvm::errs() << " (INTEGER TYPE, width=" << arg_llvm_type->getIntegerBitWidth() << ")";
        }
        llvm::errs() << "\n";

        uint32_t arg_type_id = is_buffer_param ? ptr_rarray_sb : get_type_id(builder, arg_llvm_type);
        llvm::errs() << "[SPIRVGenerator]   -> SPIR-V type ID: " << arg_type_id << "\n";

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
        case llvm::Instruction::Add: {
             uint32_t op1_add = get_value_id(builder, inst->getOperand(0), value_map);
             uint32_t op2_add = get_value_id(builder, inst->getOperand(1), value_map);
             builder.emit_op(SPIRVOp::OpIAdd, {get_type_id(builder, inst->getType()), result_id, op1_add, op2_add});
             break;
        }
             
        case llvm::Instruction::FAdd: {
             uint32_t op1 = get_value_id(builder, inst->getOperand(0), value_map);
             uint32_t op2 = get_value_id(builder, inst->getOperand(1), value_map);
             builder.emit_op(SPIRVOp::OpFAdd, {get_type_id(builder, inst->getType()), result_id, op1, op2});
             break;
        }
            
        case llvm::Instruction::Sub: {
             uint32_t op1_sub = get_value_id(builder, inst->getOperand(0), value_map);
             uint32_t op2_sub = get_value_id(builder, inst->getOperand(1), value_map);
             builder.emit_op(SPIRVOp::OpISub, {get_type_id(builder, inst->getType()), result_id, op1_sub, op2_sub});
             break;
        }

        case llvm::Instruction::FSub: {
             uint32_t op1 = get_value_id(builder, inst->getOperand(0), value_map);
             uint32_t op2 = get_value_id(builder, inst->getOperand(1), value_map);
             builder.emit_op(SPIRVOp::OpFSub, {get_type_id(builder, inst->getType()), result_id, op1, op2});
             break;
        }
            
        case llvm::Instruction::Mul: {
             uint32_t op1_mul = get_value_id(builder, inst->getOperand(0), value_map);
             uint32_t op2_mul = get_value_id(builder, inst->getOperand(1), value_map);
             builder.emit_op(SPIRVOp::OpIMul, {get_type_id(builder, inst->getType()), result_id, op1_mul, op2_mul});
             break;
        }

        case llvm::Instruction::FMul: {
             uint32_t op1 = get_value_id(builder, inst->getOperand(0), value_map);
             uint32_t op2 = get_value_id(builder, inst->getOperand(1), value_map);
             builder.emit_op(SPIRVOp::OpFMul, {get_type_id(builder, inst->getType()), result_id, op1, op2});
             break;
        }
            
        case llvm::Instruction::FDiv: {
            uint32_t op1 = get_value_id(builder, inst->getOperand(0), value_map);
            uint32_t op2 = get_value_id(builder, inst->getOperand(1), value_map);
            builder.emit_op(SPIRVOp::OpFDiv, {get_type_id(builder, inst->getType()), result_id, op1, op2});
            break;
        }

        case llvm::Instruction::SDiv:
        case llvm::Instruction::UDiv: {
            uint32_t op1 = get_value_id(builder, inst->getOperand(0), value_map);
            uint32_t op2 = get_value_id(builder, inst->getOperand(1), value_map);
            SPIRVOp op = (inst->getOpcode() == llvm::Instruction::SDiv) ? SPIRVOp::OpSDiv : SPIRVOp::OpUDiv;
            builder.emit_op(op, {get_type_id(builder, inst->getType()), result_id, op1, op2});
            break;
        }
        
        case llvm::Instruction::Load: {
            uint32_t ptr = get_value_id(builder, inst->getOperand(0), value_map);
            builder.emit_op(SPIRVOp::OpLoad, {get_type_id(builder, inst->getType()), result_id, ptr});
            break;
        }
        
        case llvm::Instruction::Store: {
            uint32_t value = get_value_id(builder, inst->getOperand(0), value_map);
            uint32_t ptr = get_value_id(builder, inst->getOperand(1), value_map);
            builder.emit_op(SPIRVOp::OpStore, {ptr, value});
            break;
        }

        case llvm::Instruction::GetElementPtr: {
            auto* gep = llvm::cast<llvm::GetElementPtrInst>(inst);
            uint32_t base = get_value_id(builder, gep->getPointerOperand(), value_map);
            
            std::vector<uint32_t> ops;
            ops.push_back(get_type_id(builder, gep->getType()));
            ops.push_back(result_id);
            ops.push_back(base);
            
            for (auto it = gep->idx_begin(); it != gep->idx_end(); ++it) {
                ops.push_back(get_value_id(builder, *it, value_map));
            }
            
            builder.emit_op(SPIRVOp::OpAccessChain, ops);
            break;
        }

        case llvm::Instruction::Alloca: {
            auto* alloca = llvm::cast<llvm::AllocaInst>(inst);
            uint32_t element_ty_id = get_type_id(builder, alloca->getAllocatedType());
            uint32_t ptr_ty_id = get_pointer_type_id(builder, element_ty_id, 7 /* Function */);
            builder.emit_op(SPIRVOp::OpVariable, {ptr_ty_id, result_id, 7 /* Function */});
            break;
        }
        
        case llvm::Instruction::Ret: {
            auto* ret = llvm::cast<llvm::ReturnInst>(inst);
            if (ret->getReturnValue()) {
                // Return with value (OpReturnValue)
                uint32_t val_id = get_value_id(builder, ret->getReturnValue(), value_map);
                builder.emit_op(SPIRVOp::OpReturnValue, {val_id});
            } else {
                // Void return (OpReturn)
                builder.emit_op(SPIRVOp::OpReturn, {});
            }
            break;
        }
            
        case llvm::Instruction::Br: {
            auto* br = llvm::cast<llvm::BranchInst>(inst);
            if (br->isUnconditional()) {
                uint32_t target = get_value_id(builder, br->getSuccessor(0), value_map);
                builder.emit_op(SPIRVOp::OpBranch, {target});
            } else {
                uint32_t cond = get_value_id(builder, br->getCondition(), value_map);
                uint32_t true_label = get_value_id(builder, br->getSuccessor(0), value_map);
                uint32_t false_label = get_value_id(builder, br->getSuccessor(1), value_map);
                builder.emit_op(SPIRVOp::OpBranchConditional, {cond, true_label, false_label});
            }
            break;
        }
        
        case llvm::Instruction::ICmp: {
            auto* cmp = llvm::cast<llvm::ICmpInst>(inst);
            uint32_t op1 = get_value_id(builder, cmp->getOperand(0), value_map);
            uint32_t op2 = get_value_id(builder, cmp->getOperand(1), value_map);
            
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
            uint32_t op1 = get_value_id(builder, cmp->getOperand(0), value_map);
            uint32_t op2 = get_value_id(builder, cmp->getOperand(1), value_map);
            
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
        
        case llvm::Instruction::Call: {
            auto* call = llvm::cast<llvm::CallInst>(inst);
            auto* func = call->getCalledFunction();
            if (func && func->getIntrinsicID() == llvm::Intrinsic::sqrt) {
                uint32_t arg = get_value_id(builder, call->getArgOperand(0), value_map);
                uint32_t ty = get_type_id(builder, inst->getType());
                // OpExtInst result_ty result_id set_id inst_index operand1
                builder.emit_op(SPIRVOp::OpExtInst, {ty, result_id, glsl_std_id_, 31 /* sqrt */, arg});
            } else {
                // Generic call (OpFunctionCall)
                uint32_t func_ptr_id = 0; // Would need a map for functions
                std::vector<uint32_t> ops = {get_type_id(builder, inst->getType()), result_id, func_ptr_id};
                for (unsigned i = 0; i < call->arg_size(); ++i) {
                    ops.push_back(get_value_id(builder, call->getArgOperand(i), value_map));
                }
                builder.emit_op(SPIRVOp::OpFunctionCall, ops);
            }
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
    
    SPIRVBuilder::Section prev_section = builder.get_current_section();
    
    // Switch to Types section
    builder.set_section(SPIRVBuilder::Section::Types);
    
    uint32_t type_id = builder.get_next_id();
    
    if (type->isVoidTy()) {
        builder.emit_op(SPIRVOp::OpTypeVoid, {type_id});
    } else if (type->isIntegerTy(1)) {
        builder.emit_op(SPIRVOp::OpTypeBool, {type_id});
    } else if (type->isIntegerTy(32)) {
        builder.emit_op(SPIRVOp::OpTypeInt, {type_id, 32, 0});
    } else if (type->isIntegerTy(64)) {
        // Emit a real 64-bit integer and declare the Int64 capability. Truncating
        // to 32 bits silently corrupted values above 2^31; the device's shaderInt64
        // support is verified by the runtime at kernel load instead.
        require_capability(builder, 11 /* Int64 */);
        builder.emit_op(SPIRVOp::OpTypeInt, {type_id, 64, 0});
    } else if (type->isFloatTy()) {
        builder.emit_op(SPIRVOp::OpTypeFloat, {type_id, 32});
    } else if (type->isDoubleTy()) {
        require_capability(builder, 10 /* Float64 */);
        builder.emit_op(SPIRVOp::OpTypeFloat, {type_id, 64});
    } else if (type->isPointerTy()) {
        // LLVM 21 opaque pointers carry no pointee type; use the kernel's active
        // element type (set per generate_from_lambda) instead of assuming float.
        llvm::Type* element_type = active_element_type_
                                       ? active_element_type_
                                       : llvm::Type::getFloatTy(type->getContext());
        uint32_t el_ty_id = get_type_id(builder, element_type);
        // Use cached pointer type helper
        uint32_t ptr_ty = get_pointer_type_id(builder, el_ty_id, 12 /* StorageBuffer */);
        type_cache_[type] = ptr_ty;
        return ptr_ty;
    } else {
        builder.emit_op(SPIRVOp::OpTypeInt, {type_id, 32, 0});
    }
    
    type_cache_[type] = type_id;
    
    // Restore previous section
    builder.set_section(prev_section);
    
    return type_id;
}

uint32_t SPIRVGenerator::get_value_id(SPIRVBuilder& builder, llvm::Value* val, std::unordered_map<llvm::Value*, uint32_t>& value_map) {
    if (value_map.count(val)) {
        return value_map[val];
    }

    if (auto* c = llvm::dyn_cast<llvm::Constant>(val)) {
        return get_constant_id(builder, c);
    }

    // ERROR: Value not found and not a constant
    llvm::errs() << "[SPIRVGenerator] ERROR: Value not in value_map: ";
    val->print(llvm::errs());
    llvm::errs() << "\n";
    llvm::errs() << "[SPIRVGenerator] This will cause invalid SPIR-V (Id = 0)\n";
    llvm::errs() << "[SPIRVGenerator] Creating placeholder constant instead\n";

    // Return a constant zero of the appropriate type as a fallback
    llvm::Type* ty = val->getType();
    if (ty->isIntegerTy()) {
        return get_constant_id(builder, llvm::ConstantInt::get(ty, 0));
    } else if (ty->isFloatingPointTy()) {
        return get_constant_id(builder, llvm::ConstantFP::get(ty, 0.0));
    }

    return 0; // Last resort - will cause SPIR-V validation error
}

uint32_t SPIRVGenerator::get_constant_id(SPIRVBuilder& builder, llvm::Constant* c) {
    if (constant_cache_.count(c)) {
        return constant_cache_[c];
    }
    
    SPIRVBuilder::Section prev_section = builder.get_current_section();
    uint32_t id = builder.get_next_id();
    uint32_t ty = get_type_id(builder, c->getType());
    
    builder.set_section(SPIRVBuilder::Section::Types);
    
    if (auto* ci = llvm::dyn_cast<llvm::ConstantInt>(c)) {
        uint32_t val = (uint32_t)ci->getZExtValue();
        builder.emit_op(SPIRVOp::OpConstant, {ty, id, val});
    } else if (auto* cf = llvm::dyn_cast<llvm::ConstantFP>(c)) {
        float fval = cf->getValueAPF().convertToFloat();
        uint32_t val;
        std::memcpy(&val, &fval, sizeof(float));
        builder.emit_op(SPIRVOp::OpConstant, {ty, id, val});
    } else if (llvm::isa<llvm::ConstantPointerNull>(c)) {
        builder.emit_op(SPIRVOp::OpConstant, {ty, id, 0});
    } else {
        builder.emit_op(SPIRVOp::OpConstant, {ty, id, 0}); // Fallback
    }
    
    constant_cache_[c] = id;
    builder.set_section(prev_section);
    return id;
}

// Recover the element type accessed through an (opaque) pointer parameter by
// inspecting the first load/store/GEP that uses it. Returns nullptr if none.
static llvm::Type* infer_pointee_type(llvm::Function* f, const llvm::Argument* arg) {
    for (auto& bb : *f) {
        for (auto& inst : bb) {
            if (auto* ld = llvm::dyn_cast<llvm::LoadInst>(&inst)) {
                if (ld->getPointerOperand() == arg) return ld->getType();
            } else if (auto* st = llvm::dyn_cast<llvm::StoreInst>(&inst)) {
                if (st->getPointerOperand() == arg) return st->getValueOperand()->getType();
            } else if (auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(&inst)) {
                if (gep->getPointerOperand() == arg) return gep->getSourceElementType();
            }
        }
    }
    return nullptr;
}

std::vector<uint32_t> SPIRVGenerator::generate_from_lambda(
    llvm::Function* lambda_func,
    const std::vector<std::string>& param_types) {
    std::cerr << "[SPIRVGenerator] generate_from_lambda called" << std::endl;

    // DEBUG: Write to stderr to prove this code is running
    std::cerr << "=== SPIRV_GEN_DEBUG: generate_from_lambda called ===" << std::endl;
    std::cerr << "=== SPIRV_GEN_DEBUG: About to emit capabilities: Shader and VariablePointersStorageBuffer ONLY ===" << std::endl;
    std::cerr << "=== SPIRV_GEN_DEBUG: NOT emitting Int64 capability! ===" << std::endl;

    SPIRVBuilder builder;
    builder.set_section(SPIRVBuilder::Section::Header);
    emit_header(builder.get_header());
    
    // Base capabilities go into the dedicated Capabilities section so that the
    // Int64 / Float64 capabilities emitted lazily during type translation still
    // precede the memory model. Extensions / imports / memory model stay here.
    emitted_capabilities_.clear();
    require_capability(builder, 1);    // Shader
    require_capability(builder, 4441); // VariablePointersStorageBuffer

    builder.set_section(SPIRVBuilder::Section::Preamble);
    std::string ext_name = "SPV_KHR_variable_pointers";
    uint32_t ext_wc = 1 + (ext_name.length() + 4) / 4;
    builder.emit_word((ext_wc << 16) | (uint32_t)SPIRVOp::OpExtension);
    builder.emit_string(ext_name);
    
    // Import GLSL.std.450 (MUST be before MemoryModel)
    glsl_std_id_ = builder.get_next_id();
    std::string glsl_name = "GLSL.std.450";
    uint32_t glsl_wc = 2 + (glsl_name.length() / 4) + 1; 
    builder.emit_word((glsl_wc << 16) | (uint32_t)SPIRVOp::OpExtInstImport);
    builder.emit_word(glsl_std_id_);
    builder.emit_string(glsl_name);
    
    builder.emit_op(SPIRVOp::OpMemoryModel, {0, 1}); // GLSL450
    
    // Translate Lambda Helper
    builder.set_section(SPIRVBuilder::Section::Code);
    uint32_t lambda_id = builder.get_next_id();

    // Build set of buffer parameter indices - ONLY for captured buffer pointers!
    // Data buffer parameters (param 0 for for_each, params 0-1 for transform) are element pointers,
    // not array pointers, so they should NOT be in buffer_param_indices.
    // Only captured buffer pointers (parameters beyond num_data_params) should be runtime arrays.
    std::set<size_t> buffer_param_indices;

    // Determine how many parameters are data buffers (not captured buffers)
    bool is_transform = !lambda_func->getReturnType()->isVoidTy();
    size_t num_data_params = is_transform ? 2 : 1;

    for (auto& arg : lambda_func->args()) {
        size_t arg_no = arg.getArgNo();
        // Only mark captured buffer pointers (beyond the data buffer params) as buffer parameters
        if (arg.getType()->isPointerTy() && arg_no >= num_data_params) {
            buffer_param_indices.insert(arg_no);
            llvm::errs() << "[SPIRVGenerator] Marking param " << arg_no << " as captured buffer (RuntimeArray)\n";
        } else if (arg.getType()->isPointerTy()) {
            llvm::errs() << "[SPIRVGenerator] Param " << arg_no << " is data buffer (element pointer, not array)\n";
        }
    }

    // Derive the kernel's element type so opaque data pointers and the data
    // buffer use the real type instead of float. for_each: the first data
    // parameter is the element pointer (recover its pointee). transform: the
    // first parameter is the element value.
    active_element_type_ = nullptr;
    if (is_transform) {
        if (lambda_func->arg_size() > 0) {
            active_element_type_ = lambda_func->getArg(0)->getType();
        }
    } else {
        for (auto& a : lambda_func->args()) {
            if (a.getType()->isPointerTy() && a.getArgNo() < num_data_params) {
                active_element_type_ = infer_pointee_type(lambda_func, &a);
                break;
            }
        }
    }
    if (active_element_type_ &&
        (active_element_type_->isVoidTy() || active_element_type_->isPointerTy())) {
        active_element_type_ = nullptr;  // unsupported; fall back to float
    }
    if (active_element_type_) {
        llvm::errs() << "[SPIRVGenerator] Kernel element type: " << *active_element_type_ << "\n";
    }

    translate_function(builder, lambda_func, lambda_id, buffer_param_indices);

    // Generate Kernel Entry Point
    uint32_t entry_id = builder.get_next_id();
    generate_kernel_wrapper(builder, entry_id, lambda_id, lambda_func);
    
    // Update Bound
    builder.get_header()[3] = builder.get_next_id();
    
    return builder.get_spirv();
}

uint32_t SPIRVGenerator::get_pointer_type_id(SPIRVBuilder& builder, uint32_t element_type_id, uint32_t storage_class) {
    auto key = std::make_pair(element_type_id, storage_class);
    if (pointer_type_cache_.count(key)) {
        return pointer_type_cache_[key];
    }

    SPIRVBuilder::Section prev_section = builder.get_current_section();
    builder.set_section(SPIRVBuilder::Section::Types);
    
    uint32_t type_id = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpTypePointer, {type_id, storage_class, element_type_id});
    
    pointer_type_cache_[key] = type_id;
    builder.set_section(prev_section);
    return type_id;
}

void SPIRVGenerator::generate_kernel_wrapper(SPIRVBuilder& builder, uint32_t entry_id, 
                                            uint32_t lambda_func_id, llvm::Function* lambda_func) {
    // 1. Setup Types & Globals (in Types/Decorations Sections)
    
    // Basic Types
    llvm::Type* float_ty = llvm::Type::getFloatTy(lambda_func->getContext());
    uint32_t float_id = get_type_id(builder, float_ty);
    llvm::Type* int32_ty = llvm::Type::getInt32Ty(lambda_func->getContext());
    uint32_t int_id = get_type_id(builder, int32_ty);

    // The data buffer element type — the kernel's active element type, not float.
    llvm::Type* data_elem_ty = active_element_type_ ? active_element_type_ : float_ty;
    uint32_t data_elem_id = get_type_id(builder, data_elem_ty);
    uint32_t data_elem_stride = static_cast<uint32_t>(data_elem_ty->getPrimitiveSizeInBits() / 8);
    if (data_elem_stride < 4) data_elem_stride = 4;

    // RuntimeArray { element }
    builder.set_section(SPIRVBuilder::Section::Types);
    uint32_t rarray_id = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpTypeRuntimeArray, {rarray_id, data_elem_id});
    builder.set_section(SPIRVBuilder::Section::Decorations);
    builder.emit_op(SPIRVOp::OpDecorate, {rarray_id, 6 /* ArrayStride */, data_elem_stride});

    // Buffer Struct { RuntimeArray }
    builder.set_section(SPIRVBuilder::Section::Types);
    uint32_t struct_id = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpTypeStruct, {struct_id, rarray_id});
    builder.set_section(SPIRVBuilder::Section::Decorations);
    builder.emit_op(SPIRVOp::OpMemberDecorate, {struct_id, 0, 35 /* Offset */, 0});
    builder.emit_op(SPIRVOp::OpDecorate, {struct_id, 2 /* Block */});

    // Pointer StorageBuffer Struct
    uint32_t ptr_struct_id = get_pointer_type_id(builder, struct_id, 12 /* StorageBuffer */);

    // Determine if this is a transform (returns non-void)
    bool is_transform = !lambda_func->getReturnType()->isVoidTy();

    // Classify parameters into data buffer params and scalar/capture params
    // IMPORTANT: Only the FIRST parameter(s) are data buffers
    // For transform: params 0-1 are input/output buffers
    // For for_each: param 0 is the data buffer
    // For for_each_n with counting_iterator: NO buffers (param 0 is the index, a scalar)
    // ALL other parameters are lambda captures and go into push constants

    std::vector<llvm::Argument*> buffer_params;
    std::vector<llvm::Argument*> scalar_params;

    llvm::errs() << "[SPIRVGenerator] Classifying " << lambda_func->arg_size()
                 << " parameters for function: " << lambda_func->getName() << "\n";

    size_t num_data_params = is_transform ? 2 : 1;  // How many initial params are data buffers

    for (auto& arg : lambda_func->args()) {
        llvm::Type* arg_type = arg.getType();
        llvm::errs() << "  Param " << arg.getArgNo() << " type: " << *arg_type;

        // Only the first num_data_params parameters can be buffers, and only if they're pointers
        if (arg.getArgNo() < num_data_params && arg_type->isPointerTy()) {
            llvm::errs() << " -> BUFFER (data array pointer)\n";
            buffer_params.push_back(&arg);
        } else {
            llvm::errs() << " -> SCALAR/CAPTURE (push constant)\n";
            scalar_params.push_back(&arg);
        }
    }

    // Further classify scalar_params into captured buffers vs true scalars
    std::vector<llvm::Argument*> captured_buffers;
    std::vector<llvm::Argument*> true_scalars;

    for (auto* param : scalar_params) {
        if (param->getType()->isPointerTy()) {
            llvm::errs() << "  Param " << param->getArgNo() << " is a captured buffer pointer\n";
            captured_buffers.push_back(param);
        } else {
            true_scalars.push_back(param);
        }
    }

    // Total number of buffer bindings = data buffers + captured buffers
    size_t num_buffers = buffer_params.size() + captured_buffers.size();

    llvm::errs() << "[SPIRVGenerator] Creating " << buffer_params.size()
                 << " data buffers, " << captured_buffers.size()
                 << " captured buffer bindings, and " << true_scalars.size()
                 << " scalar captures\n";

    // Create Variables for data buffers (binding 0, ...) and captured buffers (binding N, ...)
    std::vector<uint32_t> buffer_var_ids;
    size_t binding_idx = 0;

    // Data buffers first
    for (size_t i = 0; i < buffer_params.size(); ++i) {
        builder.set_section(SPIRVBuilder::Section::Types);
        uint32_t buffer_var_id = builder.get_next_id();
        builder.emit_op(SPIRVOp::OpVariable, {ptr_struct_id, buffer_var_id, 12});
        builder.set_section(SPIRVBuilder::Section::Decorations);
        builder.emit_op(SPIRVOp::OpDecorate, {buffer_var_id, 33 /* Binding */, static_cast<uint32_t>(binding_idx)});
        builder.emit_op(SPIRVOp::OpDecorate, {buffer_var_id, 34 /* DescriptorSet */, 0});
        buffer_var_ids.push_back(buffer_var_id);
        llvm::errs() << "[SPIRVGenerator] Data buffer " << i << " -> binding " << binding_idx << "\n";
        binding_idx++;
    }

    // Captured buffers next
    for (size_t i = 0; i < captured_buffers.size(); ++i) {
        builder.set_section(SPIRVBuilder::Section::Types);
        uint32_t buffer_var_id = builder.get_next_id();
        builder.emit_op(SPIRVOp::OpVariable, {ptr_struct_id, buffer_var_id, 12});
        builder.set_section(SPIRVBuilder::Section::Decorations);
        builder.emit_op(SPIRVOp::OpDecorate, {buffer_var_id, 33 /* Binding */, static_cast<uint32_t>(binding_idx)});
        builder.emit_op(SPIRVOp::OpDecorate, {buffer_var_id, 34 /* DescriptorSet */, 0});
        buffer_var_ids.push_back(buffer_var_id);
        llvm::errs() << "[SPIRVGenerator] Captured buffer " << i << " (param " << captured_buffers[i]->getArgNo()
                     << ") -> binding " << binding_idx << "\n";
        binding_idx++;
    }

    // NEW: Use Storage Buffer for Captures instead of Push Constants
    // Captures Struct { <true_scalars...> } - only non-pointer captures!
    uint32_t captures_struct_id = 0;
    uint32_t captures_var_id = 0;
    std::vector<uint32_t> capture_member_types;

    if (!true_scalars.empty()) {
        builder.set_section(SPIRVBuilder::Section::Types);
        captures_struct_id = builder.get_next_id();

        // Add type IDs for true scalar parameters (no pointers!)
        for (auto* scalar : true_scalars) {
            llvm::Type* scalar_type = scalar->getType();
            uint32_t scalar_type_id = get_type_id(builder, scalar_type);
            capture_member_types.push_back(scalar_type_id);
            llvm::errs() << "[SPIRVGenerator] Scalar capture param " << scalar->getArgNo()
                         << " added to captures struct\n";
        }

        // Emit OpTypeStruct for captures
        std::vector<uint32_t> struct_ops = {captures_struct_id};
        struct_ops.insert(struct_ops.end(), capture_member_types.begin(), capture_member_types.end());
        builder.emit_op(SPIRVOp::OpTypeStruct, struct_ops);

        // Add member decorations for offsets
        builder.set_section(SPIRVBuilder::Section::Decorations);
        uint32_t offset = 0;
        for (uint32_t i = 0; i < capture_member_types.size(); ++i) {
            builder.emit_op(SPIRVOp::OpMemberDecorate, {captures_struct_id, i, 35 /* Offset */, offset});
            offset += 4;  // Each member is 4 bytes (float or uint32)
        }
        builder.emit_op(SPIRVOp::OpDecorate, {captures_struct_id, 2 /* Block */});

        // Create storage buffer pointer for captures (binding after all data/captured buffers)
        uint32_t ptr_captures_storage = get_pointer_type_id(builder, captures_struct_id, 12 /* StorageBuffer */);

        // Create captures variable
        builder.set_section(SPIRVBuilder::Section::Types);
        captures_var_id = builder.get_next_id();
        builder.emit_op(SPIRVOp::OpVariable, {ptr_captures_storage, captures_var_id, 12 /* StorageBuffer */});
        builder.set_section(SPIRVBuilder::Section::Decorations);
        builder.emit_op(SPIRVOp::OpDecorate, {captures_var_id, 33 /* Binding */, static_cast<uint32_t>(binding_idx)});
        builder.emit_op(SPIRVOp::OpDecorate, {captures_var_id, 34 /* DescriptorSet */, 0});

        llvm::errs() << "[SPIRVGenerator] Created captures storage buffer at binding " << binding_idx << "\n";
        binding_idx++;
    }

    // Push Constants now only contain count (no captures!)
    builder.set_section(SPIRVBuilder::Section::Types);
    uint32_t pc_struct_id = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpTypeStruct, {pc_struct_id, int_id});  // Just { uint count }

    builder.set_section(SPIRVBuilder::Section::Decorations);
    builder.emit_op(SPIRVOp::OpMemberDecorate, {pc_struct_id, 0, 35 /* Offset */, 0});
    builder.emit_op(SPIRVOp::OpDecorate, {pc_struct_id, 2 /* Block */});

    // Pointer PushConstant Struct
    uint32_t ptr_pc_id = get_pointer_type_id(builder, pc_struct_id, 9 /* PushConstant */);

    // PC Variable
    builder.set_section(SPIRVBuilder::Section::Types);
    uint32_t pc_var_id = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpVariable, {ptr_pc_id, pc_var_id, 9});

    // GlobalInvocationID Builtin
    builder.set_section(SPIRVBuilder::Section::Types);
    uint32_t v3uint_id = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpTypeVector, {v3uint_id, int_id, 3});
    uint32_t ptr_input_v3uint_id = get_pointer_type_id(builder, v3uint_id, 1 /* Input */);
    uint32_t gl_id_var_id = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpVariable, {ptr_input_v3uint_id, gl_id_var_id, 1});
    builder.set_section(SPIRVBuilder::Section::Decorations);
    builder.emit_op(SPIRVOp::OpDecorate, {gl_id_var_id, 11 /* BuiltIn */, 28 /* GlobalInvocationID */});
    
    // Entry Point Decl
    builder.set_section(SPIRVBuilder::Section::EntryPoints);
    // Count: Model(1)+Func(1)+Name(2)+GlobalID(1)+Buffers+PC(1)+Captures(0 or 1)
    uint32_t ep_wc = 1 + 1 + 1 + 2 + 1 + static_cast<uint32_t>(buffer_var_ids.size()) + 1;
    if (captures_var_id != 0) ep_wc += 1;

    builder.emit_word((ep_wc << 16) | static_cast<uint32_t>(SPIRVOp::OpEntryPoint));
    builder.emit_word(5); // GLCompute
    builder.emit_word(entry_id);
    builder.emit_word(0x6e69616d); // "main"
    builder.emit_word(0x00000000); // "\0..."
    builder.emit_word(gl_id_var_id);
    for (auto id : buffer_var_ids) builder.emit_word(id);
    builder.emit_word(pc_var_id);
    if (captures_var_id != 0) builder.emit_word(captures_var_id);
    
    builder.emit_op(SPIRVOp::OpExecutionMode, {entry_id, 17 /* LocalSize */, 256, 1, 1});
    
    // 2. Define Main Function
    builder.set_section(SPIRVBuilder::Section::Code);
    uint32_t void_id = get_type_id(builder, llvm::Type::getVoidTy(lambda_func->getContext()));
    uint32_t main_func_type = builder.get_next_id();
    builder.set_section(SPIRVBuilder::Section::Types);
    builder.emit_op(SPIRVOp::OpTypeFunction, {main_func_type, void_id});
    
    builder.set_section(SPIRVBuilder::Section::Code);
    builder.emit_op(SPIRVOp::OpFunction, {void_id, entry_id, 0, main_func_type});
    builder.emit_op(SPIRVOp::OpLabel, {builder.get_next_id()});
    
    // Load GlobalID.x
    uint32_t id_vec = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpLoad, {v3uint_id, id_vec, gl_id_var_id});
    uint32_t id_x = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpCompositeExtract, {int_id, id_x, id_vec, 0});
    
    // Load Count from PC
    uint32_t Zero = get_constant_id(builder, llvm::ConstantInt::get(int32_ty, 0));
    uint32_t ptr_int_pc = get_pointer_type_id(builder, int_id, 9 /* PushConstant */);
    uint32_t ptr_count = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpAccessChain, {ptr_int_pc, ptr_count, pc_var_id, Zero});
    uint32_t count = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpLoad, {int_id, count, ptr_count});
    
    // Bounds Check: if (x < count)
    uint32_t cond = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpULessThan, {get_type_id(builder, llvm::Type::getInt1Ty(lambda_func->getContext())), cond, id_x, count});
    
    uint32_t label_body = builder.get_next_id();
    uint32_t label_merge = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpSelectionMerge, {label_merge, 0});
    builder.emit_op(SPIRVOp::OpBranchConditional, {cond, label_body, label_merge});
    
    builder.emit_op(SPIRVOp::OpLabel, {label_body});
    
    // Access Buffer Data for data buffer arguments
    std::vector<uint32_t> data_buffer_ptrs;

    uint32_t ptr_elem_sb = get_pointer_type_id(builder, data_elem_id, 12 /* StorageBuffer */);
    for (size_t i = 0; i < buffer_params.size(); ++i) {
        uint32_t var_id = buffer_var_ids[i];
        uint32_t element_ptr = builder.get_next_id();
        builder.emit_op(SPIRVOp::OpAccessChain, {ptr_elem_sb, element_ptr, var_id, Zero, id_x});
        data_buffer_ptrs.push_back(element_ptr);
        llvm::errs() << "[SPIRVGenerator] Data buffer param " << i << " -> element ptr\n";
    }

    // Access captured buffer pointers (these are separate buffer bindings!)
    std::vector<uint32_t> captured_buffer_ptrs;
    for (size_t i = 0; i < captured_buffers.size(); ++i) {
        uint32_t var_id = buffer_var_ids[buffer_params.size() + i];
        uint32_t array_ptr = builder.get_next_id();
        uint32_t ptr_rarray_sb = get_pointer_type_id(builder, rarray_id, 12 /* StorageBuffer */);
        builder.emit_op(SPIRVOp::OpAccessChain, {ptr_rarray_sb, array_ptr, var_id, Zero});
        captured_buffer_ptrs.push_back(array_ptr);
        llvm::errs() << "[SPIRVGenerator] Captured buffer param " << captured_buffers[i]->getArgNo()
                     << " -> buffer var " << var_id << " -> array ptr " << array_ptr << "\n";
    }

    // Load scalar parameters from storage buffer (captures) - only true scalars, no pointers!
    std::vector<uint32_t> scalar_values;
    for (size_t i = 0; i < true_scalars.size(); ++i) {
        llvm::Type* scalar_type = true_scalars[i]->getType();

        // Create constant for member index in captures struct
        uint32_t member_idx = get_constant_id(builder, llvm::ConstantInt::get(int32_ty, i));

        uint32_t scalar_type_id = get_type_id(builder, scalar_type);

        // Access chain to get pointer to this capture member in storage buffer
        uint32_t ptr_scalar_sb = get_pointer_type_id(builder, scalar_type_id, 12 /* StorageBuffer */);
        uint32_t ptr_scalar = builder.get_next_id();
        builder.emit_op(SPIRVOp::OpAccessChain, {ptr_scalar_sb, ptr_scalar, captures_var_id, member_idx});

        // Load the value from storage buffer
        uint32_t loaded_val = builder.get_next_id();
        builder.emit_op(SPIRVOp::OpLoad, {scalar_type_id, loaded_val, ptr_scalar});
        scalar_values.push_back(loaded_val);

        llvm::errs() << "[SPIRVGenerator] Loaded scalar param " << true_scalars[i]->getArgNo()
                     << " from captures buffer member " << i << "\n";
    }

    // Call Lambda - different handling for transform vs for_each
    // Note: is_transform was already determined earlier

    if (is_transform && data_buffer_ptrs.size() >= 2) {
        // Transform: lambda returns value, has separate input/output
        // Load from input buffer[0] using the input element type
        uint32_t input_val = builder.get_next_id();
        builder.emit_op(SPIRVOp::OpLoad, {data_elem_id, input_val, data_buffer_ptrs[0]});

        // Call lambda with value (not pointer), captured buffers, and scalar parameters
        uint32_t result_type_id = get_type_id(builder, lambda_func->getReturnType());
        uint32_t result_id = builder.get_next_id();
        std::vector<uint32_t> call_ops = {result_type_id, result_id, lambda_func_id, input_val};
        // Append captured buffer pointers
        call_ops.insert(call_ops.end(), captured_buffer_ptrs.begin(), captured_buffer_ptrs.end());
        // Append scalar parameters
        call_ops.insert(call_ops.end(), scalar_values.begin(), scalar_values.end());
        builder.emit_op(SPIRVOp::OpFunctionCall, call_ops);

        // Store result to output buffer[1]
        builder.emit_op(SPIRVOp::OpStore, {data_buffer_ptrs[1], result_id});
    } else {
        // For_each: lambda modifies in-place via pointer
        // Get the actual return type of the lambda function
        uint32_t lambda_return_type_id = get_type_id(builder, lambda_func->getReturnType());

        uint32_t call_id = builder.get_next_id();
        std::vector<uint32_t> call_ops = {lambda_return_type_id, call_id, lambda_func_id};
        // Add data buffer pointers
        call_ops.insert(call_ops.end(), data_buffer_ptrs.begin(), data_buffer_ptrs.end());
        // Add captured buffer pointers
        call_ops.insert(call_ops.end(), captured_buffer_ptrs.begin(), captured_buffer_ptrs.end());
        // Add scalar parameters
        call_ops.insert(call_ops.end(), scalar_values.begin(), scalar_values.end());
        builder.emit_op(SPIRVOp::OpFunctionCall, call_ops);

        llvm::errs() << "[SPIRVGenerator] Called lambda with " << data_buffer_ptrs.size()
                     << " data buffer args, " << captured_buffer_ptrs.size()
                     << " captured buffer args, and " << scalar_values.size() << " scalar args\n";
    }

    builder.emit_op(SPIRVOp::OpBranch, {label_merge});
    
    // Merge Block
    builder.emit_op(SPIRVOp::OpLabel, {label_merge});
    builder.emit_op(SPIRVOp::OpReturn, {});
    builder.emit_op(SPIRVOp::OpFunctionEnd, {});
}

void SPIRVGenerator::emit_header(std::vector<uint32_t>& spirv) {
    spirv.push_back(0x07230203);
    spirv.push_back(0x00010500); // Version 1.5
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
