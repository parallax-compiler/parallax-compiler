#include "parallax/spirv_generator.hpp"
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
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
    OpConstantNull = 46,
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
    OpConvertUToPtr = 120,
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
    OpControlBarrier = 224,
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
    
    // Pointer-chasing relocation bases are loaded lazily but must dominate every
    // dereference; reset the per-function cache and load them at the entry block.
    reloc_host_base_id_ = 0;
    reloc_dev_base_id_ = 0;

    // Translate basic blocks
    bool is_entry_block = true;
    for (auto& bb : *func) {
        // ... (label handling)
        uint32_t label_id = builder.get_next_id();
        value_map[&bb] = label_id;
        builder.emit_op(SPIRVOp::OpLabel, {label_id});

        if (is_entry_block && element_is_pointer_) {
            ensure_reloc_bases(builder, func->getContext());
        }
        is_entry_block = false;

        // Translate instructions
        for (auto& inst : bb) {
            translate_instruction(builder, &inst, value_map);
        }
    }
    
    builder.emit_op(SPIRVOp::OpFunctionEnd, {});
}

// Byte alignment for a PhysicalStorageBuffer access of `t` (Aligned operand).
static uint32_t spirv_align_of(llvm::Type* t) {
    uint32_t bytes = static_cast<uint32_t>(t->getPrimitiveSizeInBits() / 8);
    return bytes < 4 ? 4 : bytes;
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
            llvm::Value* ptr_operand = inst->getOperand(0);
            llvm::LLVMContext& ctx = inst->getType()->getContext();
            if (relocatable_values_.count(ptr_operand)) {
                // Dereference a chased host pointer: relocate it to a device
                // PhysicalStorageBuffer pointer and load with an Aligned operand.
                uint32_t host_addr = get_value_id(builder, ptr_operand, value_map);
                uint32_t pp = emit_relocate(builder, host_addr, inst->getType(), ctx);
                builder.emit_op(SPIRVOp::OpLoad,
                                {get_type_id(builder, inst->getType()), result_id, pp,
                                 0x2 /* Aligned */, spirv_align_of(inst->getType())});
            } else if (element_is_pointer_ && inst->getType()->isPointerTy()) {
                // Loading a stored pointer out of the data buffer yields a uint64
                // host address that must be relocated before any dereference.
                uint32_t ptr = get_value_id(builder, ptr_operand, value_map);
                uint32_t u64 = get_type_id(builder, llvm::Type::getInt64Ty(ctx));
                builder.emit_op(SPIRVOp::OpLoad, {u64, result_id, ptr});
                relocatable_values_.insert(inst);
            } else {
                uint32_t ptr = get_value_id(builder, ptr_operand, value_map);
                builder.emit_op(SPIRVOp::OpLoad, {get_type_id(builder, inst->getType()), result_id, ptr});
            }
            break;
        }

        case llvm::Instruction::Store: {
            llvm::Value* val_operand = inst->getOperand(0);
            llvm::Value* ptr_operand = inst->getOperand(1);
            if (relocatable_values_.count(ptr_operand)) {
                // Store through a chased host pointer via PhysicalStorageBuffer.
                llvm::LLVMContext& ctx = val_operand->getType()->getContext();
                uint32_t value = get_value_id(builder, val_operand, value_map);
                uint32_t host_addr = get_value_id(builder, ptr_operand, value_map);
                uint32_t pp = emit_relocate(builder, host_addr, val_operand->getType(), ctx);
                builder.emit_op(SPIRVOp::OpStore,
                                {pp, value, 0x2 /* Aligned */, spirv_align_of(val_operand->getType())});
            } else {
                uint32_t value = get_value_id(builder, val_operand, value_map);
                uint32_t ptr = get_value_id(builder, ptr_operand, value_map);
                builder.emit_op(SPIRVOp::OpStore, {ptr, value});
            }
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
            llvm::Intrinsic::ID iid =
                func ? func->getIntrinsicID() : llvm::Intrinsic::not_intrinsic;
            uint32_t ty = get_type_id(builder, inst->getType());

            // Map an LLVM intrinsic to a GLSL.std.450 ext-inst. Codes per the
            // GLSL.std.450 spec.
            auto extinst = [&](uint32_t glsl_op, std::vector<uint32_t> args) {
                std::vector<uint32_t> ops = {ty, result_id, glsl_std_id_, glsl_op};
                ops.insert(ops.end(), args.begin(), args.end());
                builder.emit_op(SPIRVOp::OpExtInst, ops);
            };
            auto arg = [&](unsigned i) {
                return get_value_id(builder, call->getArgOperand(i), value_map);
            };

            switch (iid) {
                case llvm::Intrinsic::sqrt:    extinst(31, {arg(0)}); break;
                case llvm::Intrinsic::fabs:    extinst(4,  {arg(0)}); break;
                case llvm::Intrinsic::floor:   extinst(8,  {arg(0)}); break;
                case llvm::Intrinsic::ceil:    extinst(9,  {arg(0)}); break;
                case llvm::Intrinsic::sin:     extinst(13, {arg(0)}); break;
                case llvm::Intrinsic::cos:     extinst(14, {arg(0)}); break;
                case llvm::Intrinsic::exp:     extinst(27, {arg(0)}); break;
                case llvm::Intrinsic::log:     extinst(28, {arg(0)}); break;
                case llvm::Intrinsic::pow:     extinst(26, {arg(0), arg(1)}); break;
                case llvm::Intrinsic::minnum:  extinst(37, {arg(0), arg(1)}); break;
                case llvm::Intrinsic::maxnum:  extinst(40, {arg(0), arg(1)}); break;
                case llvm::Intrinsic::fma:
                case llvm::Intrinsic::fmuladd: extinst(50 /* Fma */, {arg(0), arg(1), arg(2)}); break;
                // No-op intrinsics carry no SPIR-V meaning; emit nothing.
                case llvm::Intrinsic::lifetime_start:
                case llvm::Intrinsic::lifetime_end:
                case llvm::Intrinsic::dbg_declare:
                case llvm::Intrinsic::dbg_value:
                case llvm::Intrinsic::assume:
                case llvm::Intrinsic::donothing:
                    break;
                default:
                    // Unsupported call: alias the result to the first argument rather
                    // than emit an OpFunctionCall with id 0 (invalid SPIR-V). This is
                    // a graceful degradation until call-graph compilation (Phase 4).
                    llvm::errs() << "[SPIRVGenerator] Unsupported call '"
                                 << (func ? func->getName() : "<indirect>")
                                 << "'; aliasing result to first argument\n";
                    if (call->arg_size() > 0) {
                        value_map[inst] = arg(0);
                    }
                    break;
            }
            break;
        }

        case llvm::Instruction::Select: {
            // cond ? a : b  — predicates/ternaries (e.g. min/max, pred?1:0).
            auto* sel = llvm::cast<llvm::SelectInst>(inst);
            uint32_t cond = get_value_id(builder, sel->getCondition(), value_map);
            uint32_t tval = get_value_id(builder, sel->getTrueValue(), value_map);
            uint32_t fval = get_value_id(builder, sel->getFalseValue(), value_map);
            builder.emit_op(SPIRVOp::OpSelect,
                            {get_type_id(builder, inst->getType()), result_id, cond, tval, fval});
            break;
        }

        case llvm::Instruction::ZExt:
        case llvm::Instruction::SExt: {
            // Widening integer cast (e.g. i1 -> i32). SPIR-V has no direct zext/sext
            // of a bool, so select between 1 and 0 of the target type.
            llvm::Type* dst = inst->getType();
            uint32_t dst_id = get_type_id(builder, dst);
            llvm::Value* src = inst->getOperand(0);
            if (src->getType()->isIntegerTy(1)) {
                uint32_t cond = get_value_id(builder, src, value_map);
                uint32_t one = get_constant_id(builder, llvm::ConstantInt::get(dst, 1));
                uint32_t zero = get_constant_id(builder, llvm::ConstantInt::get(dst, 0));
                builder.emit_op(SPIRVOp::OpSelect, {dst_id, result_id, cond, one, zero});
            } else {
                // Same-width or wider integer: alias (SPIR-V ints are sign-agnostic).
                value_map[inst] = get_value_id(builder, src, value_map);
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
        // For a pointer-chasing kernel the data element is itself a pointer, so a
        // descriptor-resident pointer addresses a uint64 (the stored host address).
        uint32_t el_ty_id;
        if (element_is_pointer_) {
            el_ty_id = get_type_id(builder, llvm::Type::getInt64Ty(type->getContext()));
        } else {
            llvm::Type* element_type = active_element_type_
                                           ? active_element_type_
                                           : llvm::Type::getFloatTy(type->getContext());
            el_ty_id = get_type_id(builder, element_type);
        }
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
        uint64_t v = ci->getZExtValue();
        if (ci->getBitWidth() > 32) {
            // A 64-bit OpConstant value occupies two words (low, high).
            builder.emit_op(SPIRVOp::OpConstant,
                            {ty, id, (uint32_t)(v & 0xFFFFFFFFu), (uint32_t)(v >> 32)});
        } else {
            builder.emit_op(SPIRVOp::OpConstant, {ty, id, (uint32_t)v});
        }
    } else if (auto* cf = llvm::dyn_cast<llvm::ConstantFP>(c)) {
        if (c->getType()->isDoubleTy()) {
            double dval = cf->getValueAPF().convertToDouble();
            uint64_t bits;
            std::memcpy(&bits, &dval, sizeof(double));
            builder.emit_op(SPIRVOp::OpConstant,
                            {ty, id, (uint32_t)(bits & 0xFFFFFFFFu), (uint32_t)(bits >> 32)});
        } else {
            float fval = cf->getValueAPF().convertToFloat();
            uint32_t val;
            std::memcpy(&val, &fval, sizeof(float));
            builder.emit_op(SPIRVOp::OpConstant, {ty, id, val});
        }
    } else if (llvm::isa<llvm::ConstantPointerNull>(c)) {
        builder.emit_op(SPIRVOp::OpConstantNull, {ty, id});
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
    require_capability(builder, 5347); // PhysicalStorageBufferAddresses (Phase 2)

    builder.set_section(SPIRVBuilder::Section::Preamble);
    for (const char* ext : {"SPV_KHR_variable_pointers", "SPV_KHR_physical_storage_buffer"}) {
        std::string ext_name = ext;
        uint32_t ext_wc = 1 + (ext_name.length() + 4) / 4;
        builder.emit_word((ext_wc << 16) | (uint32_t)SPIRVOp::OpExtension);
        builder.emit_string(ext_name);
    }

    // Import GLSL.std.450 (MUST be before MemoryModel)
    glsl_std_id_ = builder.get_next_id();
    std::string glsl_name = "GLSL.std.450";
    uint32_t glsl_wc = 2 + (glsl_name.length() / 4) + 1; 
    builder.emit_word((glsl_wc << 16) | (uint32_t)SPIRVOp::OpExtInstImport);
    builder.emit_word(glsl_std_id_);
    builder.emit_string(glsl_name);
    
    builder.emit_op(SPIRVOp::OpMemoryModel, {5348, 1}); // PhysicalStorageBuffer64 GLSL450

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
    element_is_pointer_ = false;
    relocatable_values_.clear();
    pc_var_id_ = 0;
    reloc_host_base_id_ = 0;
    reloc_dev_base_id_ = 0;
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
    if (active_element_type_ && active_element_type_->isPointerTy()) {
        // The kernel dereferences a pointer stored in the data (pointer-chasing /
        // software unified memory). The data buffer holds uint64 host addresses;
        // each dereference relocates to a device PhysicalStorageBuffer pointer.
        element_is_pointer_ = true;
        active_element_type_ = nullptr;
        llvm::errs() << "[SPIRVGenerator] Kernel element is a pointer (pointer-chasing); "
                        "data buffer holds uint64 host addresses\n";
    } else if (active_element_type_ && active_element_type_->isVoidTy()) {
        active_element_type_ = nullptr;  // unsupported; fall back to float
    }
    if (active_element_type_) {
        llvm::errs() << "[SPIRVGenerator] Kernel element type: " << *active_element_type_ << "\n";
    }

    // Create the push-constant block (count [+ host_base/dev_base for pointer
    // kernels]) before translating the lambda, so the body can read the bases.
    setup_push_constants(builder, lambda_func->getContext());

    translate_function(builder, lambda_func, lambda_id, buffer_param_indices);

    // Generate Kernel Entry Point
    uint32_t entry_id = builder.get_next_id();
    generate_kernel_wrapper(builder, entry_id, lambda_id, lambda_func);
    
    // Update Bound
    builder.get_header()[3] = builder.get_next_id();

    std::vector<uint32_t> spirv = builder.get_spirv();
    if (const char* dump_path = std::getenv("PARALLAX_DUMP_SPIRV")) {
        std::ofstream out(dump_path, std::ios::binary);
        if (out) {
            out.write(reinterpret_cast<const char*>(spirv.data()),
                      static_cast<std::streamsize>(spirv.size() * sizeof(uint32_t)));
            std::cerr << "[SPIRVGenerator] Dumped " << spirv.size()
                      << " words to " << dump_path << "\n";
        }
    }
    return spirv;
}

std::vector<uint32_t> SPIRVGenerator::generate_reduce_kernel(ReduceElemType elem,
                                                             llvm::Function* user_op) {
    // Element kind specifics.
    const bool is_float = (elem == ReduceElemType::F32 || elem == ReduceElemType::F64);
    const bool is_wide  = (elem == ReduceElemType::F64 || elem == ReduceElemType::I64);
    const uint32_t stride = is_wide ? 8 : 4;

    // The user-op body is translated via the shared translate_instruction path,
    // which uses these member caches/state — start clean (and not pointer-chasing).
    type_cache_.clear();
    constant_cache_.clear();
    pointer_type_cache_.clear();
    active_element_type_ = nullptr;
    element_is_pointer_ = false;
    relocatable_values_.clear();

    SPIRVBuilder B;
    B.set_section(SPIRVBuilder::Section::Header);
    emit_header(B.get_header());

    // Capabilities.
    B.set_section(SPIRVBuilder::Section::Capabilities);
    B.emit_op(SPIRVOp::OpCapability, {1});            // Shader
    if (elem == ReduceElemType::F64) B.emit_op(SPIRVOp::OpCapability, {10}); // Float64
    if (elem == ReduceElemType::I64) B.emit_op(SPIRVOp::OpCapability, {11}); // Int64

    // Logical addressing (this primitive needs no physical pointers).
    B.set_section(SPIRVBuilder::Section::Preamble);
    B.emit_op(SPIRVOp::OpMemoryModel, {0, 1});        // Logical GLSL450

    // ---- Types ----
    B.set_section(SPIRVBuilder::Section::Types);
    uint32_t void_t = B.get_next_id(); B.emit_op(SPIRVOp::OpTypeVoid, {void_t});
    uint32_t fn_t   = B.get_next_id(); B.emit_op(SPIRVOp::OpTypeFunction, {fn_t, void_t});
    uint32_t uint_t = B.get_next_id(); B.emit_op(SPIRVOp::OpTypeInt, {uint_t, 32, 0});
    uint32_t bool_t = B.get_next_id(); B.emit_op(SPIRVOp::OpTypeBool, {bool_t});

    uint32_t elem_t = B.get_next_id();
    if (is_float) B.emit_op(SPIRVOp::OpTypeFloat, {elem_t, is_wide ? 64u : 32u});
    else          B.emit_op(SPIRVOp::OpTypeInt,   {elem_t, is_wide ? 64u : 32u, 1});

    uint32_t v3uint = B.get_next_id(); B.emit_op(SPIRVOp::OpTypeVector, {v3uint, uint_t, 3});
    uint32_t ptr_in_v3 = B.get_next_id(); B.emit_op(SPIRVOp::OpTypePointer, {ptr_in_v3, 1 /*Input*/, v3uint});

    uint32_t rarray = B.get_next_id(); B.emit_op(SPIRVOp::OpTypeRuntimeArray, {rarray, elem_t});
    uint32_t sb_struct = B.get_next_id(); B.emit_op(SPIRVOp::OpTypeStruct, {sb_struct, rarray});
    uint32_t ptr_sb_struct = B.get_next_id(); B.emit_op(SPIRVOp::OpTypePointer, {ptr_sb_struct, 12 /*StorageBuffer*/, sb_struct});
    uint32_t ptr_sb_elem = B.get_next_id(); B.emit_op(SPIRVOp::OpTypePointer, {ptr_sb_elem, 12, elem_t});

    // uint constants (deduped) for indices, the local size and barrier scopes.
    std::unordered_map<uint32_t, uint32_t> uconst;
    auto U = [&](uint32_t v) -> uint32_t {
        auto it = uconst.find(v);
        if (it != uconst.end()) return it->second;
        // Constants live in the Types section; restore the caller's section so
        // body instructions emitted after a U() call stay inside their block.
        SPIRVBuilder::Section prev = B.get_current_section();
        B.set_section(SPIRVBuilder::Section::Types);
        uint32_t id = B.get_next_id();
        B.emit_op(SPIRVOp::OpConstant, {uint_t, id, v});
        uconst[v] = id;
        B.set_section(prev);
        return id;
    };

    uint32_t c256 = U(256);
    uint32_t arr256 = B.get_next_id(); B.emit_op(SPIRVOp::OpTypeArray, {arr256, elem_t, c256});
    uint32_t ptr_wg_arr = B.get_next_id(); B.emit_op(SPIRVOp::OpTypePointer, {ptr_wg_arr, 4 /*Workgroup*/, arr256});
    uint32_t ptr_wg_elem = B.get_next_id(); B.emit_op(SPIRVOp::OpTypePointer, {ptr_wg_elem, 4, elem_t});

    uint32_t pc_struct = B.get_next_id(); B.emit_op(SPIRVOp::OpTypeStruct, {pc_struct, uint_t});
    uint32_t ptr_pc_struct = B.get_next_id(); B.emit_op(SPIRVOp::OpTypePointer, {ptr_pc_struct, 9 /*PushConstant*/, pc_struct});
    uint32_t ptr_pc_uint = B.get_next_id(); B.emit_op(SPIRVOp::OpTypePointer, {ptr_pc_uint, 9, uint_t});

    // Global variables.
    uint32_t gid_var  = B.get_next_id(); B.emit_op(SPIRVOp::OpVariable, {ptr_in_v3, gid_var, 1});
    uint32_t lid_var  = B.get_next_id(); B.emit_op(SPIRVOp::OpVariable, {ptr_in_v3, lid_var, 1});
    uint32_t wgid_var = B.get_next_id(); B.emit_op(SPIRVOp::OpVariable, {ptr_in_v3, wgid_var, 1});
    uint32_t in_var   = B.get_next_id(); B.emit_op(SPIRVOp::OpVariable, {ptr_sb_struct, in_var, 12});
    uint32_t out_var  = B.get_next_id(); B.emit_op(SPIRVOp::OpVariable, {ptr_sb_struct, out_var, 12});
    uint32_t sdata_var= B.get_next_id(); B.emit_op(SPIRVOp::OpVariable, {ptr_wg_arr, sdata_var, 4});
    uint32_t pc_var   = B.get_next_id(); B.emit_op(SPIRVOp::OpVariable, {ptr_pc_struct, pc_var, 9});

    uint32_t main_id = B.get_next_id();

    // Barrier operands: Workgroup execution + memory scope, WorkgroupMemory|AcquireRelease.
    uint32_t scope_wg = U(2);
    uint32_t sem = U(264);

    // ---- Decorations ----
    B.set_section(SPIRVBuilder::Section::Decorations);
    B.emit_op(SPIRVOp::OpDecorate, {rarray, 6 /*ArrayStride*/, stride});
    B.emit_op(SPIRVOp::OpMemberDecorate, {sb_struct, 0, 35 /*Offset*/, 0});
    B.emit_op(SPIRVOp::OpDecorate, {sb_struct, 2 /*Block*/});
    B.emit_op(SPIRVOp::OpDecorate, {in_var, 34 /*DescriptorSet*/, 0});
    B.emit_op(SPIRVOp::OpDecorate, {in_var, 33 /*Binding*/, 0});
    B.emit_op(SPIRVOp::OpDecorate, {out_var, 34, 0});
    B.emit_op(SPIRVOp::OpDecorate, {out_var, 33, 1});
    B.emit_op(SPIRVOp::OpMemberDecorate, {pc_struct, 0, 35, 0});
    B.emit_op(SPIRVOp::OpDecorate, {pc_struct, 2 /*Block*/});
    B.emit_op(SPIRVOp::OpDecorate, {gid_var, 11 /*BuiltIn*/, 28 /*GlobalInvocationId*/});
    B.emit_op(SPIRVOp::OpDecorate, {lid_var, 11, 27 /*LocalInvocationId*/});
    B.emit_op(SPIRVOp::OpDecorate, {wgid_var, 11, 26 /*WorkgroupId*/});

    // ---- Entry point + execution mode ----
    B.set_section(SPIRVBuilder::Section::EntryPoints);
    uint32_t iface[] = {gid_var, lid_var, wgid_var, in_var, out_var, sdata_var, pc_var};
    uint32_t ep_wc = 1 + 1 + 1 + 2 + static_cast<uint32_t>(sizeof(iface) / sizeof(iface[0]));
    B.emit_word((ep_wc << 16) | static_cast<uint32_t>(SPIRVOp::OpEntryPoint));
    B.emit_word(5);  // GLCompute
    B.emit_word(main_id);
    B.emit_word(0x6e69616d);  // "main"
    B.emit_word(0x00000000);  // "\0\0\0\0"
    for (uint32_t id : iface) B.emit_word(id);
    B.emit_op(SPIRVOp::OpExecutionMode, {main_id, 17 /*LocalSize*/, 256, 1, 1});

    // ---- Optional user binary op as a callable SPIR-V function ----
    // Translated through the shared translate_instruction path; its element/int/
    // bool types are primed to reuse the kernel's so no duplicate types appear.
    uint32_t op_fn_id = 0;
    if (user_op) {
        llvm::LLVMContext& ctx = user_op->getContext();
        type_cache_[user_op->getReturnType()] = elem_t;
        type_cache_[llvm::Type::getInt32Ty(ctx)] = uint_t;
        type_cache_[llvm::Type::getInt1Ty(ctx)]  = bool_t;

        B.set_section(SPIRVBuilder::Section::Types);
        uint32_t op_fntype = B.get_next_id();
        B.emit_op(SPIRVOp::OpTypeFunction, {op_fntype, elem_t, elem_t, elem_t});

        B.set_section(SPIRVBuilder::Section::Code);
        op_fn_id = B.get_next_id();
        B.emit_op(SPIRVOp::OpFunction, {elem_t, op_fn_id, 0, op_fntype});
        std::unordered_map<llvm::Value*, uint32_t> op_vmap;
        for (auto& arg : user_op->args()) {
            uint32_t pid = B.get_next_id();
            B.emit_op(SPIRVOp::OpFunctionParameter, {elem_t, pid});
            op_vmap[&arg] = pid;
        }
        // Pre-assign block labels so forward branches resolve.
        for (auto& bb : *user_op) op_vmap[&bb] = B.get_next_id();
        for (auto& bb : *user_op) {
            B.emit_op(SPIRVOp::OpLabel, {op_vmap[&bb]});
            for (auto& inst : bb) translate_instruction(B, &inst, op_vmap);
        }
        B.emit_op(SPIRVOp::OpFunctionEnd, {});
    }

    // ---- Function body ----
    B.set_section(SPIRVBuilder::Section::Code);
    B.emit_op(SPIRVOp::OpFunction, {void_t, main_id, 0, fn_t});
    B.emit_op(SPIRVOp::OpLabel, {B.get_next_id()});

    auto load_x = [&](uint32_t var) -> uint32_t {
        uint32_t vec = B.get_next_id();
        B.emit_op(SPIRVOp::OpLoad, {v3uint, vec, var});
        uint32_t x = B.get_next_id();
        B.emit_op(SPIRVOp::OpCompositeExtract, {uint_t, x, vec, 0});
        return x;
    };
    uint32_t gid = load_x(gid_var);
    uint32_t tid = load_x(lid_var);
    uint32_t wgid = load_x(wgid_var);

    // count = pc.count
    uint32_t pc_count_ptr = B.get_next_id();
    B.emit_op(SPIRVOp::OpAccessChain, {ptr_pc_uint, pc_count_ptr, pc_var, U(0)});
    uint32_t count = B.get_next_id();
    B.emit_op(SPIRVOp::OpLoad, {uint_t, count, pc_count_ptr});

    // blockActive = count - wgid*256: the number of valid elements this workgroup
    // owns. The reduction combines only in-range lanes (tid+s < blockActive), so it
    // needs no identity padding and works for any associative op. Full blocks have
    // blockActive >= 256 > any (tid+s), so the guard is a no-op there.
    uint32_t base = B.get_next_id();
    B.emit_op(SPIRVOp::OpIMul, {uint_t, base, wgid, U(256)});
    uint32_t block_active = B.get_next_id();
    B.emit_op(SPIRVOp::OpISub, {uint_t, block_active, count, base});

    // if (gid < count) sdata[tid] = indata[gid];  (lanes tid>=blockActive are never read)
    uint32_t p_sd_tid = B.get_next_id();
    B.emit_op(SPIRVOp::OpAccessChain, {ptr_wg_elem, p_sd_tid, sdata_var, tid});

    uint32_t inb = B.get_next_id();
    B.emit_op(SPIRVOp::OpULessThan, {bool_t, inb, gid, count});
    uint32_t then0 = B.get_next_id();
    uint32_t m0 = B.get_next_id();
    B.emit_op(SPIRVOp::OpSelectionMerge, {m0, 0});
    B.emit_op(SPIRVOp::OpBranchConditional, {inb, then0, m0});
    B.emit_op(SPIRVOp::OpLabel, {then0});
    {
        uint32_t p_in = B.get_next_id();
        B.emit_op(SPIRVOp::OpAccessChain, {ptr_sb_elem, p_in, in_var, U(0), gid});
        uint32_t v = B.get_next_id();
        B.emit_op(SPIRVOp::OpLoad, {elem_t, v, p_in});
        B.emit_op(SPIRVOp::OpStore, {p_sd_tid, v});
        B.emit_op(SPIRVOp::OpBranch, {m0});
    }
    B.emit_op(SPIRVOp::OpLabel, {m0});
    B.emit_op(SPIRVOp::OpControlBarrier, {scope_wg, scope_wg, sem});

    // Unrolled tree reduction: for (s = 128; s > 0; s >>= 1)
    //   if (tid < s && tid + s < blockActive) sdata[tid] = op(sdata[tid], sdata[tid+s]);
    SPIRVOp add_op = is_float ? SPIRVOp::OpFAdd : SPIRVOp::OpIAdd;
    for (uint32_t s = 128; s > 0; s >>= 1) {
        uint32_t cs = U(s);
        uint32_t c1 = B.get_next_id();
        B.emit_op(SPIRVOp::OpULessThan, {bool_t, c1, tid, cs});
        uint32_t idx2 = B.get_next_id();
        B.emit_op(SPIRVOp::OpIAdd, {uint_t, idx2, tid, cs});
        uint32_t c2 = B.get_next_id();
        B.emit_op(SPIRVOp::OpULessThan, {bool_t, c2, idx2, block_active});
        uint32_t doit = B.get_next_id();
        B.emit_op(SPIRVOp::OpLogicalAnd, {bool_t, doit, c1, c2});
        uint32_t thens = B.get_next_id();
        uint32_t ms = B.get_next_id();
        B.emit_op(SPIRVOp::OpSelectionMerge, {ms, 0});
        B.emit_op(SPIRVOp::OpBranchConditional, {doit, thens, ms});
        B.emit_op(SPIRVOp::OpLabel, {thens});
        {
            uint32_t p_a = B.get_next_id();
            B.emit_op(SPIRVOp::OpAccessChain, {ptr_wg_elem, p_a, sdata_var, tid});
            uint32_t a = B.get_next_id();
            B.emit_op(SPIRVOp::OpLoad, {elem_t, a, p_a});
            uint32_t p_b = B.get_next_id();
            B.emit_op(SPIRVOp::OpAccessChain, {ptr_wg_elem, p_b, sdata_var, idx2});
            uint32_t b = B.get_next_id();
            B.emit_op(SPIRVOp::OpLoad, {elem_t, b, p_b});
            uint32_t sum = B.get_next_id();
            if (user_op) B.emit_op(SPIRVOp::OpFunctionCall, {elem_t, sum, op_fn_id, a, b});
            else         B.emit_op(add_op, {elem_t, sum, a, b});
            B.emit_op(SPIRVOp::OpStore, {p_a, sum});
            B.emit_op(SPIRVOp::OpBranch, {ms});
        }
        B.emit_op(SPIRVOp::OpLabel, {ms});
        B.emit_op(SPIRVOp::OpControlBarrier, {scope_wg, scope_wg, sem});
    }

    // if (tid == 0) partials[wgid] = sdata[0];
    uint32_t is_leader = B.get_next_id();
    B.emit_op(SPIRVOp::OpIEqual, {bool_t, is_leader, tid, U(0)});
    uint32_t thenf = B.get_next_id();
    uint32_t mf = B.get_next_id();
    B.emit_op(SPIRVOp::OpSelectionMerge, {mf, 0});
    B.emit_op(SPIRVOp::OpBranchConditional, {is_leader, thenf, mf});
    B.emit_op(SPIRVOp::OpLabel, {thenf});
    {
        uint32_t p_s0 = B.get_next_id();
        B.emit_op(SPIRVOp::OpAccessChain, {ptr_wg_elem, p_s0, sdata_var, U(0)});
        uint32_t r = B.get_next_id();
        B.emit_op(SPIRVOp::OpLoad, {elem_t, r, p_s0});
        uint32_t p_out = B.get_next_id();
        B.emit_op(SPIRVOp::OpAccessChain, {ptr_sb_elem, p_out, out_var, U(0), wgid});
        B.emit_op(SPIRVOp::OpStore, {p_out, r});
        B.emit_op(SPIRVOp::OpBranch, {mf});
    }
    B.emit_op(SPIRVOp::OpLabel, {mf});
    B.emit_op(SPIRVOp::OpReturn, {});
    B.emit_op(SPIRVOp::OpFunctionEnd, {});

    B.get_header()[3] = B.get_next_id();  // Bound

    std::vector<uint32_t> spirv = B.get_spirv();
    if (const char* dump_path = std::getenv("PARALLAX_DUMP_SPIRV")) {
        std::ofstream out(dump_path, std::ios::binary);
        if (out) out.write(reinterpret_cast<const char*>(spirv.data()),
                           static_cast<std::streamsize>(spirv.size() * sizeof(uint32_t)));
    }
    return spirv;
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

// The SPIR-V type id of one data-buffer element. For an ordinary kernel this is
// the active element type (float/int/double); for a pointer-chasing kernel the
// stored element is a 64-bit host address, so the buffer holds uint64.
uint32_t SPIRVGenerator::data_element_type_id(SPIRVBuilder& builder, llvm::LLVMContext& ctx) {
    if (element_is_pointer_) {
        return get_type_id(builder, llvm::Type::getInt64Ty(ctx));
    }
    llvm::Type* t = active_element_type_ ? active_element_type_ : llvm::Type::getFloatTy(ctx);
    return get_type_id(builder, t);
}

// Create the push-constant block and variable once, before the lambda body is
// translated (so the body can read host_base/dev_base and the wrapper can read
// count). Layout matches the runtime's push: { uint count @0 } for ordinary
// kernels, extended to { uint count @0, uint64 host_base @8, uint64 dev_base @16 }
// for pointer-chasing kernels. count stays at offset 0 in both, so ordinary
// kernels are unaffected.
void SPIRVGenerator::setup_push_constants(SPIRVBuilder& builder, llvm::LLVMContext& ctx) {
    uint32_t int_id = get_type_id(builder, llvm::Type::getInt32Ty(ctx));
    pc_int32_id_ = int_id;

    builder.set_section(SPIRVBuilder::Section::Types);
    uint32_t pc_struct_id = builder.get_next_id();
    if (element_is_pointer_) {
        uint32_t u64 = get_type_id(builder, llvm::Type::getInt64Ty(ctx));
        builder.set_section(SPIRVBuilder::Section::Types);
        builder.emit_op(SPIRVOp::OpTypeStruct, {pc_struct_id, int_id, u64, u64});
        builder.set_section(SPIRVBuilder::Section::Decorations);
        builder.emit_op(SPIRVOp::OpMemberDecorate, {pc_struct_id, 0, 35 /* Offset */, 0});
        builder.emit_op(SPIRVOp::OpMemberDecorate, {pc_struct_id, 1, 35 /* Offset */, 8});
        builder.emit_op(SPIRVOp::OpMemberDecorate, {pc_struct_id, 2, 35 /* Offset */, 16});
        builder.emit_op(SPIRVOp::OpDecorate, {pc_struct_id, 2 /* Block */});
    } else {
        builder.emit_op(SPIRVOp::OpTypeStruct, {pc_struct_id, int_id});
        builder.set_section(SPIRVBuilder::Section::Decorations);
        builder.emit_op(SPIRVOp::OpMemberDecorate, {pc_struct_id, 0, 35 /* Offset */, 0});
        builder.emit_op(SPIRVOp::OpDecorate, {pc_struct_id, 2 /* Block */});
    }

    uint32_t ptr_pc_id = get_pointer_type_id(builder, pc_struct_id, 9 /* PushConstant */);

    builder.set_section(SPIRVBuilder::Section::Types);
    pc_var_id_ = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpVariable, {ptr_pc_id, pc_var_id_, 9 /* PushConstant */});
}

// Load host_base/dev_base from the push-constant block once per lambda function
// (cached in reloc_*_base_id_). Emitted at the lambda's entry block so the values
// dominate every dereference site.
void SPIRVGenerator::ensure_reloc_bases(SPIRVBuilder& builder, llvm::LLVMContext& ctx) {
    if (reloc_host_base_id_ != 0) return;
    uint32_t u64 = get_type_id(builder, llvm::Type::getInt64Ty(ctx));
    llvm::Type* i32 = llvm::Type::getInt32Ty(ctx);
    uint32_t ptr_u64_pc = get_pointer_type_id(builder, u64, 9 /* PushConstant */);
    uint32_t one = get_constant_id(builder, llvm::ConstantInt::get(i32, 1));
    uint32_t two = get_constant_id(builder, llvm::ConstantInt::get(i32, 2));

    builder.set_section(SPIRVBuilder::Section::Code);
    uint32_t p_host = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpAccessChain, {ptr_u64_pc, p_host, pc_var_id_, one});
    reloc_host_base_id_ = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpLoad, {u64, reloc_host_base_id_, p_host});

    uint32_t p_dev = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpAccessChain, {ptr_u64_pc, p_dev, pc_var_id_, two});
    reloc_dev_base_id_ = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpLoad, {u64, reloc_dev_base_id_, p_dev});
}

// Relocate a loaded host address into a device PhysicalStorageBuffer pointer:
//   gpu = dev_base + (host_addr - host_base)
// then OpConvertUToPtr to a pointer to `pointee`. The caller issues the load/
// store through the returned pointer with an Aligned memory operand.
uint32_t SPIRVGenerator::emit_relocate(SPIRVBuilder& builder, uint32_t host_addr_id,
                                       llvm::Type* pointee, llvm::LLVMContext& ctx) {
    ensure_reloc_bases(builder, ctx);
    uint32_t u64 = get_type_id(builder, llvm::Type::getInt64Ty(ctx));

    builder.set_section(SPIRVBuilder::Section::Code);
    uint32_t off = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpISub, {u64, off, host_addr_id, reloc_host_base_id_});
    uint32_t gpu = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpIAdd, {u64, gpu, reloc_dev_base_id_, off});

    uint32_t pelem = get_type_id(builder, pointee);
    uint32_t pptr = get_pointer_type_id(builder, pelem, 5349 /* PhysicalStorageBuffer */);
    uint32_t pp = builder.get_next_id();
    builder.set_section(SPIRVBuilder::Section::Code);
    builder.emit_op(SPIRVOp::OpConvertUToPtr, {pptr, pp, gpu});
    return pp;
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
    // For a pointer-chasing kernel the stored element is a uint64 host address.
    llvm::LLVMContext& ctx = lambda_func->getContext();
    uint32_t data_elem_id = data_element_type_id(builder, ctx);
    uint32_t data_elem_stride;
    if (element_is_pointer_) {
        data_elem_stride = 8;
    } else {
        llvm::Type* data_elem_ty = active_element_type_ ? active_element_type_ : float_ty;
        data_elem_stride = static_cast<uint32_t>(data_elem_ty->getPrimitiveSizeInBits() / 8);
        if (data_elem_stride < 4) data_elem_stride = 4;
    }

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

        if (is_transform) {
            // Transform: the in/out data buffers are synthesized (binding 0/1), not
            // lambda args. arg0 is the element value loaded from in[gid] and passed
            // to the call; any further args are captures. (A unary transform lambda
            // takes the element BY VALUE, so it has no pointer data-buffer arg.)
            if (arg.getArgNo() == 0) {
                llvm::errs() << " -> TRANSFORM INPUT VALUE\n";
            } else {
                llvm::errs() << " -> SCALAR/CAPTURE (push constant)\n";
                scalar_params.push_back(&arg);
            }
        } else if (arg.getArgNo() < num_data_params && arg_type->isPointerTy()) {
            llvm::errs() << " -> BUFFER (data array pointer)\n";
            buffer_params.push_back(&arg);
        } else {
            llvm::errs() << " -> SCALAR/CAPTURE (push constant)\n";
            scalar_params.push_back(&arg);
        }
    }

    // Number of data buffers: 1 for for_each (the element), 2 for transform (in,
    // out). For transform these are synthesized — no backing lambda argument.
    const size_t num_data_buffers = is_transform ? 2 : buffer_params.size();

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
    size_t num_buffers = num_data_buffers + captured_buffers.size();

    llvm::errs() << "[SPIRVGenerator] Creating " << num_data_buffers
                 << " data buffers, " << captured_buffers.size()
                 << " captured buffer bindings, and " << true_scalars.size()
                 << " scalar captures\n";

    // Create Variables for data buffers (binding 0, ...) and captured buffers (binding N, ...)
    std::vector<uint32_t> buffer_var_ids;
    size_t binding_idx = 0;

    // Data buffers first
    for (size_t i = 0; i < num_data_buffers; ++i) {
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

    // Push constants were created up front (setup_push_constants): { uint count }
    // for ordinary kernels, extended with host_base/dev_base for pointer-chasing.
    // count is at member 0 in both layouts.

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
    builder.emit_word(pc_var_id_);
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
    builder.emit_op(SPIRVOp::OpAccessChain, {ptr_int_pc, ptr_count, pc_var_id_, Zero});
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
    for (size_t i = 0; i < num_data_buffers; ++i) {
        uint32_t var_id = buffer_var_ids[i];
        uint32_t element_ptr = builder.get_next_id();
        builder.emit_op(SPIRVOp::OpAccessChain, {ptr_elem_sb, element_ptr, var_id, Zero, id_x});
        data_buffer_ptrs.push_back(element_ptr);
        llvm::errs() << "[SPIRVGenerator] Data buffer param " << i << " -> element ptr\n";
    }

    // Access captured buffer pointers (these are separate buffer bindings!)
    std::vector<uint32_t> captured_buffer_ptrs;
    for (size_t i = 0; i < captured_buffers.size(); ++i) {
        uint32_t var_id = buffer_var_ids[num_data_buffers + i];
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

        if (predicate_count_) {
            // count_if: the lambda is a predicate (returns bool); store 1/0 of the
            // element type as this element's contribution to the count.
            llvm::Type* t = active_element_type_ ? active_element_type_ : float_ty;
            llvm::Constant* c_one = t->isFloatingPointTy()
                ? (llvm::Constant*)llvm::ConstantFP::get(t, 1.0)
                : (llvm::Constant*)llvm::ConstantInt::get(t, 1);
            llvm::Constant* c_zero = t->isFloatingPointTy()
                ? (llvm::Constant*)llvm::ConstantFP::get(t, 0.0)
                : (llvm::Constant*)llvm::ConstantInt::get(t, 0);
            uint32_t one = get_constant_id(builder, c_one);
            uint32_t zero = get_constant_id(builder, c_zero);
            uint32_t cnt = builder.get_next_id();
            builder.emit_op(SPIRVOp::OpSelect, {data_elem_id, cnt, result_id, one, zero});
            builder.emit_op(SPIRVOp::OpStore, {data_buffer_ptrs[1], cnt});
        } else {
            // Store result to output buffer[1]
            builder.emit_op(SPIRVOp::OpStore, {data_buffer_ptrs[1], result_id});
        }
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
