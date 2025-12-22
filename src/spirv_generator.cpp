#include "parallax/spirv_generator.hpp"
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <iostream>
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
        Preamble,     // Capabilities, MemoryModel
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
    std::vector<uint32_t> preamble_;
    std::vector<uint32_t> entry_points_;
    std::vector<uint32_t> decorations_;
    std::vector<uint32_t> types_;
    std::vector<uint32_t> code_;
};

SPIRVGenerator::SPIRVGenerator()
    : vulkan_major_(1), vulkan_minor_(3) {}

SPIRVGenerator::~SPIRVGenerator() = default;

void SPIRVGenerator::set_target_vulkan_version(uint32_t major, uint32_t minor) {
    vulkan_major_ = major;
    vulkan_minor_ = minor;
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
    builder.emit_op(SPIRVOp::OpCapability, {1}); // Shader
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

void SPIRVGenerator::translate_function(SPIRVBuilder& builder, llvm::Function* func, uint32_t func_id) {
    std::unordered_map<llvm::Value*, uint32_t> value_map;
    
    // Emit type declarations
    builder.set_section(SPIRVBuilder::Section::Types);
    uint32_t void_type = get_type_id(builder, llvm::Type::getVoidTy(func->getContext()));
    
    std::vector<uint32_t> func_type_operands;
    func_type_operands.push_back(void_type); // Return type
    for (auto& arg : func->args()) {
        func_type_operands.push_back(get_type_id(builder, arg.getType()));
    }
    
    uint32_t func_type = builder.get_next_id();
    std::vector<uint32_t> all_func_ops = {func_type};
    all_func_ops.insert(all_func_ops.end(), func_type_operands.begin(), func_type_operands.end());
    builder.emit_op(SPIRVOp::OpTypeFunction, all_func_ops);
    
    // Emit function
    builder.set_section(SPIRVBuilder::Section::Code);
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
        
        case llvm::Instruction::Ret:
            builder.emit_op(SPIRVOp::OpReturn, {});
            break;
            
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
    } else if (type->isFloatTy()) {
        builder.emit_op(SPIRVOp::OpTypeFloat, {type_id, 32});
    } else if (type->isDoubleTy()) {
        builder.emit_op(SPIRVOp::OpTypeFloat, {type_id, 64});
    } else if (type->isPointerTy()) {
        // LLVM 21+ uses opaque pointers. For MVP, we assume float elements for array access.
        llvm::Type* element_type = llvm::Type::getFloatTy(type->getContext());
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
    
    return 0; // Should not happen in valid IR
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

std::vector<uint32_t> SPIRVGenerator::generate_from_lambda(
    llvm::Function* lambda_func,
    const std::vector<std::string>& param_types) {
    std::cerr << "[SPIRVGenerator] generate_from_lambda called" << std::endl;
    SPIRVBuilder builder;
    builder.set_section(SPIRVBuilder::Section::Header);
    emit_header(builder.get_header());
    
    // Caps & MemModel
    builder.set_section(SPIRVBuilder::Section::Preamble);
    builder.emit_op(SPIRVOp::OpCapability, {1}); // Shader
    builder.emit_op(SPIRVOp::OpCapability, {4442}); // VariablePointersStorageBuffer
    
    std::string ext_name = "SPV_KHR_variable_pointers";
    uint32_t ext_wc = 1 + (ext_name.length() + 4) / 4;
    builder.emit_word((ext_wc << 16) | (uint32_t)SPIRVOp::OpExtension);
    builder.emit_string(ext_name);
    
    builder.emit_op(SPIRVOp::OpMemoryModel, {0, 1}); // GLSL450
    
    // Translate Lambda Helper
    builder.set_section(SPIRVBuilder::Section::Code);
    uint32_t lambda_id = builder.get_next_id();
    translate_function(builder, lambda_func, lambda_id);
    
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
    
    // RuntimeArray { float }
    builder.set_section(SPIRVBuilder::Section::Types);
    uint32_t rarray_id = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpTypeRuntimeArray, {rarray_id, float_id});
    builder.set_section(SPIRVBuilder::Section::Decorations);
    builder.emit_op(SPIRVOp::OpDecorate, {rarray_id, 6 /* ArrayStride */, 4});

    // Buffer Struct { RuntimeArray }
    builder.set_section(SPIRVBuilder::Section::Types);
    uint32_t struct_id = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpTypeStruct, {struct_id, rarray_id});
    builder.set_section(SPIRVBuilder::Section::Decorations);
    builder.emit_op(SPIRVOp::OpMemberDecorate, {struct_id, 0, 35 /* Offset */, 0});
    builder.emit_op(SPIRVOp::OpDecorate, {struct_id, 2 /* Block */});

    // Pointer StorageBuffer Struct
    uint32_t ptr_struct_id = get_pointer_type_id(builder, struct_id, 12 /* StorageBuffer */);
    
    // Buffer Variable (Set 0, Binding 0)
    builder.set_section(SPIRVBuilder::Section::Types);
    uint32_t buffer_var_id = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpVariable, {ptr_struct_id, buffer_var_id, 12});
    builder.set_section(SPIRVBuilder::Section::Decorations);
    builder.emit_op(SPIRVOp::OpDecorate, {buffer_var_id, 33 /* Binding */, 0});
    builder.emit_op(SPIRVOp::OpDecorate, {buffer_var_id, 34 /* DescriptorSet */, 0});

    // Push Constants Struct { uint count, float multiplier }
    builder.set_section(SPIRVBuilder::Section::Types);
    uint32_t pc_struct_id = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpTypeStruct, {pc_struct_id, int_id, float_id, int_id, int_id}); 
    builder.set_section(SPIRVBuilder::Section::Decorations);
    builder.emit_op(SPIRVOp::OpMemberDecorate, {pc_struct_id, 0, 35 /* Offset */, 0});
    builder.emit_op(SPIRVOp::OpMemberDecorate, {pc_struct_id, 1, 35 /* Offset */, 4});
    builder.emit_op(SPIRVOp::OpMemberDecorate, {pc_struct_id, 2, 35 /* Offset */, 8});
    builder.emit_op(SPIRVOp::OpMemberDecorate, {pc_struct_id, 3, 35 /* Offset */, 12});
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
    uint32_t ep_wc = 1 + 1 + 1 + 2 + 1; // Model(1)+Func(1)+Name(2)+Interface(1)
    builder.emit_word((ep_wc << 16) | static_cast<uint32_t>(SPIRVOp::OpEntryPoint));
    builder.emit_word(5); // GLCompute
    builder.emit_word(entry_id);
    builder.emit_word(0x6e69616d); // "main"
    builder.emit_word(0x00000000); // "\0..."
    builder.emit_word(gl_id_var_id); // Interface
    
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
    
    // Access Buffer Data: float* element_ptr = &buffer.data[x]
    uint32_t ptr_float_sb = get_pointer_type_id(builder, float_id, 12 /* StorageBuffer */);
    uint32_t element_ptr = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpAccessChain, {ptr_float_sb, element_ptr, buffer_var_id, Zero, id_x});
    
    // Call Lambda(element_ptr)
    uint32_t call_id = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpFunctionCall, {void_id, call_id, lambda_func_id, element_ptr});
    
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
