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
    SPIRVBuilder builder;
    
    // Emit header
    emit_header(builder.get_header());
    uint32_t bound_id = builder.get_next_id();
    // We need to update bound ID later... but emit_header likely pushed fixed words.
    // Wait, emit_header (default impl) pushed 5 words.
    // Word 3 (index 3) is bound.
    // But builder.get_header() is the vector.
    // If I used emit_header(builder.get_header()), it pushed words.
    // I need to update it at the end?
    // Or just let it be large? No, valid SPIR-V requires correct bound.
    
    // Caps & MemModel
    builder.set_section(SPIRVBuilder::Section::Preamble);
    builder.emit_op(SPIRVOp::OpCapability, {1}); // Shader
    builder.emit_op(SPIRVOp::OpMemoryModel, {0, 1}); // Logical GLSL450
    
    builder.set_section(SPIRVBuilder::Section::Types);
    
    builder.set_section(SPIRVBuilder::Section::Code);
    
    // Find entry point
    for (auto& func : module->functions()) {
        if (!func.isDeclaration()) {
            uint32_t func_id = builder.get_next_id();
            
            // Entry point needs separate handling to put in Header/Entry section?
            // Actually EntryPoint is its own section before Types.
            // My SPIRVBuilder only has 4 sections.
            // Header, Decoration, Types, Code.
            // EntryPoint op goes where?
            // Spec: 1. Header 2. Caps... 4. EntryPoints ... 7. Types 8. Functions
            // So EntryPoint is BEFORE Types.
            // I should put it in Decorations section? Or add EntryPoints section?
            // Using Decorations section for EntryPoint/ExecMode is roughly fine (Both are "Preamble").
            
            builder.set_section(SPIRVBuilder::Section::EntryPoints);
            
            // Emit entry point with correct word count and name string
            std::string func_name = func.getName().str();
            size_t name_words = (func_name.length() + 4) / 4;
            uint32_t ep_wc = 1 + 1 + 1 + name_words; 
            builder.emit_word((ep_wc << 16) | static_cast<uint32_t>(SPIRVOp::OpEntryPoint));
            builder.emit_word(5); // GLCompute
            builder.emit_word(func_id);
            builder.emit_string(func_name);
            
            // Emit execution mode
            builder.emit_op(SPIRVOp::OpExecutionMode, {func_id, 17 /* LocalSize */, 256, 1, 1});
            
            builder.set_section(SPIRVBuilder::Section::Code);
            
            // Translate function
            translate_function(builder, &func, func_id);
            break;
        }
    }
    
    // Update bound
    builder.get_header()[3] = builder.get_next_id();
    
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
                uint32_t op1 = get_value_id(builder, inst->getOperand(0), value_map);
                uint32_t op2 = get_value_id(builder, inst->getOperand(1), value_map);
                builder.emit_op(SPIRVOp::OpIAdd, {get_type_id(builder, inst->getType()), result_id, op1, op2});
            } else {
                uint32_t op1 = get_value_id(builder, inst->getOperand(0), value_map);
                uint32_t op2 = get_value_id(builder, inst->getOperand(1), value_map);
                builder.emit_op(SPIRVOp::OpFAdd, {get_type_id(builder, inst->getType()), result_id, op1, op2});
            }
            break;
            
        case llvm::Instruction::Sub:
            if (inst->getType()->isIntegerTy()) {
                uint32_t op1 = get_value_id(builder, inst->getOperand(0), value_map);
                uint32_t op2 = get_value_id(builder, inst->getOperand(1), value_map);
                builder.emit_op(SPIRVOp::OpISub, {get_type_id(builder, inst->getType()), result_id, op1, op2});
            } else {
                uint32_t op1 = get_value_id(builder, inst->getOperand(0), value_map);
                uint32_t op2 = get_value_id(builder, inst->getOperand(1), value_map);
                builder.emit_op(SPIRVOp::OpFSub, {get_type_id(builder, inst->getType()), result_id, op1, op2});
            }
            break;
            
        case llvm::Instruction::Mul:
            if (inst->getType()->isIntegerTy()) {
                uint32_t op1 = get_value_id(builder, inst->getOperand(0), value_map);
                uint32_t op2 = get_value_id(builder, inst->getOperand(1), value_map);
                builder.emit_op(SPIRVOp::OpIMul, {get_type_id(builder, inst->getType()), result_id, op1, op2});
            } else {
                uint32_t op1 = get_value_id(builder, inst->getOperand(0), value_map);
                uint32_t op2 = get_value_id(builder, inst->getOperand(1), value_map);
                builder.emit_op(SPIRVOp::OpFMul, {get_type_id(builder, inst->getType()), result_id, op1, op2});
            }
            break;
            
        case llvm::Instruction::FDiv: {
            uint32_t op1 = get_value_id(builder, inst->getOperand(0), value_map);
            uint32_t op2 = get_value_id(builder, inst->getOperand(1), value_map);
            builder.emit_op(SPIRVOp::OpFDiv, {get_type_id(builder, inst->getType()), result_id, op1, op2});
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
    
    // Switch to Types section
    builder.set_section(SPIRVBuilder::Section::Types);
    
    uint32_t type_id = builder.get_next_id();
    
    if (type->isVoidTy()) {
        builder.emit_op(SPIRVOp::OpTypeVoid, {type_id});
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
        // Default to StorageBuffer(12) for GPU args
        builder.emit_op(SPIRVOp::OpTypePointer, {type_id, 12 /* StorageBuffer */, el_ty_id});
    } else {
        builder.emit_op(SPIRVOp::OpTypeInt, {type_id, 32, 0});
    }
    
    type_cache_[type] = type_id;
    
    // Restore to Code section
    builder.set_section(SPIRVBuilder::Section::Code);
    
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
        // Technically OpConstantNull, but we'll use 0 for now
        builder.emit_op(SPIRVOp::OpConstant, {ty, id, 0});
    } else {
        // Fallback
        builder.emit_op(SPIRVOp::OpConstant, {ty, id, 0});
    }
    
    constant_cache_[c] = id;
    builder.set_section(SPIRVBuilder::Section::Code);
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
    uint32_t ptr_id = builder.get_next_id();
    builder.set_section(SPIRVBuilder::Section::Types);
    builder.emit_op(SPIRVOp::OpTypePointer, {ptr_id, storage_class, element_type_id});
    builder.set_section(SPIRVBuilder::Section::Code);
    return ptr_id;
}

void SPIRVGenerator::generate_kernel_wrapper(SPIRVBuilder& builder, uint32_t entry_id, 
                                            uint32_t lambda_func_id, llvm::Function* lambda_func) {
    // 1. Setup Types & Globals (in Types/Decorations Sections)
    
    // Float Type
    llvm::Type* float_ty = llvm::Type::getFloatTy(lambda_func->getContext());
    uint32_t float_id = get_type_id(builder, float_ty);
    
    // Int Type
    llvm::Type* int32_ty = llvm::Type::getInt32Ty(lambda_func->getContext());
    uint32_t int_id = get_type_id(builder, int32_ty);
    
    builder.set_section(SPIRVBuilder::Section::Types);

    // RuntimeArray { float }
    uint32_t rarray_id = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpTypeRuntimeArray, {rarray_id, float_id});
    builder.set_section(SPIRVBuilder::Section::Decorations);
    builder.emit_op(SPIRVOp::OpDecorate, {rarray_id, 71 /* ArrayStride */, 4});

    // Struct { RuntimeArray }
    builder.set_section(SPIRVBuilder::Section::Types);
    uint32_t struct_id = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpTypeStruct, {struct_id, rarray_id});
    builder.set_section(SPIRVBuilder::Section::Decorations);
    builder.emit_op(SPIRVOp::OpMemberDecorate, {struct_id, 0, 35 /* Offset */, 0});
    builder.emit_op(SPIRVOp::OpDecorate, {struct_id, 2 /* Block */});

    // Pointer StorageBuffer Struct
    uint32_t ptr_struct_id = get_pointer_type_id(builder, struct_id, 12 /* StorageBuffer */);
    
    // Variable Buffer (Set 0, Binding 0)
    builder.set_section(SPIRVBuilder::Section::Types);
    uint32_t buffer_var_id = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpVariable, {ptr_struct_id, buffer_var_id, 12});
    builder.set_section(SPIRVBuilder::Section::Decorations);
    builder.emit_op(SPIRVOp::OpDecorate, {buffer_var_id, 33 /* Binding */, 0});
    builder.emit_op(SPIRVOp::OpDecorate, {buffer_var_id, 34 /* DescriptorSet */, 0});
    
    // Push Constants { uint count, float multiplier }
    builder.set_section(SPIRVBuilder::Section::Types);
    uint32_t pc_struct_id = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpTypeStruct, {pc_struct_id, int_id, float_id, int_id, int_id}); // Pad to 16 bytes? No, standard layout.
    // Layout: 0: count (4 bytes), 4: multiplier (4 bytes).
    
    builder.set_section(SPIRVBuilder::Section::Decorations);
    builder.emit_op(SPIRVOp::OpMemberDecorate, {pc_struct_id, 0, 35 /* Offset */, 0});
    builder.emit_op(SPIRVOp::OpMemberDecorate, {pc_struct_id, 1, 35 /* Offset */, 4}); // Ensure generic launch (no multiplier) or legacy? 
    // KernelLauncher sets {count, 0, 0, 0} then memcpy multiplier at offset 4.
    // So struct should be { int, float }.
    
    builder.emit_op(SPIRVOp::OpDecorate, {pc_struct_id, 2 /* Block */});
    
    // Pointer PushConstant Struct
    uint32_t ptr_pc_id = get_pointer_type_id(builder, pc_struct_id, 9 /* PushConstant */);
    
    // Variable PC
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
    builder.set_section(SPIRVBuilder::Section::Decorations); // Or EntryPoint specific section? 
    // EntryPoint must be before Types.
    // We are putting it in Decorations. Hopefully sufficient (Decorations < Types).
    // Actually Spec requires: EntryPoints < ExecutionModes < Debug < TypeAnnotations(Decorations) < Types.
    // My SPIRVBuilder order: Header, Decorations, Types, Code.
    // If I put EntryPoint in Decorations, it is before Types.
    // But ExecutionMode is also required.
    
    builder.emit_op(SPIRVOp::OpEntryPoint, {5 /* GLCompute */, entry_id, 0x6e69616d /* "main" */, gl_id_var_id}); // Interface includes Inputs
    // Name "main" is "m" "a" "i" "n" = 0x6e69616d (little endian: n i a m).
    // 0x006e69616d? No, emit_string handles it.
    // Manually:
    // builder.emit_op(SPIRVOp::OpEntryPoint, {5, entry_id}); builder.emit_string("main"); ...
    // But operand vector needed.
    // Can't mix emit_op operands and subsequent emit_string.
    // emit_op takes vector. It writes header with length.
    // String must be PART of operands?
    // My emit_op impl counts implementation.
    // I should create a separate emit_entry_point helper or do minimal hack.
    // Hack: Just emit explicit words for EntryPoint.
    // OpEntryPoint: OpCode | WordCount. Model, FunctionID, Name (literals), Interface IDs...
    // Name "main" (null terminated) -> "main\0" -> 5 bytes -> 2 words.
    // Words: 'm' 'a' 'i' 'n', '\0' 0 0 0.
    // 0x6e69616d, 0x00000000.
    
    // Let's use emit_op but I can't pass string literals easily.
    // I will use emit_string helper inside decorations section manually?
    // SPIRVBuilder::emit_string works by pushing words.
    // But emit_op writes length first.
    // It's confusing.
    // I'll skip name for now (empty string "").
    // Empty string: 1 word 0x00000000.
    
    std::vector<uint32_t> ep_operands = {5 /* GLCompute */, entry_id};
    // Append interface
    ep_operands.push_back(gl_id_var_id);
    
    // Correct order for EntryPoint:
    // OpEntryPoint {Execution Model, Entry Point <id>, Name, Interface <id>, ...}
    
    // Using custom words to handle name string properly
    builder.set_section(SPIRVBuilder::Section::EntryPoints);
    
    // Calculate word count:
    // OpEntryPoint Word (1) + Model (1) + FuncID (1) + Name ("main\0\0\0" -> 2 words) + Interface (1)
    uint32_t ep_wc = 1 + 1 + 1 + 2 + 1; 
    builder.emit_word((ep_wc << 16) | static_cast<uint32_t>(SPIRVOp::OpEntryPoint));
    builder.emit_word(5); // GLCompute
    builder.emit_word(entry_id);
    builder.emit_word(0x6e69616d); // "main"
    builder.emit_word(0x00000000); // "\0..."
    builder.emit_word(gl_id_var_id); // Interface
    
    // ExecutionMode follows EntryPoints
    builder.emit_op(SPIRVOp::OpExecutionMode, {entry_id, 17 /* LocalSize */, 256, 1, 1});
    
    // 2. Define Main Code
    builder.set_section(SPIRVBuilder::Section::Code);
    
    // Void Type
    uint32_t void_id = get_type_id(builder, llvm::Type::getVoidTy(lambda_func->getContext()));
    uint32_t main_func_type = builder.get_next_id();
    builder.set_section(SPIRVBuilder::Section::Types);
    builder.emit_op(SPIRVOp::OpTypeFunction, {main_func_type, void_id});
    
    builder.set_section(SPIRVBuilder::Section::Code);
    builder.emit_op(SPIRVOp::OpFunction, {void_id, entry_id, 0, main_func_type});
    builder.emit_op(SPIRVOp::OpLabel, {builder.get_next_id()});
    
    // Body Logic
    // Load GlobalID
    uint32_t id_vec = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpLoad, {v3uint_id, id_vec, gl_id_var_id});
    
    // Extract X
    uint32_t id_x = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpCompositeExtract, {int_id, id_x, id_vec, 0});
    
    // Access Count (Set 0 in PC struct)
    // Need pointer to int (Uniform/PushConstant)
    // Offset 0 is int count.
    // Ptr -> PC -> Member 0
    uint32_t Zero = builder.get_next_id();
    builder.set_section(SPIRVBuilder::Section::Types);
    builder.emit_op(SPIRVOp::OpConstant, {int_id, Zero, 0});
    builder.set_section(SPIRVBuilder::Section::Code);
    
    uint32_t ptr_count = builder.get_next_id();
    uint32_t ptr_int_pc = get_pointer_type_id(builder, int_id, 9 /* PushConstant */);
    
    builder.emit_op(SPIRVOp::OpAccessChain, {ptr_int_pc, ptr_count, pc_var_id, Zero});
    
    uint32_t count_val = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpLoad, {int_id, count_val, ptr_count});
    
    // Bounds Check: id < count
    // u32 < u32 ? OpULessThan
    uint32_t bool_ty = builder.get_next_id();
    builder.set_section(SPIRVBuilder::Section::Types);
    builder.emit_op(SPIRVOp::OpTypeBool, {bool_ty});
    builder.set_section(SPIRVBuilder::Section::Code);
    
    uint32_t cond = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpULessThan, {bool_ty, cond, id_x, count_val});
    
    // Branch
    uint32_t label_body = builder.get_next_id();
    uint32_t label_merge = builder.get_next_id();
    
    builder.emit_op(SPIRVOp::OpSelectionMerge, {label_merge, 0 /* None */});
    builder.emit_op(SPIRVOp::OpBranchConditional, {cond, label_body, label_merge});
    
    // Body Block
    builder.emit_op(SPIRVOp::OpLabel, {label_body});
    
    // Get Data Pointer
    // Buffer -> Member 0 (RuntimeArray) -> Index id_x
    // Returns pointer to float (StorageBuffer)
    uint32_t ptr_float_sb = get_pointer_type_id(builder, float_id, 12 /* StorageBuffer */);
    uint32_t element_ptr = builder.get_next_id();
    
    // AccessChain indices: 0 (member), id_x (index)
    // Create constant 0 for member index
    // Using Zero we already created.
    builder.emit_op(SPIRVOp::OpAccessChain, {ptr_float_sb, element_ptr, buffer_var_id, Zero, id_x});
    
    // For lambda taking float& (pointer), we pass element_ptr.
    // Call Lambda
    // Note: Lambda func logic needs to handle StorageBuffer pointer? 
    // If lambda was compiled with generic pointers or CrossWorkgroup?
    // get_type_id uses CrossWorkgroup(5) for pointer args.
    // If we pass StorageBuffer(12) pointer to CrossWorkgroup(5) param... SPIR-V might allow it?
    // OpFunctionCall operands must match types.
    // We need to Cast? or Ensure lambda uses StorageBuffer?
    // But lambda is compiled generally.
    // Let's assume Lambda uses CrossWorkgroup.
    // We pass StorageBuffer ptr.
    // This is type mismatch.
    // SPIR-V 1.4+ allows generic pointers (StorageClass=8)?
    // Or we use OpCopyObject? 
    // Or we just load value, pass value?
    // Loop lambda: [](float& x).
    // It expects reference.
    // If I change get_type_id to use StorageBuffer(12) for float pointers?
    // Or Generic(8)?
    // Vulnerability loop: Lambda compiled with type X, we pass type Y.
    // Fix: Hardcode get_type_id to use StorageBuffer(12) for implementation inside lambda?
    // But variables inside lambda (alloca) use Function(7).
    
    // Simplification: Load, Call(with value), Store.
    // But `[](float& x)` writes back.
    // If I pass by value, it won't write back.
    // I MUST pass pointer.
    
    // Hack: Change get_type_id to return StorageBuffer(12) pointer for float* if it's an argument?
    // But `get_type_id` is generic.
    
    // Let's assume Lambda uses CrossWorkgroup(5). 
    // I can cast StorageBuffer(12) to CrossWorkgroup(5) ??
    // Not trivially.
    
    // For MVP: Let's change `get_type_id` pointer storage class default (in my previous edit it was 5).
    // Let's change `ptr_float_sb` (the pointer we got from buffer) to be ... specific?
    // The AccessChain returns a pointer with storage class matching the base (StorageBuffer).
    
    // FIX: Change `get_type_id` to use StorageBuffer(12) for pointers?
    // But then Allocas fail.
    
    // We can iterate the arguments of lambda func and get their type ID.
    // We generated them in `translate_function`.
    // But we don't have access to the IDs generated inside `translate_function`.
    // Wait, `get_type_id` is cached!
    // So if `translate_function` called `get_type_id(float*)`, it got ID X.
    // We call `get_type_id(float*)` here, we get ID X.
    // Check `get_type_id` impl: `builder.emit_op(..., {type_id, 5, ...})`.
    // So ID X is CrossWorkgroup.
    // But our Buffer `OpVariable` is StorageBuffer(12).
    // AccessChain will result in StorageBuffer ptr.
    // Mismatch!
    
    // If I assume `get_type_id` returns CrossWorkgroup.
    // I should create my buffer variable as CrossWorkgroup(5)?
    // Vulkan allows StorageBuffer resources in Workgroup? No.
    // Uses StorageBuffer or Uniform.
    
    // Maybe I modify `get_type_id` to use `StorageBuffer(12)` for pointers?
    // And `alloca` (local vars) use `Function(7)`.
    // `translate_instruction` Alloca logic?
    // I haven't implemented `Alloca` in `translate_instruction`.
    
    // Let's assume lambda arguments are `float*` pointing to buffer.
    // I'll update `get_type_id` to use `StorageBuffer(12)` for pointers. 
    // For lambda taking float& (pointer), we pass element_ptr.
    // Call Lambda
    uint32_t call_id = builder.get_next_id();
    builder.emit_op(SPIRVOp::OpFunctionCall, {void_id, call_id, lambda_func_id, element_ptr}); // Pass pointer
    
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
