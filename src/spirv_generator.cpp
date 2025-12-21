#include "parallax/spirv_generator.hpp"
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>

namespace parallax {

SPIRVGenerator::SPIRVGenerator()
    : vulkan_major_(1), vulkan_minor_(3) {}

SPIRVGenerator::~SPIRVGenerator() = default;

void SPIRVGenerator::set_target_vulkan_version(uint32_t major, uint32_t minor) {
    vulkan_major_ = major;
    vulkan_minor_ = minor;
}

std::vector<uint32_t> SPIRVGenerator::generate(llvm::Module* module) {
    std::vector<uint32_t> spirv;
    
    // Emit SPIR-V header
    emit_header(spirv);
    emit_capabilities(spirv);
    emit_extensions(spirv);
    emit_memory_model(spirv);
    
    // Find entry point function
    for (auto& func : module->functions()) {
        if (!func.isDeclaration()) {
            emit_entry_point(spirv, func.getName().str());
            emit_execution_mode(spirv);
            emit_decorations(spirv);
            emit_types(spirv);
            emit_function(spirv, &func);
            break; // Only handle first function for now
        }
    }
    
    return spirv;
}

std::vector<uint32_t> SPIRVGenerator::generate_from_lambda(
    llvm::Function* lambda_func,
    const std::vector<std::string>& param_types) {
    
    std::vector<uint32_t> spirv;
    
    emit_header(spirv);
    emit_capabilities(spirv);
    emit_memory_model(spirv);
    emit_entry_point(spirv, lambda_func->getName().str());
    emit_execution_mode(spirv);
    emit_function(spirv, lambda_func);
    
    return spirv;
}

void SPIRVGenerator::emit_header(std::vector<uint32_t>& spirv) {
    // SPIR-V magic number
    spirv.push_back(0x07230203);
    
    // Version (1.6 = 0x00010600)
    spirv.push_back(0x00010600);
    
    // Generator magic number (0 = unknown)
    spirv.push_back(0x000d000b);
    
    // Bound (will be updated)
    spirv.push_back(0x00000100);
    
    // Schema (0 = reserved)
    spirv.push_back(0x00000000);
}

void SPIRVGenerator::emit_capabilities(std::vector<uint32_t>& spirv) {
    // OpCapability Shader
    spirv.push_back(0x00020011);
    spirv.push_back(0x00000001);
}

void SPIRVGenerator::emit_extensions(std::vector<uint32_t>& spirv) {
    // No extensions for now
}

void SPIRVGenerator::emit_memory_model(std::vector<uint32_t>& spirv) {
    // OpMemoryModel Logical GLSL450
    spirv.push_back(0x0003000e);
    spirv.push_back(0x00000000);
    spirv.push_back(0x00000001);
}

void SPIRVGenerator::emit_entry_point(std::vector<uint32_t>& spirv, const std::string& name) {
    // OpEntryPoint GLCompute %main "main"
    spirv.push_back(0x0006000f);
    spirv.push_back(0x00000005); // GLCompute
    spirv.push_back(0x00000004); // Function ID
    
    // Name (null-terminated string)
    for (char c : name) {
        spirv.push_back(static_cast<uint32_t>(c));
    }
    spirv.push_back(0);
}

void SPIRVGenerator::emit_execution_mode(std::vector<uint32_t>& spirv) {
    // OpExecutionMode %main LocalSize 256 1 1
    spirv.push_back(0x00060010);
    spirv.push_back(0x00000004); // Function ID
    spirv.push_back(0x00000011); // LocalSize
    spirv.push_back(0x00000100); // 256
    spirv.push_back(0x00000001); // 1
    spirv.push_back(0x00000001); // 1
}

void SPIRVGenerator::emit_decorations(std::vector<uint32_t>& spirv) {
    // Decorations for bindings, etc.
    // This is a simplified version
}

void SPIRVGenerator::emit_types(std::vector<uint32_t>& spirv) {
    // OpTypeVoid
    spirv.push_back(0x00020013);
    spirv.push_back(0x00000002);
    
    // OpTypeFunction
    spirv.push_back(0x00030021);
    spirv.push_back(0x00000003);
    spirv.push_back(0x00000002);
    
    // OpTypeFloat 32
    spirv.push_back(0x00030016);
    spirv.push_back(0x00000006);
    spirv.push_back(0x00000020);
    
    // OpTypeInt 32 0
    spirv.push_back(0x00040015);
    spirv.push_back(0x00000007);
    spirv.push_back(0x00000020);
    spirv.push_back(0x00000000);
}

void SPIRVGenerator::emit_function(std::vector<uint32_t>& spirv, llvm::Function* func) {
    // OpFunction
    spirv.push_back(0x00050036);
    spirv.push_back(0x00000002); // Return type
    spirv.push_back(0x00000004); // Result ID
    spirv.push_back(0x00000000); // Function control
    spirv.push_back(0x00000003); // Function type
    
    // OpLabel
    spirv.push_back(0x000200f8);
    spirv.push_back(0x00000005);
    
    // Translate LLVM IR instructions to SPIR-V
    // This is a simplified version - full implementation would handle all IR instructions
    
    // OpReturn
    spirv.push_back(0x000100fd);
    
    // OpFunctionEnd
    spirv.push_back(0x00010038);
}

} // namespace parallax
