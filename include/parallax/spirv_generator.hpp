#ifndef PARALLAX_SPIRV_GENERATOR_HPP
#define PARALLAX_SPIRV_GENERATOR_HPP

#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>
#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace parallax {

class SPIRVBuilder;

// SPIR-V Generator - converts LLVM IR to SPIR-V
class SPIRVGenerator {
public:
    SPIRVGenerator();
    ~SPIRVGenerator();
    
    // Generate SPIR-V from LLVM IR module
    std::vector<uint32_t> generate(llvm::Module* module);
    
    // Generate SPIR-V from lambda IR
    std::vector<uint32_t> generate_from_lambda(
        llvm::Function* lambda_func,
        const std::vector<std::string>& param_types);
    
    // Set target Vulkan version
    void set_target_vulkan_version(uint32_t major, uint32_t minor);
    
private:
    uint32_t vulkan_major_;
    uint32_t vulkan_minor_;
    
    // Helper methods for SPIR-V generation
    void emit_header(std::vector<uint32_t>& spirv);
    void emit_capabilities(std::vector<uint32_t>& spirv);
    void emit_extensions(std::vector<uint32_t>& spirv);
    void emit_memory_model(std::vector<uint32_t>& spirv);
    void emit_entry_point(std::vector<uint32_t>& spirv, const std::string& name);
    void emit_execution_mode(std::vector<uint32_t>& spirv);
    void emit_decorations(std::vector<uint32_t>& spirv);
    void emit_types(std::vector<uint32_t>& spirv);
    void emit_function(std::vector<uint32_t>& spirv, llvm::Function* func);
    
    // IR translation helpers
    // IR translation helpers
    void translate_function(SPIRVBuilder& builder, llvm::Function* func, uint32_t func_id);
    void translate_instruction(SPIRVBuilder& builder, llvm::Instruction* inst,
                               std::unordered_map<llvm::Value*, uint32_t>& value_map);
    uint32_t get_type_id(SPIRVBuilder& builder, llvm::Type* type);
    uint32_t get_pointer_type_id(SPIRVBuilder& builder, uint32_t element_type_id, uint32_t storage_class);
    
    // Kernel generation helpers
    void generate_kernel_wrapper(SPIRVBuilder& builder, uint32_t entry_id, uint32_t lambda_func_id, llvm::Function* lambda_func);
    
private:
    std::unordered_map<llvm::Type*, uint32_t> type_cache_;
    std::unordered_map<std::string, uint32_t> builtin_types_; // For manual types
};

} // namespace parallax

#endif // PARALLAX_SPIRV_GENERATOR_HPP
