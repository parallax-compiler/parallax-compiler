#ifndef PARALLAX_SPIRV_GENERATOR_HPP
#define PARALLAX_SPIRV_GENERATOR_HPP

#include <llvm/IR/Module.h>
#include <vector>
#include <cstdint>
#include <string>

namespace parallax {

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
};

} // namespace parallax

#endif // PARALLAX_SPIRV_GENERATOR_HPP
