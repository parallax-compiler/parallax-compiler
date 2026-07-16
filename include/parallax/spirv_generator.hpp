#ifndef PARALLAX_SPIRV_GENERATOR_HPP
#define PARALLAX_SPIRV_GENERATOR_HPP

#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>
#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <set>

namespace parallax {

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

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

    // Phase 3: the kernel's scalar element kind for fixed-skeleton primitives
    // (reduce/scan/…). SPIR-V types are emitted directly, so no LLVM type/context
    // is needed — just the element's width and float/int-ness.
    enum class ReduceElemType { F32, F64, I32, I64 };

    // Emit a workgroup tree-reduction kernel for the given element type. Shared
    // memory + barriers, one partial per workgroup, fully unrolled, guarded (no
    // identity element). Logical GLSL450; the runtime dispatches it iteratively
    // (see launch_reduce). If user_op is non-null it is a compiled binary op
    // (T(T,T)); the kernel emits it as a SPIR-V function and calls it at each
    // combine step (the user op path). Null = baked-in '+'.
    std::vector<uint32_t> generate_reduce_kernel(ReduceElemType elem,
                                                 llvm::Function* user_op = nullptr);

    // Phase 5: inclusive prefix scan. Two fixed kernels the runtime dispatches in 3
    // passes (see launch_scan): generate_scan_kernel does a per-workgroup Hillis-
    // Steele inclusive scan in place (data@0) and writes each chunk total to
    // blocksums@1; generate_scan_add_kernel adds each block's exclusive offset back
    // (data@0, offsets@1). Logical GLSL450, baked-in '+'. Bindings/push match
    // dispatch_reduce_level so the runtime treats them like the reduce kernel.
    std::vector<uint32_t> generate_scan_kernel(ReduceElemType elem);
    std::vector<uint32_t> generate_scan_add_kernel(ReduceElemType elem);

    // Phase 5: exclusive-scan finalize/shift. in@0 holds an INCLUSIVE scan; out@1 receives
    // the exclusive scan: out[i] = init + (i>0 ? in[i-1] : 0), so out[0]=init and
    // out[i]=init+sum(src[0..i-1]). push { uint count@0, elem init@8 } (16 bytes). Pairs
    // with the two inclusive-scan kernels (run over a scratch copy of the input) to give a
    // fully-on-GPU std::exclusive_scan (default '+').
    std::vector<uint32_t> generate_exclusive_shift_kernel(ReduceElemType elem);

    // Phase 5: one global bitonic compare-exchange stage (ascending). data@0, push
    // { uint count, uint k, uint j }. The runtime (launch_sort) dispatches it over
    // the O(log^2 n) (k,j) schedule. No shared memory or barriers. Default '<' only.
    std::vector<uint32_t> generate_sort_kernel(ReduceElemType elem);

    // Phase 5: compaction scatter. input@0, output@1, positions@3 (the inclusive scan
    // of the 1/0 flags), push { uint count }. Each kept element (positions[i] differs
    // from positions[i-1]) is written to output[positions[i]-1]. Pairs with the
    // predicate-flags transform + scan to implement copy_if.
    std::vector<uint32_t> generate_scatter_kernel(ReduceElemType elem);

    // Phase 5: unique flags. input@0, flags@1, push { uint count }. Flags the first
    // element of each run of equal adjacent values: flag[i] = (i==0 || in[i]!=in[i-1]).
    // Pairs with scan + scatter to implement std::unique (in place).
    std::vector<uint32_t> generate_unique_flags_kernel(ReduceElemType elem);

    // Phase 5: partition scatter. input@0, output@1, positions@3, push {count,num_true}.
    // Writes EVERY element: a kept element (positions[i]!=positions[i-1]) goes to its
    // rank among kept (positions[i]-1); a not-kept element goes after the kept block
    // (num_true + i - positions[i]). Pairs with predicate-flags + scan for std::partition.
    std::vector<uint32_t> generate_partition_scatter_kernel(ReduceElemType elem);
    
    // Set target Vulkan version
    void set_target_vulkan_version(uint32_t major, uint32_t minor);

    // count_if: build the next transform kernel from a predicate (T->bool), storing
    // int 1/0 per element so a '+' reduce yields the count.
    void set_predicate_count(bool v) { predicate_count_ = v; }

    // copy_if flags: like predicate_count but stores 1/0 in the ELEMENT type T (so a
    // float scan can turn the flags into output positions for the scatter).
    void set_predicate_flags(bool v) { predicate_flags_ = v; }

    // remove_if: keep elements where the predicate is FALSE. Negates the flag so the
    // same scatter compacts the not-removed elements.
    void set_predicate_negate(bool v) { predicate_negate_ = v; }
    
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
    void translate_function(SPIRVBuilder& builder, llvm::Function* func, uint32_t func_id,
                            const std::set<size_t>& buffer_param_indices = {});
    void translate_instruction(SPIRVBuilder& builder, llvm::Instruction* inst,
                               std::unordered_map<llvm::Value*, uint32_t>& value_map);
    // True if a scalar load/store through `ptr` is an offset-0 struct field whose GEP
    // LLVM elided (needs a synthesized member-0 access). Bails (translation_failed_)
    // if the offset-0 member's type doesn't match the scalar.
    bool     is_elided_struct_field(llvm::Value* ptr, llvm::Type* scalar_ty);
    uint32_t emit_member0_ptr(SPIRVBuilder& builder, uint32_t base,
                              llvm::Type* scalar_ty, llvm::LLVMContext& ctx);
    uint32_t get_type_id(SPIRVBuilder& builder, llvm::Type* type);
    uint32_t get_pointer_type_id(SPIRVBuilder& builder, uint32_t element_type_id, uint32_t storage_class);
    uint32_t get_value_id(SPIRVBuilder& builder, llvm::Value* val, std::unordered_map<llvm::Value*, uint32_t>& value_map);
    uint32_t get_constant_id(SPIRVBuilder& builder, llvm::Constant* c);
    
    // Kernel generation helpers
    void generate_kernel_wrapper(SPIRVBuilder& builder, uint32_t entry_id, uint32_t lambda_func_id, llvm::Function* lambda_func);

    // Phase 2d (pointer-chasing / software unified memory). When the kernel's
    // data element is itself a pointer (e.g. a stored float* / Node::next), it is
    // a 64-bit HOST address that must be relocated before it can be dereferenced
    // on the device: gpu = dev_base + (host_ptr - host_base). These helpers
    // create the push-constant block carrying {count, host_base, dev_base}, treat
    // the data element as uint64, and emit the relocation + PhysicalStorageBuffer
    // load/store at each dereference of a loaded pointer.
    void     setup_push_constants(SPIRVBuilder& builder, llvm::LLVMContext& ctx);
    uint32_t data_element_type_id(SPIRVBuilder& builder, llvm::LLVMContext& ctx);
    void     ensure_reloc_bases(SPIRVBuilder& builder, llvm::LLVMContext& ctx);
    uint32_t emit_relocate(SPIRVBuilder& builder, uint32_t host_addr_id,
                           llvm::Type* pointee, llvm::LLVMContext& ctx);

    // Emit an OpCapability into the Capabilities section exactly once. Used to
    // declare Int64 / Float64 lazily when those types appear, so we emit real
    // 64-bit types instead of silently truncating them.
    void require_capability(SPIRVBuilder& builder, uint32_t capability);

private:
    std::unordered_map<llvm::Type*, uint32_t> type_cache_;
    std::unordered_map<llvm::Constant*, uint32_t> constant_cache_;
    std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t, pair_hash> pointer_type_cache_;
    std::unordered_map<std::string, uint32_t> builtin_types_;
    std::set<uint32_t> emitted_capabilities_;
    uint32_t glsl_std_id_;

    // The element type the kernel operates on (the T in for_each<T>/transform<T>).
    // LLVM 21 opaque pointers carry no pointee type, so rather than assume float
    // everywhere, the generator is monomorphic in this type: data buffers, the
    // lambda's pointer parameters, and element loads/stores all use it. Set per
    // generate_from_lambda(); null falls back to float.
    llvm::Type* active_element_type_ = nullptr;

    // Phase 2d state. element_is_pointer_ is set when the data element is a
    // pointer: the data buffer then stores uint64 host addresses. pc_var_id_ /
    // pc_int32_id_ are the push-constant variable and its uint32 type, created
    // once up front so both the lambda body (relocation) and the wrapper (count)
    // can reference them. reloc_*_base_id_ are the host_base/dev_base values
    // loaded once at the lambda's entry block (reset per function).
    // relocatable_values_ marks SSA values that are loaded host pointers and so
    // must be relocated before any dereference.
    bool        element_is_pointer_ = false;
    // Phase 3 count_if: when set, a transform kernel built from a predicate (T->bool)
    // stores 1/0 of the element type (the per-element count) instead of the bool.
    bool        predicate_count_ = false;
    bool        predicate_flags_ = false;
    bool        predicate_negate_ = false;
    // A transform/predicate functor whose INPUT is taken by reference (const T&), so its
    // first parameter is a pointer to the element (not the value). The kernel must pass
    // the input buffer element POINTER, not a loaded value. Set per generate_from_lambda.
    bool        transform_byref_input_ = false;
    uint32_t    pc_var_id_ = 0;
    uint32_t    pc_int32_id_ = 0;
    uint32_t    reloc_host_base_id_ = 0;
    uint32_t    reloc_dev_base_id_ = 0;
    std::unordered_set<llvm::Value*> relocatable_values_;

    // Struct element support. data_layout_ (set per generate_from_lambda) gives the
    // host member offsets/sizes so the device reads the same bytes the host wrote.
    // struct_element_ptrs_ holds pointer values that point at the whole struct element
    // (the lambda's element-pointer param); LLVM elides the offset-0 field GEP
    // (`p.x` -> `load T, ptr %p`), so a scalar load/store through such a pointer needs
    // a synthesized member-0 OpAccessChain (a raw OpLoad of T through a ptr-to-struct
    // is invalid SPIR-V).
    const llvm::DataLayout* data_layout_ = nullptr;
    std::unordered_set<llvm::Value*> struct_element_ptrs_;

    // Set true when translate_instruction hits an LLVM instruction it cannot lower
    // (an unsupported construct in the user callable). generate_from_lambda checks
    // this and returns EMPTY SPIR-V so the rewriter leaves the algorithm on the CPU,
    // rather than emitting invalid SPIR-V silently. Reset at the start of each
    // generate_from_lambda. The name of the offending opcode is logged.
    bool        translation_failed_ = false;
};

} // namespace parallax

#endif // PARALLAX_SPIRV_GENERATOR_HPP
