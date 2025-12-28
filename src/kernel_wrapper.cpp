#include "parallax/kernel_wrapper.hpp"
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

namespace parallax {

llvm::Function* KernelWrapper::generateWrapper(
    const ClassContext& context,
    llvm::Module* module) {

    llvm::errs() << "[KernelWrapper] Generating wrapper for: "
                 << context.record->getNameAsString() << "\n";

    // Generate kernel signature
    llvm::FunctionType* func_type = generateKernelSignature(context, module);

    // Create the kernel function
    std::string kernel_name = "kernel_" + context.record->getNameAsString();
    llvm::Function* kernel = llvm::Function::Create(
        func_type,
        llvm::Function::ExternalLinkage,
        kernel_name,
        module
    );

    // Set kernel as entry point for GPU
    kernel->setCallingConv(llvm::CallingConv::SPIR_KERNEL);

    // Create entry basic block
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(
        llvm_ctx_, "entry", kernel
    );
    llvm::IRBuilder<> builder(entry);

    // Generate parameter marshalling and body
    generateParameterMarshalling(kernel, context);

    // Find the operator() function in the module
    llvm::Function* operator_func = nullptr;
    for (auto& func : *module) {
        if (func.getName() == "operator()") {
            operator_func = &func;
            break;
        }
    }

    if (!operator_func) {
        llvm::errs() << "[KernelWrapper] ERROR: operator() not found in module\n";
        return nullptr;
    }

    // Inline the operator body
    inlineOperatorBody(kernel, operator_func, context);

    // Add return
    builder.CreateRetVoid();

    // Verify the function
    std::string verify_errors;
    llvm::raw_string_ostream error_stream(verify_errors);
    if (llvm::verifyFunction(*kernel, &error_stream)) {
        llvm::errs() << "[KernelWrapper] ERROR: Kernel verification failed:\n"
                     << verify_errors << "\n";
        return nullptr;
    }

    llvm::errs() << "[KernelWrapper] Successfully generated kernel: "
                 << kernel_name << "\n";

    return kernel;
}

llvm::FunctionType* KernelWrapper::generateKernelSignature(
    const ClassContext& context,
    llvm::Module* module) {

    // Build parameter types:
    // 1. Primary data buffer (array being processed)
    // 2. Element index (thread ID)
    // 3. Captured member variables (one per member)

    std::vector<llvm::Type*> param_types;

    // First parameter: pointer to element type (void* or specific type)
    param_types.push_back(llvm::PointerType::get(llvm_ctx_, 0));

    // Second parameter: index (uint32)
    param_types.push_back(llvm::Type::getInt32Ty(llvm_ctx_));

    // Add captured member variables as parameters
    for (clang::FieldDecl* field : context.member_variables) {
        clang::QualType field_type = field->getType();
        llvm::Type* llvm_type = convertClangType(field_type);
        if (llvm_type) {
            param_types.push_back(llvm_type);
        }
    }

    // Return type is void
    llvm::Type* return_type = llvm::Type::getVoidTy(llvm_ctx_);

    return llvm::FunctionType::get(return_type, param_types, false);
}

void KernelWrapper::generateParameterMarshalling(
    llvm::Function* wrapper,
    const ClassContext& context) {

    llvm::IRBuilder<> builder(&wrapper->getEntryBlock());

    // Get function arguments
    auto arg_it = wrapper->arg_begin();
    llvm::Value* data_ptr = &(*arg_it++);
    data_ptr->setName("data_ptr");

    llvm::Value* index = &(*arg_it++);
    index->setName("index");

    // Create a vector to store captured member values
    std::vector<llvm::Value*> captured_values;

    // Extract captured member values from arguments
    for (size_t i = 0; i < context.member_variables.size(); ++i) {
        llvm::Value* member_arg = &(*arg_it++);
        member_arg->setName(context.member_variables[i]->getNameAsString());
        captured_values.push_back(member_arg);
    }

    // Create a struct to hold captured values (simulating 'this' pointer)
    // For now, we'll pass them directly to the operator()
}

void KernelWrapper::inlineOperatorBody(
    llvm::Function* wrapper,
    llvm::Function* operator_impl,
    const ClassContext& context) {

    llvm::IRBuilder<> builder(&wrapper->getEntryBlock());

    // Get the operator() parameter (should be a reference to element)
    // We need to compute the element pointer: data_ptr + index
    auto arg_it = wrapper->arg_begin();
    llvm::Value* data_ptr = &(*arg_it++);
    llvm::Value* index = &(*arg_it++);

    // Compute element pointer
    llvm::Value* element_ptr = builder.CreateGEP(
        llvm::Type::getInt8Ty(llvm_ctx_),  // i8 for byte addressing
        data_ptr,
        index,
        "element_ptr"
    );

    // Bitcast to correct element type (assuming float for now)
    llvm::Value* typed_ptr = builder.CreateBitCast(
        element_ptr,
        llvm::PointerType::get(llvm_ctx_, 0),
        "typed_ptr"
    );

    // Load the element value
    llvm::Value* element_value = builder.CreateLoad(
        llvm::Type::getFloatTy(llvm_ctx_),
        typed_ptr,
        "element"
    );

    // Call operator() with the element value AND captured variables
    std::vector<llvm::Value*> call_args;
    call_args.push_back(element_value);

    // Add captured member variables (lambda captures) as additional arguments
    for (size_t i = 0; i < context.member_variables.size(); ++i) {
        llvm::Value* capture_arg = &(*arg_it++);
        call_args.push_back(capture_arg);
        llvm::errs() << "[inlineOperatorBody] Adding capture argument: "
                     << context.member_variables[i]->getNameAsString() << "\n";
    }

    llvm::Value* result = builder.CreateCall(
        operator_impl,
        call_args,
        "result"
    );

    // Store result back
    builder.CreateStore(result, typed_ptr);
}

llvm::Type* KernelWrapper::convertClangType(clang::QualType clang_type) {
    clang::QualType canonical = clang_type.getCanonicalType();

    // Handle pointers
    if (canonical->isPointerType()) {
        return llvm::PointerType::get(llvm_ctx_, 0);
    }

    // Handle builtin types
    if (const auto* builtin = llvm::dyn_cast<clang::BuiltinType>(canonical)) {
        switch (builtin->getKind()) {
            case clang::BuiltinType::Void:
                return llvm::Type::getVoidTy(llvm_ctx_);
            case clang::BuiltinType::Bool:
                return llvm::Type::getInt1Ty(llvm_ctx_);
            case clang::BuiltinType::Int:
            case clang::BuiltinType::UInt:
                return llvm::Type::getInt32Ty(llvm_ctx_);
            case clang::BuiltinType::Long:
            case clang::BuiltinType::ULong:
            case clang::BuiltinType::LongLong:
            case clang::BuiltinType::ULongLong:
                return llvm::Type::getInt64Ty(llvm_ctx_);
            case clang::BuiltinType::Float:
                return llvm::Type::getFloatTy(llvm_ctx_);
            case clang::BuiltinType::Double:
                return llvm::Type::getDoubleTy(llvm_ctx_);
            default:
                break;
        }
    }

    // Default fallback
    return llvm::Type::getInt32Ty(llvm_ctx_);
}

} // namespace parallax
