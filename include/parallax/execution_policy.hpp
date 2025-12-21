#ifndef PARALLAX_EXECUTION_POLICY_HPP
#define PARALLAX_EXECUTION_POLICY_HPP

#include <execution>

// Parallax intercepts std::execution::par transparently
// Users write pure ISO C++ code with std::execution::par
// No custom execution policies needed!

namespace parallax {

// Forward declarations
class VulkanBackend;
class MemoryManager;
class KernelLauncher;
class LambdaCompiler;

// Internal implementation (not exposed to users)
class ExecutionPolicyImpl {
public:
    static ExecutionPolicyImpl& instance();
    
    void initialize(VulkanBackend* backend, MemoryManager* memory);
    
    // These are called internally when std::for_each(std::execution::par, ...) is detected
    template<typename Iterator, typename UnaryFunction>
    void for_each_impl(Iterator first, Iterator last, UnaryFunction f);
    
    template<typename InputIt, typename OutputIt, typename UnaryOperation>
    OutputIt transform_impl(InputIt first, InputIt last, OutputIt d_first, UnaryOperation unary_op);
    
private:
    ExecutionPolicyImpl() = default;
    // Note: Runtime integration (backend, memory, launcher) happens in samples
    // This is just the compiler-side framework
};

} // namespace parallax

// NOTE: Actual interception of std::execution::par happens via:
// 1. Compiler plugin (Clang plugin intercepts at compile time)
// 2. OR library preloading (LD_PRELOAD intercepts at runtime)
// 3. OR template specialization in <execution> header

// Users write pure ISO C++:
// std::for_each(std::execution::par, vec.begin(), vec.end(), [](auto& x) { x *= 2; });
// No parallax:: namespace needed!

#endif // PARALLAX_EXECUTION_POLICY_HPP
