#ifndef PARALLAX_EXECUTION_POLICY_HPP
#define PARALLAX_EXECUTION_POLICY_HPP

#include <execution>
#include <iterator>
#include <type_traits>

namespace parallax {

// Custom execution policy for Parallax GPU acceleration
struct parallax_execution_policy {};

// Global instance
inline constexpr parallax_execution_policy par{};

// Execution policy traits
template<>
struct std::is_execution_policy<parallax::parallax_execution_policy> : std::true_type {};

// Forward declarations
class VulkanBackend;
class MemoryManager;
class KernelLauncher;

// Execution policy implementation
class ExecutionPolicyImpl {
public:
    static ExecutionPolicyImpl& instance();
    
    void initialize(VulkanBackend* backend, MemoryManager* memory);
    
    template<typename Iterator, typename UnaryFunction>
    void for_each(parallax_execution_policy, Iterator first, Iterator last, UnaryFunction f);
    
    template<typename InputIt, typename OutputIt, typename UnaryOperation>
    OutputIt transform(parallax_execution_policy, InputIt first, InputIt last, 
                      OutputIt d_first, UnaryOperation unary_op);
    
private:
    ExecutionPolicyImpl() = default;
    VulkanBackend* backend_ = nullptr;
    MemoryManager* memory_ = nullptr;
    std::unique_ptr<KernelLauncher> launcher_;
};

} // namespace parallax

// Override std::for_each for Parallax execution policy
namespace std {

template<typename Iterator, typename UnaryFunction>
void for_each(parallax::parallax_execution_policy policy,
              Iterator first, Iterator last,
              UnaryFunction f) {
    parallax::ExecutionPolicyImpl::instance().for_each(policy, first, last, f);
}

template<typename InputIt, typename OutputIt, typename UnaryOperation>
OutputIt transform(parallax::parallax_execution_policy policy,
                   InputIt first, InputIt last,
                   OutputIt d_first,
                   UnaryOperation unary_op) {
    return parallax::ExecutionPolicyImpl::instance().transform(
        policy, first, last, d_first, unary_op);
}

} // namespace std

#endif // PARALLAX_EXECUTION_POLICY_HPP
