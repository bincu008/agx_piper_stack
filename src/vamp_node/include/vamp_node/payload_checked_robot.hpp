#pragma once

#include <functional>
#include <utility>
#include <vector>

#include <vamp/collision/environment.hh>
#include <vamp/vector.hh>

namespace vamp_node {

// VAMP calls this model for every sample along every searched edge. The base
// model retains SIMD self-collision checks; the request adds frozen-scene,
// gripper and attached-object checks for each lane.
template <typename Base>
struct PayloadCheckedRobot : Base {
    inline static thread_local std::function<bool(const std::vector<float> &)> validator;

    template <std::size_t width>
    static bool fkcc(
        const vamp::collision::Environment<vamp::FloatVector<width>> &environment,
        const typename Base::template ConfigurationBlock<width> &configuration) noexcept {
        if (!Base::template fkcc<width>(environment, configuration) || !validator) return false;
        for (std::size_t lane = 0; lane < width; ++lane) {
            std::vector<float> joints(Base::dimension);
            for (std::size_t joint = 0; joint < Base::dimension; ++joint) {
                joints[joint] = configuration[joint].to_array()[lane];
            }
            if (!validator(joints)) return false;
        }
        return true;
    }

    class ScopedValidator {
    public:
        explicit ScopedValidator(std::function<bool(const std::vector<float> &)> callback)
            : previous_(std::move(validator)) { validator = std::move(callback); }
        ~ScopedValidator() { validator = std::move(previous_); }
        ScopedValidator(const ScopedValidator &) = delete;
        ScopedValidator &operator=(const ScopedValidator &) = delete;
    private:
        std::function<bool(const std::vector<float> &)> previous_;
    };
};

}  // namespace vamp_node
