#include <array>
#include <gtest/gtest.h>
#include "vamp_node/payload_checked_robot.hpp"

struct TestRobot {
    static constexpr std::size_t dimension = 6;
    inline static bool base_safe = true;
    template <std::size_t width>
    using ConfigurationBlock = std::array<vamp::FloatVector<width>, dimension>;
    template <std::size_t width>
    static bool fkcc(const vamp::collision::Environment<vamp::FloatVector<width>> &,
        const ConfigurationBlock<width> &) { return base_safe; }
};
using Checked = vamp_node::PayloadCheckedRobot<TestRobot>;

TEST(PayloadRobot, RejectsMissingValidatorAndRestoresPreviousScope) {
    vamp::collision::Environment<vamp::FloatVector<4>> environment;
    TestRobot::ConfigurationBlock<4> configuration{};
    EXPECT_FALSE(Checked::fkcc<4>(environment, configuration));
    {
        Checked::ScopedValidator valid([](const auto &) { return true; });
        EXPECT_TRUE(Checked::fkcc<4>(environment, configuration));
        {
            Checked::ScopedValidator blocked([](const auto &) { return false; });
            EXPECT_FALSE(Checked::fkcc<4>(environment, configuration));
        }
        EXPECT_TRUE(Checked::fkcc<4>(environment, configuration));
    }
    EXPECT_FALSE(Checked::fkcc<4>(environment, configuration));
}

TEST(PayloadRobot, RejectsPayloadCollisionAtIntermediateLane) {
    vamp::collision::Environment<vamp::FloatVector<4>> environment;
    TestRobot::ConfigurationBlock<4> configuration{};
    const std::array<float, 4> values{0.0F, 0.3F, 0.6F, 1.0F};
    configuration[0] = vamp::FloatVector<4>(values.data());
    Checked::ScopedValidator callback([](const auto &joints) {
        return joints[0] < 0.5F || joints[0] > 0.7F;
    });
    EXPECT_FALSE(Checked::fkcc<4>(environment, configuration));
}

TEST(PayloadRobot, RetainsBaseSelfCollisionRejection) {
    vamp::collision::Environment<vamp::FloatVector<4>> environment;
    TestRobot::ConfigurationBlock<4> configuration{};
    Checked::ScopedValidator callback([](const auto &) { return true; });
    TestRobot::base_safe = false;
    EXPECT_FALSE(Checked::fkcc<4>(environment, configuration));
    TestRobot::base_safe = true;
}
