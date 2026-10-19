#include <array>
#include <gtest/gtest.h>
#include "vamp_node/payload_checked_robot.hpp"
#include <vamp/robots/piper.hh>

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

TEST(PiperLimits, SamplingScaleIncludesAllConfiguredJointBounds) {
    using Robot = vamp::robots::Piper;
    const auto original_a = Robot::s_a;
    const auto original_m = Robot::s_m;
    const auto original_d = Robot::d_m;
    const std::array<float, 6> lower = {-2.65F, -.035F, -3.0F, -1.78F, -1.26F, -2.13F};
    const std::array<float, 6> upper = {2.65F, 3.18F, .035F, 1.78F, 1.26F, 2.13F};
    Robot::configure_joint_limits(lower, upper);
    for (std::size_t joint = 0; joint < 6; ++joint) {
        EXPECT_FLOAT_EQ(Robot::s_a[joint], lower[joint]);
        EXPECT_NEAR(Robot::s_a[joint]+Robot::s_m[joint], upper[joint], 1.e-6F);
        EXPECT_NEAR(Robot::d_m[joint]*Robot::s_m[joint], 1.F, 1.e-6F);
    }
    Robot::s_a = original_a;
    Robot::s_m = original_m;
    Robot::d_m = original_d;
}

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
