#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <vamp/robots/piper.hh>

#include "vamp_node/vamp_utils.hpp"

// --- Empty Hand Withdrawal Tests ---
TEST(EmptyWithdrawal, ContactCanShiftInsideFrozenRegionWhileSeparating) {
    vamp_node::WithdrawalContacts c;
    EXPECT_TRUE(c.seed(7, .004F));
    EXPECT_TRUE(c.seed(8, -.001F));
    c.begin();
    EXPECT_TRUE(c.permits(7, .003F, .010F, false));
    EXPECT_TRUE(c.permits(8, .0005F, .010F, false));
    EXPECT_TRUE(c.separating(.010F));
    EXPECT_FALSE(c.permits(9, .0001F, .010F, false));
    EXPECT_FALSE(c.permits(7, .007F, .010F, false));
    EXPECT_FALSE(c.permits(7, .001F, .031F, false));
    EXPECT_FALSE(c.permits(7, .001F, .010F, true));
}

TEST(EmptyWithdrawal, RejectsDeepAndInvalidContacts) {
    vamp_node::WithdrawalContacts c;
    EXPECT_FALSE(c.seed(0, .031F));
    EXPECT_FALSE(c.seed(0, -.0021F));
    EXPECT_FALSE(c.seed(0, std::numeric_limits<float>::quiet_NaN()));
    EXPECT_EQ(c.size(), 0u);
    EXPECT_TRUE(c.seed(4, .016F));
    c.begin();
    EXPECT_FALSE(c.permits(4, .001F, -.001F, false));
    EXPECT_FALSE(c.permits(4, -.001F, .001F, false));
}

TEST(EmptyWithdrawal, TangentialContactKeepsFixedBudgetAndMustClearAtEndpoint) {
    vamp_node::WithdrawalContacts c;
    EXPECT_TRUE(c.seed(2, .004F));
    for (int step = 0; step < 25; ++step) {
        c.begin();
        EXPECT_TRUE(c.permits(2, .002F, .001F * step, false));
        EXPECT_TRUE(c.separating(.001F * step));
        EXPECT_FALSE(c.permits(3, .001F, .001F * step, false));
    }
    c.begin();
    EXPECT_TRUE(c.permits(2, .004F, .020F, false));
    EXPECT_TRUE(c.separating(.020F));
    c.begin();
    EXPECT_TRUE(c.permits(2, .00396F, .002F, false));
    EXPECT_TRUE(c.separating(.002F));
    c.begin();
    EXPECT_TRUE(c.permits(2, .0045F, .002F, false));
    EXPECT_FALSE(c.separating(.002F));
    EXPECT_FALSE(c.permits(2, .001F, .020F, true));
    EXPECT_FALSE(c.permits(2, .001F, .031F, false));
}

TEST(EmptyWithdrawal, InitiallyClearHandGetsNoNewContactAllowance) {
    vamp_node::WithdrawalContacts c;
    EXPECT_TRUE(c.seed(2, -.001F));
    c.begin();
    EXPECT_TRUE(c.permits(2, .0001F, .001F, false));
    EXPECT_FALSE(c.separating(.001F));
    c.begin();
    EXPECT_TRUE(c.separating(.050F));
}

// --- Payload Checked Robot Tests ---
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
        EXPECT_NEAR(Robot::s_a[joint] + Robot::s_m[joint], upper[joint], 1.e-6F);
        EXPECT_NEAR(Robot::d_m[joint] * Robot::s_m[joint], 1.F, 1.e-6F);
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

// --- Placement Ranking Tests ---
TEST(PlacementRanking, PackingPreferenceWinsOverTravelCost)
{
    EXPECT_TRUE(vamp_node::preferPlacement(-0.001F, 0.2F, -0.002F, 0.01F));
    EXPECT_FALSE(vamp_node::preferPlacement(-0.002F, 0.01F, -0.001F, 0.2F));
}

TEST(PlacementRanking, EqualPackingUsesPoseAndTravelCost)
{
    EXPECT_TRUE(vamp_node::preferPlacement(-0.001F, 0.01F, -0.001F, 0.2F));
    EXPECT_FALSE(vamp_node::preferPlacement(-0.001F, 0.2F, -0.001F, 0.01F));
}

TEST(PlacementRanking, RejectsNonfiniteInput)
{
    EXPECT_FALSE(vamp_node::preferPlacement(
        std::numeric_limits<float>::quiet_NaN(), 0.01F, -0.001F, 0.2F));
}

// --- Planar Body Mask Tests ---
TEST(PlanarBodyMask, PreservesPlateAndRaisedRimRejectsLowerStand) {
    std::vector<Eigen::Vector3f> points;
    for (int i = 0; i < 25; ++i) {
        for (int j = 0; j < 25; ++j) points.emplace_back(i * .005f, j * .005f, .06f);
    }
    const auto base = points.size();
    points.emplace_back(.12f, .06f, .075f);
    for (int i = 0; i < 30; ++i) points.emplace_back(.06f, .06f, .01f + i * .0005f);
    const auto keep = vamp_node::planar_body_mask(points);
    for (size_t i = 0; i < keep.size(); ++i) EXPECT_EQ(keep[i], i <= base);
}

TEST(PlanarBodyMask, InsufficientEvidencePreservesInput) {
    std::vector<Eigen::Vector3f> points(10, Eigen::Vector3f::Zero());
    for (bool keep : vamp_node::planar_body_mask(points)) EXPECT_TRUE(keep);
}

