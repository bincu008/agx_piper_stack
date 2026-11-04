#include <gtest/gtest.h>
#include "vamp_node/observe_view.hpp"

namespace {
Eigen::Isometry3f fk(const std::vector<float> &q) {
    auto pose = Eigen::Isometry3f::Identity();
    pose.translation() = Eigen::Vector3f(q[0],q[1],q[2]);
    return pose;
}
bool solve(const Eigen::Isometry3f &pose,const std::vector<float> &,float,
           std::vector<float> &q) {
    q = {pose.translation().x(),pose.translation().y(),pose.translation().z(),0,0,0};
    return true;
}
}

TEST(ObserveView, CameraAxesAreRelativeAndRotationOrderIsExplicit) {
    auto camera = Eigen::Isometry3f::Identity();
    camera.linear()=Eigen::AngleAxisf(M_PI/2,Eigen::Vector3f::UnitZ()).toRotationMatrix();
    const std::array<float,6> delta{{.02F,-.02F,0,0,0,.1F}};
    auto target=vamp_node::relativeCameraView(camera,delta,1.F);
    EXPECT_NEAR(target.translation().x(),.02F,1.e-6);
    EXPECT_NEAR(target.translation().y(),.02F,1.e-6);
    const Eigen::Matrix3f expected=camera.linear()*
        Eigen::AngleAxisf(.1F,Eigen::Vector3f::UnitZ()).toRotationMatrix();
    EXPECT_LT((target.linear()-expected).norm(),1.e-6);
}

TEST(ObserveView, SafeTargetReturnsNewJointsAndMeasuredEndpoint) {
    const auto identity=Eigen::Isometry3f::Identity();
    auto result=vamp_node::calculateObserveView(std::vector<float>(6,0),identity,identity,
        std::array<float,6>{{.02F,-.02F,0,0,0,0}},solve,
        [](const auto &) {return true;},fk);
    EXPECT_TRUE(result.moved); EXPECT_TRUE(result.endpoint_safe);
    EXPECT_EQ(result.status,"EXACT");
    EXPECT_NEAR(result.joints[0],.02F,1.e-6);
    EXPECT_NEAR(result.joints[1],-.02F,1.e-6);
}

TEST(ObserveView, CollisionReturnsExactCurrentJointsWithoutPartialDetour) {
    const auto identity=Eigen::Isometry3f::Identity();
    const std::vector<float> current(6,0);
    int checked=0;
    auto result=vamp_node::calculateObserveView(current,identity,identity,
        std::array<float,6>{{.02F,0,0,0,0,0}},solve,
        [&](const auto &) {++checked;return false;},fk);
    EXPECT_EQ(result.status,"TARGET_COLLISION"); EXPECT_EQ(result.joints,current);
    EXPECT_FALSE(result.moved); EXPECT_FALSE(result.endpoint_safe); EXPECT_EQ(checked,1);
}

TEST(ObserveView, IKExhaustionIsAnUnchangedDecisionNotSafeOrSuccessfulIK) {
    const auto identity=Eigen::Isometry3f::Identity();
    const std::vector<float> current(6,0);
    auto result=vamp_node::calculateObserveView(current,identity,identity,
        std::array<float,6>{{.02F,0,0,0,0,0}},
        [](const auto &,const auto &,float,auto &) {return false;},
        [](const auto &) {ADD_FAILURE();return true;},fk);
    EXPECT_EQ(result.status,"NO_REACHABLE_CHANGE"); EXPECT_EQ(result.joints,current);
    EXPECT_EQ(result.tested,6U); EXPECT_FALSE(result.endpoint_safe);
}

TEST(ObserveView, LooseOrientationAndSmallerReachAreBoundedAlternatives) {
    const auto identity=Eigen::Isometry3f::Identity();
    auto result=vamp_node::calculateObserveView(std::vector<float>(6,0),identity,identity,
        std::array<float,6>{{.02F,0,0,0,0,0}},
        [](const auto &pose,const auto &seed,float angle,auto &q) {
            return angle>.1F && solve(pose,seed,angle,q);
        },[](const auto &) {return true;},fk);
    EXPECT_EQ(result.status,"RELAXED"); EXPECT_TRUE(result.moved);
    result=vamp_node::calculateObserveView(std::vector<float>(6,0),identity,identity,
        std::array<float,6>{{.02F,0,0,0,0,0}},
        [](const auto &pose,const auto &seed,float angle,auto &q) {
            return pose.translation().norm()<.015F && solve(pose,seed,angle,q);
        },[](const auto &) {return true;},fk);
    EXPECT_EQ(result.status,"PARTIAL"); EXPECT_NEAR(result.fraction,.5F,1.e-6);
}

TEST(ObserveView, ZeroDeltaMakesNoSolverOrSafetyClaim) {
    const auto identity=Eigen::Isometry3f::Identity();
    auto result=vamp_node::calculateObserveView(std::vector<float>(6,0),identity,identity,
        std::array<float,6>{},
        [](const auto &,const auto &,float,auto &) {ADD_FAILURE();return false;},
        [](const auto &) {ADD_FAILURE();return true;},fk);
    EXPECT_EQ(result.status,"NO_CHANGE"); EXPECT_FALSE(result.endpoint_safe);
}

TEST(ObserveView, UnchangedLooseSolutionDoesNotHideReachablePartialRotation) {
    const auto identity=Eigen::Isometry3f::Identity();
    auto result=vamp_node::calculateObserveView(std::vector<float>(6,0),identity,identity,
        std::array<float,6>{{0,0,0,0,0,.1F}},
        [](const auto &pose,const auto &,float tolerance,auto &q) {
            q=std::vector<float>(6,0);
            if (tolerance>.1F) return true;  // loose solve may accept unchanged state
            const float angle=Eigen::AngleAxisf(pose.linear()).angle();
            if (angle>.06F) return false;
            q[3]=angle;return true;
        },[](const auto &) {return true;},
        [](const auto &q) {
            auto pose=Eigen::Isometry3f::Identity();
            pose.linear()=Eigen::AngleAxisf(q[3],Eigen::Vector3f::UnitZ()).toRotationMatrix();
            return pose;
        });
    EXPECT_EQ(result.status,"PARTIAL"); EXPECT_NEAR(result.fraction,.5F,1.e-6);
    EXPECT_TRUE(result.endpoint_safe); EXPECT_TRUE(result.moved);
}
