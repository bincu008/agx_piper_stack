#include <gtest/gtest.h>
#include "vamp_node/placement_geometry.hpp"

TEST(PadAlignment, RejectsSlopedPlateExtentPair) {
    const Eigen::Vector3f closing(0,1,0);
    EXPECT_TRUE(vamp_node::padNormalsAligned(closing, -closing, closing, 0.9F));
    EXPECT_FALSE(vamp_node::padNormalsAligned(
        Eigen::Vector3f(0,0.6F,0.8F), Eigen::Vector3f(0,-0.6F,0.8F), closing, 0.9F));
    EXPECT_FALSE(vamp_node::padNormalsAligned(closing, closing, closing, 0.9F));
}
TEST(AttachedObject, PlacedObjectBlocksEvenAtFinalContact) {
    vamp_node::AttachedObjectScene scene;
    scene.add(Eigen::Vector3f(0,0,0), false);
    EXPECT_FALSE(scene.clear({Eigen::Vector3f(0,0,0)}, Eigen::Isometry3f::Identity(), 0, true));
}
TEST(AttachedObject, OnlyBottomMayTouchNamedSupportAtFinalContact) {
    vamp_node::AttachedObjectScene scene;
    scene.add(Eigen::Vector3f(0,0,0), true);
    const std::vector<Eigen::Vector3f> points{Eigen::Vector3f(0,0,0)};
    EXPECT_TRUE(scene.clear(points, Eigen::Isometry3f::Identity(), 0, true));
    EXPECT_FALSE(scene.clear(points, Eigen::Isometry3f::Identity(), 0, false));
    EXPECT_FALSE(scene.clear(points, Eigen::Isometry3f::Identity(), -0.01F, true));
}
TEST(AttachedObject, FixedAttachmentMovesCloudWithGripper) {
    vamp_node::AttachedObjectScene scene;
    scene.add(Eigen::Vector3f(0.1F,0,0.05F), false);
    Eigen::Isometry3f delta = Eigen::Isometry3f::Identity();
    delta.translation().x() = 0.1F;
    const std::vector<Eigen::Vector3f> points{Eigen::Vector3f(0,0,0.05F)};
    EXPECT_TRUE(scene.clear(points, Eigen::Isometry3f::Identity(), 0, false));
    EXPECT_FALSE(scene.clear(points, delta, 0, false));
}
TEST(PlacementGripper, RealOpeningControlsFingerCollision) {
    const std::vector<Eigen::Vector3f> obstacle{Eigen::Vector3f(0,0.025F,0.10F)};
    EXPECT_FALSE(vamp_node::fingerBodiesClear(obstacle,
        Eigen::Isometry3f::Identity(), 0.04F, -0.0615F, 0.01F, 0.02F, 0.07F));
    EXPECT_TRUE(vamp_node::fingerBodiesClear(obstacle,
        Eigen::Isometry3f::Identity(), 0.08F, -0.0615F, 0.01F, 0.02F, 0.07F));
}

#include <random>
#include "vamp_node/placement_point_index.hpp"

TEST(PlacementIndex, ExactSphereQueriesMatchBruteForceAcrossCellBoundaries) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> xyz(-0.15F,0.15F);
    std::vector<Eigen::Vector3f> points;
    for (int i=0;i<1000;++i) points.emplace_back(xyz(rng),xyz(rng),xyz(rng));
    vamp_node::PlacementPointIndex index(points);
    for (int i=0;i<100;++i) {
        Eigen::Vector3f center(xyz(rng),xyz(rng),xyz(rng));
        const float radius=0.001F+0.001F*i;
        bool clear=true;
        for (const auto &point:points) if ((point-center).norm()<radius) clear=false;
        EXPECT_EQ(index.sphereClear(center,radius),clear);
    }
}
TEST(PlacementIndex, RotatedFingerQueriesRetainExactGeometry) {
    std::mt19937 rng(17);
    std::uniform_real_distribution<float> xyz(-0.20F,0.20F);
    std::vector<Eigen::Vector3f> points;
    for (int i=0;i<1000;++i) points.emplace_back(xyz(rng),xyz(rng),xyz(rng));
    vamp_node::PlacementPointIndex index(points);
    for (int i=0;i<20;++i) {
        Eigen::Isometry3f pose=Eigen::Isometry3f::Identity();
        pose.linear()=Eigen::AngleAxisf(0.2F*i,Eigen::Vector3f(1,2,3).normalized()).toRotationMatrix();
        pose.translation()=Eigen::Vector3f(xyz(rng),xyz(rng),xyz(rng));
        EXPECT_EQ(index.fingersClear(pose,0.08F,-0.0615F,0.01F,0.02F,0.07F),
            vamp_node::fingerBodiesClear(points,pose,0.08F,-0.0615F,0.01F,0.02F,0.07F));
    }
}
TEST(AttachedObject, TerminalSupportAllowanceNeverPermitsDeepPenetration) {
    vamp_node::AttachedObjectScene scene;
    scene.add(Eigen::Vector3f::Zero(),true);
    Eigen::Isometry3f delta=Eigen::Isometry3f::Identity();
    delta.translation().z()=-0.002F;
    EXPECT_FALSE(scene.clear({Eigen::Vector3f::Zero()},delta,0,true));
    delta.translation().z()=0.002F;
    EXPECT_TRUE(scene.clear({Eigen::Vector3f::Zero()},delta,0,true));
    EXPECT_FALSE(scene.clear({Eigen::Vector3f::Zero()},delta,0,false));
}

#include "vamp_node/cartesian_grasp_approach.hpp"
TEST(PlacementDescent, TerminalBandHandlesBoundedIkErrorWithoutExemptingOtherObjects) {
    Eigen::Isometry3f start=Eigen::Isometry3f::Identity(),goal=start;
    start.translation().z()=0.01F;
    goal.translation().z()=0.002F;
    const auto solve=[](const Eigen::Isometry3f &pose,const std::vector<float> &seed,
        std::vector<float> &solution) {
        solution=seed;
        solution[0]=pose.translation().z()-0.001F;
        return true;
    };
    for (bool named_support:{true,false}) {
        vamp_node::AttachedObjectScene scene;
        scene.add(Eigen::Vector3f::Zero(),named_support);
        auto path=vamp_node::planCartesianApproach(start,goal,{0.01F,0,0,0,0,0},
            0.002F,0.05F,solve,[&](const auto &joints,size_t,bool) {
                Eigen::Isometry3f delta=Eigen::Isometry3f::Identity();
                delta.translation().z()=joints[0];
                return scene.clear({Eigen::Vector3f::Zero()},delta,0,
                    std::abs(joints[0]-goal.translation().z())<=0.006F);
            });
        EXPECT_EQ(path.success,named_support);
        if (path.success) EXPECT_EQ(path.waypoint_joints.size(),4U);
        else EXPECT_TRUE(path.waypoint_joints.empty());
    }
}

TEST(PlacementHover, IncludesMinimumForNonMultipleConfiguredOffsets) {
    const auto distances = vamp_node::placementHoverDistances(0.05F, true);
    ASSERT_EQ(distances.size(), 3U);
    EXPECT_FLOAT_EQ(distances.front(), 0.05F);
    EXPECT_FLOAT_EQ(distances.back(), 0.02F);
    EXPECT_EQ(vamp_node::placementHoverDistances(0.05F, false),
        std::vector<float>({0.05F}));
    EXPECT_TRUE(vamp_node::placementHoverDistances(-0.01F, true).empty());
}

TEST(AttachedObject, InitialSupportIncludesContactNeighborhoodAtFootprintEdge) {
    const Eigen::Vector3f minimum(.2F,.25F,0), maximum(.28F,.33F,.08F);
    const Eigen::Vector3f boundary(.1998F,.255F,-.001F);
    EXPECT_TRUE(vamp_node::initialLiftSupportSample(boundary,minimum,maximum,0));
    EXPECT_FALSE(vamp_node::initialLiftSupportSample(
        Eigen::Vector3f(.19F,.255F,0),minimum,maximum,0));
    EXPECT_FALSE(vamp_node::initialLiftSupportSample(
        Eigen::Vector3f(.21F,.26F,.008F),minimum,maximum,0));
    vamp_node::AttachedObjectScene scene;
    scene.add(Eigen::Vector3f(.21F,.26F,.0015F),true);
    // Naming a support never permits excessive initial penetration.
    EXPECT_FALSE(scene.clear({Eigen::Vector3f(.21F,.26F,0)},
        Eigen::Isometry3f::Identity(),0,true));
}
