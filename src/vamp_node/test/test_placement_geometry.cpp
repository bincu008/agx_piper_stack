#include <gtest/gtest.h>
#include "vamp_node/placement_geometry.hpp"

TEST(AttachedObject, NoiseBudgetCountsObstaclePointsNotRepeatedHeldContacts) {
    vamp_node::AttachedObjectScene scene;
    for (int index = 0; index < 21; ++index)
        scene.add(Eigen::Vector3f(0.00001F * index,0,0), false);
    const std::vector<Eigen::Vector3f> held(100,Eigen::Vector3f::Zero());
    vamp_node::AttachedObjectScene::NoiseBudget budget;
    budget.maximum_points = 32;
    budget.maximum_voxels = 8;
    EXPECT_FALSE(scene.clear(held,Eigen::Isometry3f::Identity(),-1,false));
    EXPECT_TRUE(scene.clear(held,Eigen::Isometry3f::Identity(),-1,false,
        nullptr,nullptr,0.001F,nullptr,0,&budget));
    EXPECT_EQ(budget.points.size(),21u);
    EXPECT_TRUE(scene.clear(held,Eigen::Isometry3f::Identity(),-1,false,
        nullptr,nullptr,0.001F,nullptr,0,&budget));
    EXPECT_EQ(budget.points.size(),21u);
}

TEST(AttachedObject, NoiseBudgetRejectsDenseContactsButAllowsSeparatedSparseContacts) {
    vamp_node::AttachedObjectScene dense, sparse_wall;
    for (int index = 0; index < 33; ++index)
        dense.add(Eigen::Vector3f(0.00001F * index,0,0), false);
    vamp_node::AttachedObjectScene::NoiseBudget budget;
    budget.maximum_points = 32;
    budget.maximum_voxels = 8;
    EXPECT_FALSE(dense.clear({Eigen::Vector3f::Zero()},Eigen::Isometry3f::Identity(),
        -1,false,nullptr,nullptr,0.001F,nullptr,0,&budget));
    EXPECT_TRUE(budget.points.empty());
    sparse_wall.add(Eigen::Vector3f::Zero(),false);
    sparse_wall.add(Eigen::Vector3f(0.03F,0,0),false);
    EXPECT_TRUE(sparse_wall.clear({Eigen::Vector3f::Zero(),Eigen::Vector3f(0.03F,0,0)},
        Eigen::Isometry3f::Identity(),-1,false,nullptr,nullptr,0.001F,nullptr,0,&budget));
    EXPECT_EQ(budget.points.size(),2u);
}

TEST(AttachedObject, NoiseBudgetAccumulatesAcrossWaypointsWithoutCommittingFailure) {
    vamp_node::AttachedObjectScene scene;
    scene.add(Eigen::Vector3f::Zero(),false);
    scene.add(Eigen::Vector3f(0.008F,0,0),false);
    vamp_node::AttachedObjectScene::NoiseBudget budget;
    budget.maximum_points = 1;
    budget.maximum_voxels = 8;
    EXPECT_TRUE(scene.clear({Eigen::Vector3f::Zero()},Eigen::Isometry3f::Identity(),
        -1,false,nullptr,nullptr,0.001F,nullptr,0,&budget));
    auto delta = Eigen::Isometry3f::Identity();
    delta.translation().x() = 0.008F;
    EXPECT_FALSE(scene.clear({Eigen::Vector3f::Zero()},delta,
        -1,false,nullptr,nullptr,0.001F,nullptr,0,&budget));
    EXPECT_EQ(budget.points.size(),1u);
    EXPECT_EQ(budget.observed_points,2u);
}

TEST(AttachedObject, NoiseBudgetAlsoBoundsOccupiedVoxels) {
    vamp_node::AttachedObjectScene scene;
    scene.add(Eigen::Vector3f::Zero(),false);
    scene.add(Eigen::Vector3f(0.008F,0,0),false);
    vamp_node::AttachedObjectScene::NoiseBudget budget;
    budget.maximum_points = 32;
    budget.maximum_voxels = 1;
    EXPECT_FALSE(scene.clear({Eigen::Vector3f::Zero(),Eigen::Vector3f(0.008F,0,0)},
        Eigen::Isometry3f::Identity(),-1,true,nullptr,nullptr,0.001F,nullptr,0,&budget));
    EXPECT_TRUE(budget.points.empty());
}

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
TEST(AttachedObject, SeparatingDeparturePreservesExistingClearanceOnly) {
    vamp_node::AttachedObjectScene scene;
    scene.add(Eigen::Vector3f::Zero(), false);
    const std::vector<Eigen::Vector3f> source{Eigen::Vector3f(0.001F,0,0)};
    const auto contacts=scene.initialContacts(source);
    Eigen::Isometry3f delta=Eigen::Isometry3f::Identity();
    EXPECT_FALSE(scene.clear(source,delta,-1,false));
    EXPECT_TRUE(scene.clear(source,delta,-1,false,nullptr,nullptr,0.001F,&contacts));
    delta.translation().x()=0.001F;
    EXPECT_TRUE(scene.clear(source,delta,-1,false,nullptr,nullptr,0.001F,&contacts));
    EXPECT_FALSE(scene.clear(source,delta,-1,false));
    delta.translation().x()=-0.0005F;
    EXPECT_FALSE(scene.clear(source,delta,-1,false,nullptr,nullptr,0.001F,&contacts));
}
TEST(AttachedObject, SeparatingDepartureRejectsNewOrCoincidentContacts) {
    vamp_node::AttachedObjectScene scene;
    scene.add(Eigen::Vector3f::Zero(), false);
    scene.add(Eigen::Vector3f(0.006F,0,0),false);
    const std::vector<Eigen::Vector3f> source{Eigen::Vector3f(0.001F,0,0)};
    const auto contacts=scene.initialContacts(source);
    Eigen::Isometry3f delta=Eigen::Isometry3f::Identity();
    delta.translation().x()=0.004F;
    EXPECT_FALSE(scene.clear(source,delta,-1,false,nullptr,nullptr,0.001F,&contacts));
    const std::vector<Eigen::Vector3f> coincident{Eigen::Vector3f::Zero()};
    const auto overlap=scene.initialContacts(coincident);
    EXPECT_FALSE(scene.clear(coincident,Eigen::Isometry3f::Identity(),-1,false,
                            nullptr,nullptr,0.001F,&overlap));
}
TEST(AttachedObject, OnlyBottomMayTouchNamedSupportAtFinalContact) {
    vamp_node::AttachedObjectScene scene;
    scene.add(Eigen::Vector3f(0,0,0), true);
    const std::vector<Eigen::Vector3f> points{Eigen::Vector3f(0,0,0)};
    EXPECT_TRUE(scene.clear(points, Eigen::Isometry3f::Identity(), 0, true));
    EXPECT_FALSE(scene.clear(points, Eigen::Isometry3f::Identity(), 0, false));
    EXPECT_FALSE(scene.clear(points, Eigen::Isometry3f::Identity(), -0.01F, true));
}
TEST(AttachedObject, SourceSupportUncertaintyEndsBeforeTransferAndProtectsOtherObjects) {
    EXPECT_FLOAT_EQ(vamp_node::initialLiftSupportBand(0), 0.004F);
    EXPECT_FLOAT_EQ(vamp_node::initialLiftContactTravel(0), 0.006F);
    EXPECT_NEAR(vamp_node::initialLiftContactTravel(0.002F), 0.015F, 1e-7F);
    EXPECT_NEAR(vamp_node::initialLiftSupportBand(0.02F), 0.012F, 1e-7F);
    vamp_node::AttachedObjectScene table, other_object;
    table.add(Eigen::Vector3f(0,0,0.006F), true);
    other_object.add(Eigen::Vector3f(0,0,0.006F), false);
    const std::vector<Eigen::Vector3f> source{Eigen::Vector3f::Zero()};
    const std::vector<uint8_t> bottom{2};
    Eigen::Isometry3f delta = Eigen::Isometry3f::Identity();
    delta.translation().z() = 0.004F;
    EXPECT_TRUE(table.clear(source,delta,0,true,nullptr,&bottom,0.003F));
    EXPECT_FALSE(other_object.clear(source,delta,0,true,nullptr,&bottom,0.003F));
    EXPECT_FALSE(table.clear(source,delta,0,false,nullptr,&bottom,0.003F));
    EXPECT_FALSE(table.clear(source,delta,0,true));  // Final placement's 1 mm limit.
}
TEST(AttachedObject, DepartureAllowanceIsBoundedAndDoesNotChangeOrdinaryChecks) {
    vamp_node::AttachedObjectScene scene;
    scene.add(Eigen::Vector3f::Zero(), false);
    const std::vector<Eigen::Vector3f> source{Eigen::Vector3f(0.002F,0,0)};
    const auto contacts = scene.initialContacts(source);
    Eigen::Isometry3f delta = Eigen::Isometry3f::Identity();
    delta.translation().x() = -0.0008F;
    EXPECT_TRUE(scene.clear(source,delta,-1,false,nullptr,nullptr,0.001F,&contacts,0.001F));
    EXPECT_FALSE(scene.clear(source,delta,-1,false));
    EXPECT_FALSE(scene.clear(source,delta,-1,true));
    delta.translation().x() = -0.0015F;
    EXPECT_FALSE(scene.clear(source,delta,-1,false,nullptr,nullptr,0.001F,&contacts,0.001F));
    delta.translation().x() = 0.0061F;
    delta.translation().z() = -0.0061F;
    scene.add(Eigen::Vector3f(0.008F,0,-0.006F), false);
    EXPECT_FALSE(scene.clear(source,delta,-1,false,nullptr,nullptr,0.001F,&contacts,0.002F));
}
TEST(AttachedObject, DepartureAllowanceDoesNotAdmitNewNeighborContacts) {
    vamp_node::AttachedObjectScene scene;
    scene.add(Eigen::Vector3f::Zero(), false);
    scene.add(Eigen::Vector3f(0.006F,0,0), false);
    const std::vector<Eigen::Vector3f> source{Eigen::Vector3f(0.002F,0,0)};
    const auto contacts = scene.initialContacts(source);
    Eigen::Isometry3f delta = Eigen::Isometry3f::Identity();
    delta.translation().x() = 0.003F;
    EXPECT_FALSE(scene.clear(source,delta,-1,false,nullptr,nullptr,0.001F,&contacts,0.002F));
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

TEST(AttachedObject, TightNeighborhoodMatchesExactContactRules) {
    std::mt19937 random(723);
    std::uniform_real_distribution<float> xyz(-0.025F, 0.025F);
    std::vector<vamp_node::AttachedObjectScene::Sample> samples;
    vamp_node::AttachedObjectScene scene;
    for (int i = 0; i < 1000; ++i) {
        samples.push_back({Eigen::Vector3f(xyz(random), xyz(random), xyz(random)), i % 2 == 0});
        scene.add(samples.back().point, samples.back().support);
    }
    for (int i = 0; i < 3000; ++i) {
        Eigen::Vector3f source(xyz(random), xyz(random), xyz(random));
        Eigen::Isometry3f delta = Eigen::Isometry3f::Identity();
        delta.translation() = Eigen::Vector3f(xyz(random), xyz(random), xyz(random));
        const Eigen::Vector3f point = delta * source;
        const bool final_contact = i % 2 == 0;
        const float bottom = source.z() + (i % 3 == 0 ? -0.01F : 0.0F);
        bool expected = true;
        for (const auto &sample : samples) {
            if ((point - sample.point).squaredNorm() > 9e-6F) continue;
            if (final_contact && sample.support && source.z() <= bottom + 0.003F &&
                point.z() >= sample.point.z() - 0.001F) continue;
            expected = false;
            break;
        }
        EXPECT_EQ(scene.clear({source}, delta, bottom, final_contact), expected);
    }
}

TEST(AttachedObject, TightNeighborhoodRetainsThresholdAndCellBoundaryContacts) {
    for (float coordinate : {-0.01F, -0.005F, 0.0F, 0.005F, 0.01F}) {
        const Eigen::Vector3f source(coordinate, coordinate, coordinate);
        for (float distance : {-0.003001F, -0.003F, -0.002999F, 0.0F,
                               0.002999F, 0.003F, 0.003001F}) {
            const Eigen::Vector3f obstacle = source + Eigen::Vector3f(distance, 0, 0);
            vamp_node::AttachedObjectScene scene;
            scene.add(obstacle, false);
            EXPECT_EQ(scene.clear({source}, Eigen::Isometry3f::Identity(), source.z(), false),
                (source - obstacle).squaredNorm() > 9e-6F);
        }
    }
}

#include "vamp_node/layout_scene_cache.hpp"

TEST(LayoutSceneCache, ExactCloudAndSupportPolicyControlReuse) {
    sensor_msgs::msg::PointCloud2 cloud;
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(4, "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "z", 1, sensor_msgs::msg::PointField::FLOAT32,
        "support", 1, sensor_msgs::msg::PointField::UINT8);
    modifier.resize(1);
    sensor_msgs::PointCloud2Iterator<float> x(cloud, "x"), y(cloud, "y"), z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> support(cloud, "support");
    *x = *y = *z = 0.0F;
    *support = 1;
    vamp_node::LayoutSceneCache cache;
    auto first = cache.get(cloud, "edge");
    EXPECT_EQ(first, cache.get(cloud, "edge"));
    EXPECT_TRUE(first->attached.clear({Eigen::Vector3f::Zero()}, Eigen::Isometry3f::Identity(), 0, true));
    auto stack = cache.get(cloud, "stack");
    EXPECT_NE(stack, first);
    EXPECT_FALSE(stack->attached.clear({Eigen::Vector3f::Zero()}, Eigen::Isometry3f::Identity(), 0, true));
    *support = 2;
    auto changed = cache.get(cloud, "stack");
    EXPECT_NE(changed, stack);
    EXPECT_TRUE(changed->attached.clear({Eigen::Vector3f::Zero()}, Eigen::Isometry3f::Identity(), 0, true));
    auto automatic = cache.get(cloud, "auto");
    EXPECT_TRUE(automatic->attached.clear(
        {Eigen::Vector3f::Zero()}, Eigen::Isometry3f::Identity(), 0, true));
    // Support stays blocking before final bottom contact, and edge excludes tops.
    EXPECT_FALSE(automatic->attached.clear(
        {Eigen::Vector3f::Zero()}, Eigen::Isometry3f::Identity(), 0, false));
    EXPECT_FALSE(cache.get(cloud, "edge")->attached.clear(
        {Eigen::Vector3f::Zero()}, Eigen::Isometry3f::Identity(), 0, true));
    *support = 1;
    EXPECT_TRUE(cache.get(cloud, "auto")->attached.clear(
        {Eigen::Vector3f::Zero()}, Eigen::Isometry3f::Identity(), 0, true));
    *support = 3;
    auto protected_object = cache.get(cloud, "stack");
    EXPECT_EQ(protected_object->support_labels[0], 3);
    EXPECT_FALSE(protected_object->attached.clear(
        {Eigen::Vector3f::Zero()}, Eigen::Isometry3f::Identity(), 0, true));
    *x = 0.02F;
    auto moved = cache.get(cloud, "stack");
    EXPECT_NE(moved, changed);
    EXPECT_FLOAT_EQ(moved->points[0].x(), 0.02F);
    cloud.header.stamp.sec = 123;
    EXPECT_NE(cache.get(cloud, "stack"), moved);
    // Retained earlier snapshots remain immutable after all updates.
    EXPECT_FLOAT_EQ(first->points[0].x(), 0.0F);
}

#include "vamp_node/organized_xyz_reader.hpp"

TEST(OrganizedXYZReader, MatchesIteratorWithReorderedPaddedFieldsAndInvalidDepth) {
    sensor_msgs::msg::PointCloud2 cloud;
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(4, "z", 1, sensor_msgs::msg::PointField::FLOAT32,
        "rgb", 1, sensor_msgs::msg::PointField::FLOAT32,
        "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32);
    modifier.resize(600);
    cloud.width = 30; cloud.height = 20; cloud.row_step = cloud.width * cloud.point_step;
    sensor_msgs::PointCloud2Iterator<float> x(cloud, "x"), y(cloud, "y"), z(cloud, "z");
    for (int i = 0; i < 600; ++i, ++x, ++y, ++z) {
        *x = i * 0.001F; *y = -i * 0.001F; *z = i % 3 == 0 ? 0.04F : 0.2F;
        if (i % 17 == 0) *x = std::numeric_limits<float>::quiet_NaN();
    }
    for (const auto &frame : {"base_link", "camera_optical_frame"}) {
        cloud.header.frame_id = frame;
        vamp_node::OrganizedXYZReader reader(cloud);
        sensor_msgs::PointCloud2ConstIterator<float> rx(cloud, "x"), ry(cloud, "y"), rz(cloud, "z");
        for (int i = 0; i < 600; ++i, ++rx, ++ry, ++rz) {
            std::array<float, 3> point;
            const bool expected = std::isfinite(*rx) && std::isfinite(*ry) && std::isfinite(*rz) &&
                (frame == std::string("base_link") || *rz > 0.05F);
            EXPECT_EQ(reader.read(i % 30, i / 30, point), expected);
            if (expected) EXPECT_EQ(point, (std::array<float,3>{*rx, *ry, *rz}));
        }
        std::array<float, 3> point;
        EXPECT_FALSE(reader.read(-1, 0, point));
        EXPECT_FALSE(reader.read(30, 0, point));
        EXPECT_FALSE(reader.read(0, 20, point));
    }
    cloud.data.resize(1);
    std::array<float, 3> point;
    EXPECT_FALSE(vamp_node::OrganizedXYZReader(cloud).read(0, 0, point));
    cloud.fields.clear();
    EXPECT_FALSE(vamp_node::OrganizedXYZReader(cloud).read(0, 0, point));
}

#include "vamp_node/rim_seed.hpp"
TEST(RimSeed, RecoversSupportedRimBeyondThreeCentimeters) {
    std::vector<Eigen::Vector3f> points{Eigen::Vector3f(0.02F,0,0.03F)};
    for (int i = 0; i < 40; ++i)
        points.emplace_back(0.045F + i*0.00001F, 0, 0.03F);
    const auto seed = vamp_node::supportedRimSeed(points, Eigen::Vector3f::Zero(), 0.03F);
    ASSERT_TRUE(seed);
    EXPECT_GT(seed->x(), 0.04F);
}
TEST(RimSeed, RejectsUnsupportedOrDistantSamples) {
    EXPECT_FALSE(vamp_node::supportedRimSeed(
        {Eigen::Vector3f(0.045F,0,0.03F)}, Eigen::Vector3f::Zero(), 0.03F));
    std::vector<Eigen::Vector3f> points(40, Eigen::Vector3f(0.07F,0,0.03F));
    EXPECT_FALSE(vamp_node::supportedRimSeed(points, Eigen::Vector3f::Zero(), 0.03F));
}

#include "vamp_node/piper_finger_geometry.hpp"
TEST(PiperFingerGeometry, PreservesPalmAndNominalOpening) {
    const auto pose=Eigen::Isometry3f::Identity();
    const Eigen::Vector3f center(.01F,.04F,.12F);
    EXPECT_EQ(vamp_node::piperFingerSphereCenter(81,center,pose,.02F),center);
    EXPECT_EQ(vamp_node::piperFingerSphereCenter(95,center,pose,.07F),center);
}
TEST(PiperFingerGeometry, TranslatesBothDetailedFingerLinksAlongActualJawAxis) {
    Eigen::Isometry3f pose=Eigen::Isometry3f::Identity();
    pose.linear()=Eigen::AngleAxisf(1.5707963268F,Eigen::Vector3f::UnitZ()).toRotationMatrix();
    const Eigen::Vector3f center(.01F,.04F,.12F);
    const auto left=vamp_node::piperFingerSphereCenter(82,center,pose,.02F);
    const auto right=vamp_node::piperFingerSphereCenter(89,center,pose,.02F);
    EXPECT_NEAR((left-center).dot(pose.linear().col(1)),.025F,1e-6F);
    EXPECT_NEAR((right-center).dot(pose.linear().col(1)),-.025F,1e-6F);
    EXPECT_NEAR(left.z(),center.z(),1e-6F);
}

TEST(RimSeed, PreservesSeparatedSupportedContactRegions) {
    std::vector<Eigen::Vector3f> points;
    for (const float x : {-0.025F, 0.025F})
        for (const float y : {-0.025F, 0.025F})
            for (int i = 0; i < 40; ++i)
                points.emplace_back(x + i * 0.000001F, y, 0.03F);
    const auto seeds = vamp_node::supportedRimSeeds(
        points, Eigen::Vector3f::Zero(), 0.03F);
    ASSERT_EQ(seeds.size(), 4U);
    for (size_t i = 0; i < seeds.size(); ++i)
        for (size_t j = i + 1; j < seeds.size(); ++j)
            EXPECT_GE((seeds[i] - seeds[j]).norm(), 0.025F);
}

TEST(AttachedObject, ExplicitBottomLabelsAllowTiltedSupportOnly) {
    vamp_node::AttachedObjectScene scene;
    scene.add(Eigen::Vector3f(0.04F, 0, 0.004F), true);
    const std::vector<Eigen::Vector3f> points{Eigen::Vector3f(0.04F, 0, 0.004F)};
    const std::vector<uint8_t> bottom{1}, side{0};
    const auto identity = Eigen::Isometry3f::Identity();
    EXPECT_FALSE(scene.clear(points, identity, 0.0F, true));
    EXPECT_TRUE(scene.clear(points, identity, 0.0F, true, nullptr, &bottom));
    EXPECT_FALSE(scene.clear(points, identity, 0.0F, true, nullptr, &side));
    EXPECT_FALSE(scene.clear(points, identity, 0.0F, false, nullptr, &bottom));
    auto penetrating = Eigen::Isometry3f::Identity();
    penetrating.translation().z() -= 0.002F;
    EXPECT_FALSE(scene.clear(points, penetrating, 0.0F, true, nullptr, &bottom));
}

TEST(AttachedObject, InitialDepthNoiseAllowanceDoesNotChangeTerminalContact) {
    vamp_node::AttachedObjectScene scene;
    scene.add(Eigen::Vector3f(0, 0, 0.002F), true);
    const std::vector<Eigen::Vector3f> points{Eigen::Vector3f::Zero()};
    const std::vector<uint8_t> bottom{1};
    const auto identity = Eigen::Isometry3f::Identity();
    EXPECT_FALSE(scene.clear(points, identity, 0.0F, true, nullptr, &bottom));
    EXPECT_TRUE(scene.clear(points, identity, 0.0F, true, nullptr, &bottom, 0.003F));
    EXPECT_FALSE(scene.clear(points, identity, 0.0F, false, nullptr, &bottom, 0.003F));
    const std::vector<uint8_t> side{0};
    EXPECT_FALSE(scene.clear(points, identity, 0.0F, true, nullptr, &side, 0.003F));
}

TEST(PiperFingerGeometry, MeshBoundsRejectOnlyEmptyJawSpace) {
    const auto pose = Eigen::Isometry3f::Identity();
    EXPECT_FALSE(vamp_node::piperFingerMeshBoundsContain(
        88, Eigen::Vector3f(0, 0, 0.12F), pose, 0.01F));
    EXPECT_TRUE(vamp_node::piperFingerMeshBoundsContain(
        88, Eigen::Vector3f(0, -0.01F, 0.12F), pose, 0.01F));
    EXPECT_TRUE(vamp_node::piperFingerMeshBoundsContain(
        95, Eigen::Vector3f(0, 0.01F, 0.12F), pose, 0.01F));
    // Palm/arm hits must never be dismissed by a finger-only refinement.
    EXPECT_TRUE(vamp_node::piperFingerMeshBoundsContain(
        74, Eigen::Vector3f(0, 0, 0.12F), pose, 0.01F));
    EXPECT_TRUE(vamp_node::piperFingerMeshBoundsContain(
        88, Eigen::Vector3f(0, 0, 0.12F), pose, -1.0F));
}
