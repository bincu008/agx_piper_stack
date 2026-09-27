#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "vamp_node/analytical_grasp_generator.hpp"
#include "vamp_node/deterministic_ik.hpp"
#include "vamp_node/gpd_diversity.hpp"

namespace {

std::vector<Eigen::Vector3f> makeBox() {
  std::vector<Eigen::Vector3f> cloud;
  for (int x = -10; x <= 10; ++x) for (int y = -6; y <= 6; ++y) {
    const float xf = x * 0.004F, yf = y * 0.004F;
    cloud.emplace_back(xf, yf, 0.00F); cloud.emplace_back(xf, yf, 0.06F);
  }
  for (int x = -10; x <= 10; ++x) for (int z = 0; z <= 15; ++z) {
    const float xf = x * 0.004F, zf = z * 0.004F;
    cloud.emplace_back(xf, -0.024F, zf); cloud.emplace_back(xf, 0.024F, zf);
  }
  return cloud;
}

std::vector<Eigen::Vector3f> makeCylinder() {
  std::vector<Eigen::Vector3f> cloud;
  for (int z = 0; z <= 15; ++z) for (int angle = 0; angle < 72; ++angle) {
    const float radians = angle * 2.0F * static_cast<float>(M_PI) / 72.0F;
    cloud.emplace_back(0.035F * std::cos(radians), 0.035F * std::sin(radians), z * 0.004F);
  }
  return cloud;
}

std::vector<Eigen::Vector3f> makeHandle() {
  std::vector<Eigen::Vector3f> cloud;
  for (int x = -20; x <= 20; ++x) for (int y = -2; y <= 2; ++y) for (int z = -2; z <= 2; ++z) {
    cloud.emplace_back(x * 0.003F, y * 0.003F, z * 0.003F + 0.04F);
  }
  return cloud;
}

void expectValid(const std::vector<vamp_node::GraspProposal> &proposals) {
  ASSERT_FALSE(proposals.empty());
  for (const auto &proposal : proposals) {
    EXPECT_TRUE(proposal.has_contacts);
    EXPECT_TRUE(proposal.approach.allFinite());
    EXPECT_TRUE(proposal.closing.allFinite());
    EXPECT_NEAR(proposal.approach.norm(), 1.0F, 1.0e-4F);
    EXPECT_NEAR(proposal.closing.norm(), 1.0F, 1.0e-4F);
    EXPECT_GT(proposal.opening_m, 0.005F);
    EXPECT_LE(proposal.opening_m, 0.100F);
  }
}

TEST(AnalyticalGraspGenerator, DeterministicBoxCylinderAndHandle) {
  vamp_node::AnalyticalGraspParameters parameters;
  parameters.maximum_proposals = 200;
  parameters.voxel_size_m = 0.004F;
  parameters.normal_radius_m = 0.012F;
  parameters.min_normal_neighbors = 8;
  vamp_node::AnalyticalGraspGenerator generator(parameters);
  const Eigen::Vector3f seed(0.0F, 0.0F, 0.03F);
  const Eigen::Vector3f camera(0.0F, -0.35F, 0.20F);
  for (const auto &cloud : {makeBox(), makeCylinder(), makeHandle()}) {
    const auto first = generator.generate(cloud, seed, camera);
    const auto second = generator.generate(cloud, seed, camera);
    expectValid(first);
    ASSERT_EQ(first.size(), second.size());
    for (std::size_t index = 0; index < first.size(); ++index) {
      EXPECT_NEAR((first[index].contact_left - second[index].contact_left).norm(), 0.0F, 1.0e-7F);
      EXPECT_NEAR((first[index].approach - second[index].approach).norm(), 0.0F, 1.0e-7F);
      EXPECT_NEAR(first[index].opening_m, second[index].opening_m, 1.0e-7F);
    }
  }
}

TEST(AnalyticalGraspGenerator, SelectsSpatiallyDiverseContactFamilies) {
  vamp_node::AnalyticalGraspParameters parameters;
  parameters.maximum_proposals = 100;
  parameters.local_pca_searches = 50;
  parameters.global_pca_searches = 50;
  parameters.maximum_approach_variants_per_pair = 1;
  parameters.min_contact_center_separation_m = 0.020F;
  parameters.voxel_size_m = 0.004F;
  parameters.normal_radius_m = 0.012F;
  parameters.min_normal_neighbors = 8;
  vamp_node::AnalyticalGraspGenerator generator(parameters);
  vamp_node::AnalyticalGraspDiagnostics diagnostics;
  const auto proposals = generator.generate(
    makeCylinder(), Eigen::Vector3f(0.0F, 0.0F, 0.03F),
    Eigen::Vector3f(0.0F, -0.35F, 0.20F), &diagnostics);
  ASSERT_FALSE(proposals.empty());
  EXPECT_GT(diagnostics.selected_contact_families, 1U);

  std::vector<Eigen::Vector3f> distinct_centers;
  for (const auto &proposal : proposals) {
    const Eigen::Vector3f center =
      (proposal.contact_left + proposal.contact_right) * 0.5F;
    bool distinct = true;
    for (const auto &existing : distinct_centers) {
      if ((center - existing).norm() < parameters.min_contact_center_separation_m) {
        distinct = false;
        break;
      }
    }
    if (distinct) distinct_centers.push_back(center);
  }
  EXPECT_GE(distinct_centers.size(), 2U);
  EXPECT_TRUE(std::any_of(proposals.begin(), proposals.end(),
    [](const vamp_node::GraspProposal &proposal) {
      return proposal.model_supported_contacts;
    }));
}

TEST(DeterministicIk, StableSeedOrderAndSolutionSelection) {
  const vamp_node::PiperJointArray current{{0.10F, 1.00F, -1.20F, 0.10F, 0.00F, 0.10F}};
  const vamp_node::PiperJointArray previous{{0.15F, 1.05F, -1.10F, 0.00F, 0.05F, 0.05F}};
  const vamp_node::PiperJointArray home{{0.00F, 0.80F, -1.40F, 0.00F, 0.00F, 0.00F}};
  const auto first = vamp_node::deterministicIkSeedBank(current, previous, home, 32);
  const auto second = vamp_node::deterministicIkSeedBank(current, previous, home, 32);
  ASSERT_EQ(first.size(), second.size());
  ASSERT_GE(first.size(), 32U);
  for (std::size_t index = 0; index < first.size(); ++index) {
    EXPECT_EQ(first[index], second[index]);
    for (std::size_t joint = 0; joint < first[index].size(); ++joint) {
      EXPECT_GE(first[index][joint], vamp_node::kPiperJointLower[joint]);
      EXPECT_LE(first[index][joint], vamp_node::kPiperJointUpper[joint]);
    }
  }

  const vamp_node::PiperJointArray farther{{0.40F, 1.00F, -1.20F, 0.10F, 0.00F, 0.10F}};
  const vamp_node::PiperJointArray closer{{0.11F, 1.00F, -1.20F, 0.10F, 0.00F, 0.10F}};
  EXPECT_TRUE(vamp_node::deterministicIkSolutionPreferred(closer, 9, farther, 0, current));
  EXPECT_FALSE(vamp_node::deterministicIkSolutionPreferred(farther, 0, closer, 9, current));
  EXPECT_TRUE(vamp_node::deterministicIkSolutionPreferred(current, 1, current, 2, current));
}

TEST(GpdDiversity, SymmetricClosingAxesAndDifferentApproaches) {
  const Eigen::Vector3f position(0.2F, 0.0F, 0.1F);
  const Eigen::Vector3f approach = Eigen::Vector3f::UnitZ();
  const Eigen::Vector3f closing = Eigen::Vector3f::UnitY();
  EXPECT_TRUE(vamp_node::gpdPosesAreDuplicates(
    position, approach, closing, position + Eigen::Vector3f(0.005F, 0.0F, 0.0F),
    approach, -closing, 0.015F, 15.0F, 15.0F));
  EXPECT_FALSE(vamp_node::gpdPosesAreDuplicates(
    position, approach, closing, position + Eigen::Vector3f(0.005F, 0.0F, 0.0F),
    Eigen::Vector3f::UnitX(), closing, 0.015F, 15.0F, 15.0F));
  EXPECT_FALSE(vamp_node::gpdPosesAreDuplicates(
    position, approach, closing, position + Eigen::Vector3f(0.020F, 0.0F, 0.0F),
    approach, closing, 0.015F, 15.0F, 15.0F));
}

}  // namespace
