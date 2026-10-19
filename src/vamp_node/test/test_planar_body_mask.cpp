#include "vamp_node/planar_body_mask.hpp"
#include <gtest/gtest.h>
TEST(PlanarBodyMask, PreservesPlateAndRaisedRimRejectsLowerStand) {
  std::vector<Eigen::Vector3f> points;
  for (int i=0;i<25;++i) for (int j=0;j<25;++j) points.emplace_back(i*.005f,j*.005f,.06f);
  const auto base=points.size();
  points.emplace_back(.12f,.06f,.075f);
  for (int i=0;i<30;++i) points.emplace_back(.06f,.06f,.01f+i*.0005f);
  const auto keep=vamp_node::planar_body_mask(points);
  for (size_t i=0;i<keep.size();++i) EXPECT_EQ(keep[i], i<=base);
}
TEST(PlanarBodyMask, InsufficientEvidencePreservesInput) {
  std::vector<Eigen::Vector3f> points(10, Eigen::Vector3f::Zero());
  for (bool keep:vamp_node::planar_body_mask(points)) EXPECT_TRUE(keep);
}
