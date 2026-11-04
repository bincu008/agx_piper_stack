#include <gtest/gtest.h>
#include "vision_node/spatial_grid.hpp"
#include <limits>

using vision_node::SpatialGrid;
TEST(SpatialGrid, StableLayoutAndBounds) {
  EXPECT_EQ(SpatialGrid::cell({.1F,-.5F,.1F}),"x0_y0_z0");
  EXPECT_EQ(SpatialGrid::cell({.4F,.4F,.4F}),"x1_y3_z1");
  EXPECT_EQ(SpatialGrid::cell({-.01F,0.F,0.F}),"");
  EXPECT_EQ(SpatialGrid::cell({.6F,0.F,0.F}),"");
  for(int i=0;i<16;++i)
    EXPECT_EQ(SpatialGrid::index(SpatialGrid::minimum(i)+Eigen::Vector3f::Constant(.15F)),i);
}
TEST(SpatialGrid, EmptyTraversalHitAndOcclusion) {
  SpatialGrid grid;
  grid.update({-.1F,-.45F,.15F},std::vector<Eigen::Vector3f>(3,{.45F,-.45F,.15F}),100);
  EXPECT_EQ(grid.last_scanned[0],100); // Empty cell before endpoint.
  EXPECT_EQ(grid.last_scanned[1],100); // Endpoint cell.
  grid.update({-.1F,-.45F,.15F},std::vector<Eigen::Vector3f>(3,{.1F,-.45F,.15F}),200);
  EXPECT_EQ(grid.last_scanned[0],200);
  EXPECT_EQ(grid.last_scanned[1],100); // Behind occluder remains old.
  EXPECT_EQ(grid.observed[1],0U);
}
TEST(SpatialGrid, InvalidMissingSparseAndOldSamples) {
  SpatialGrid grid;
  grid.update({-.1F,-.45F,.15F},{{.1F,-.45F,.15F}},100);
  EXPECT_EQ(grid.last_scanned[0],0); // A single corner ray cannot refresh.
  const float nan=std::numeric_limits<float>::quiet_NaN();
  grid.update({-.1F,-.45F,.15F},std::vector<Eigen::Vector3f>(3,{nan,0,0}),200);
  EXPECT_EQ(grid.last_scanned[0],0);
  grid.update({-.1F,-.45F,.15F},std::vector<Eigen::Vector3f>(3,{.1F,-.45F,.15F}),300);
  grid.update({-.1F,-.45F,.15F},std::vector<Eigen::Vector3f>(3,{.1F,-.45F,.15F}),100);
  EXPECT_EQ(grid.last_scanned[0],300);
}
TEST(SpatialGrid, CameraOutsideAndNoInfiniteRay) {
  EXPECT_TRUE(SpatialGrid::traverses({-1.F,-.45F,.15F},{1.F,-.45F,.15F},0));
  EXPECT_FALSE(SpatialGrid::traverses({-1.F,-.45F,.15F},{-.1F,-.45F,.15F},0));
  EXPECT_FALSE(SpatialGrid::traverses({-.1F,.8F,.15F},{1.F,.8F,.15F},0));
}

TEST(SpatialGrid, SurfaceOnBoundaryBelongsToHitCellToo) {
  SpatialGrid grid;
  grid.update({-.1F,-.45F,.15F},std::vector<Eigen::Vector3f>(3,{.30F,-.45F,.15F}),400);
  EXPECT_EQ(grid.last_scanned[0],400);
  EXPECT_EQ(grid.last_scanned[1],400);
}
TEST(SpatialGrid, InvalidCameraCannotMarkEndpoint) {
  SpatialGrid grid;
  const float nan=std::numeric_limits<float>::quiet_NaN();
  grid.update({nan,0,0},std::vector<Eigen::Vector3f>(3,{.1F,-.45F,.15F}),100);
  EXPECT_EQ(grid.last_scanned[0],0);
}
