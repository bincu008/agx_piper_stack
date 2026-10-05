#include <gtest/gtest.h>
#include <limits>
#include "vamp_node/placement_ranking.hpp"

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
