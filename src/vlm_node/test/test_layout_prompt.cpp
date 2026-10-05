#include <gtest/gtest.h>
#include "vlm_node/layout_prompt.hpp"

TEST(LayoutPrompt, SupportsArbitraryOrderedObjects) {
    const auto result = vlm_node::parseLayoutPrompt(
        "place the cup then the plate then an apple onto the tray");
    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.objects, (std::vector<std::string>{"cup", "plate", "apple"}));
    EXPECT_EQ(result.destination, "tray");
}
TEST(LayoutPrompt, CommasAndConjunctionKeepOrder) {
    const auto result = vlm_node::parseLayoutPrompt(
        "Place a cup, the white plate, and an apple into the tray.");
    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.objects, (std::vector<std::string>{"cup", "white plate", "apple"}));
    EXPECT_EQ(result.destination, "tray");
}
TEST(LayoutPrompt, RejectsMissingObjectOrDestination) {
    EXPECT_FALSE(vlm_node::parseLayoutPrompt("place cup onto tray").valid);
    EXPECT_FALSE(vlm_node::parseLayoutPrompt("place cup then onto tray").valid);
    EXPECT_FALSE(vlm_node::parseLayoutPrompt("place cup then plate").valid);
}
