#include <gtest/gtest.h>
#include "vlm_node/inference_capture.hpp"

TEST(InferenceCapture, SeparatesDestinationAndObjectInSameFrame) {
    vlm_node::CaptureAssociations cache;
    const std::array<int,4> cup{10,20,30,40},tray{50,60,90,100};
    cache.remember("frame1",cup,"object");cache.remember("frame1",tray,"destination");
    EXPECT_EQ(cache.find("frame1",cup),"object");
    EXPECT_EQ(cache.find("frame1",tray),"destination");
    EXPECT_TRUE(cache.find("frame2",cup).empty());
    EXPECT_EQ(vlm_node::captureRole("the complete visible placement surface of the tray"),"destination");
    EXPECT_EQ(vlm_node::captureRole("beige cup, grasp its body"),"object");
}
TEST(InferenceCapture, BoundsAssociationAndSupportsExpandedBoxes) {
    vlm_node::CaptureAssociations cache;
    const std::array<int,4> box{10,20,30,40};
    cache.remember("old",box,"object");
    EXPECT_EQ(cache.find("old",{9,19,31,41}),"object");
    for(int i=0;i<32;++i) cache.remember(std::to_string(i),box,"destination");
    EXPECT_TRUE(cache.find("old",box).empty());
    EXPECT_EQ(cache.find("31",box),"destination");
}
