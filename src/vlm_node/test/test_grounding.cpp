#include <gtest/gtest.h>
#include "vlm_node/grounding.hpp"

TEST(Grounding, RequestSettingsAndImageOrder)
{
    nlohmann::json payload = {{"stream", false}, {"max_tokens", 256}};
    vlm_node::configureGroundingRequest(payload);
    EXPECT_EQ(payload["max_tokens"], 48);
    EXPECT_EQ(payload["temperature"], 0);
    EXPECT_EQ(payload["seed"], 1234);
    EXPECT_EQ(payload["ignore_eos"], false);
    EXPECT_EQ(payload["cache_prompt"], false);
    EXPECT_EQ(payload["stream"], false);
    const auto prompt = vlm_node::groundingPrompt("the cup");
    EXPECT_EQ(prompt, "Please provide the bounding box coordinate of the region this sentence describes: the cup. Return only [x1, y1, x2, y2] in normalized 0-1000 coordinates. Cover the full visible extent of the object, including its handle if present.");
    const auto content = vlm_node::imageFirstContent("original-image", prompt);
    EXPECT_EQ(content[0]["image_url"]["url"], "original-image");
    EXPECT_EQ(content[1]["text"], prompt);
}

TEST(Grounding, StrictBoxAndNonSquareInclusivePixels)
{
    const std::string reply = " [250, 100, 750, 900]\n";
    std::smatch match;
    ASSERT_TRUE(vlm_node::parseGroundingBox(reply, match));
    EXPECT_EQ(vlm_node::groundingPixel(std::stoi(match[1]), 1280), 320);
    EXPECT_EQ(vlm_node::groundingPixel(std::stoi(match[2]), 720), 72);
    EXPECT_EQ(vlm_node::groundingPixel(std::stoi(match[3]), 1280), 960);
    EXPECT_EQ(vlm_node::groundingPixel(std::stoi(match[4]), 720), 648);
    EXPECT_EQ(vlm_node::groundingPixel(1000, 1280), 1279);
    for (const std::string invalid : {"", "[0,0,1]", "[0.5,0,2,2]",
        "[-1,0,2,2]", "[0,0,1001,1000]", "[10,0,1,1]", "[0,1,1,1]",
        "[0,0,1,1] [1,1,2,2]", "box: [0,0,1,1]", "[999999999999,0,1,1]"}) {
        EXPECT_FALSE(vlm_node::parseGroundingBox(invalid, match)) << invalid;
    }
}

TEST(Grounding, ContactDirectiveDoesNotCropRigidObjectGeometry)
{
    EXPECT_EQ(vlm_node::wholeObjectGroundingTarget("beige cup, grasp its body"), "beige cup");
    EXPECT_EQ(vlm_node::wholeObjectGroundingTarget("beige cup, grasp its rim"), "beige cup");
    EXPECT_EQ(vlm_node::wholeObjectGroundingTarget("cup, left of tray"), "cup, left of tray");
    EXPECT_EQ(vlm_node::wholeObjectGroundingTarget("the complete visible placement surface of the tray"),
              "the complete visible placement surface of the tray");
}
