#pragma once

#include <cmath>
#include <algorithm>
#include <regex>
#include <string>
#include "vlm_node/json.hpp"

namespace vlm_node
{
inline std::string groundingPrompt(const std::string &target)
{
    return "Please provide the bounding box coordinate of the region this sentence describes: " +
        target + ". Return only [x1, y1, x2, y2] in normalized 0-1000 coordinates. "
        "Cover the full visible extent of the object, including its handle if present.";
}

inline bool parseGroundingBox(const std::string &reply, std::smatch &match)
{
    static const std::regex pattern(
        R"(^\s*\[\s*([0-9]{1,4})\s*,\s*([0-9]{1,4})\s*,\s*([0-9]{1,4})\s*,\s*([0-9]{1,4})\s*\]\s*$)");
    if (!std::regex_match(reply, match, pattern)) return false;
    const int x1 = std::stoi(match[1]), y1 = std::stoi(match[2]);
    const int x2 = std::stoi(match[3]), y2 = std::stoi(match[4]);
    return x1 < x2 && y1 < y2 && x2 <= 1000 && y2 <= 1000;
}

// Preserve the existing inclusive pixel-coordinate convention used by EdgeSAM.
inline int groundingPixel(double coordinate, int dimension)
{
    return std::clamp(static_cast<int>(std::lround(coordinate * dimension / 1000.0)),
        0, dimension - 1);
}

inline nlohmann::json imageFirstContent(const std::string &image_url, const std::string &prompt)
{
    auto content = nlohmann::json::array();
    if (!image_url.empty()) {
        content.push_back({{"type", "image_url"}, {"image_url", {{"url", image_url}}}});
    }
    content.push_back({{"type", "text"}, {"text", prompt}});
    return content;
}

inline void configureGroundingRequest(nlohmann::json &payload)
{
    payload["temperature"] = 0;
    payload["seed"] = 1234;
    payload["max_tokens"] = 48;
    payload["ignore_eos"] = false;
    payload["cache_prompt"] = false;
}
}  // namespace vlm_node
