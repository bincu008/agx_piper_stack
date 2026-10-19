#pragma once

#include <cmath>
#include <algorithm>
#include <regex>
#include <string>
#include "vlm_node/json.hpp"

namespace vlm_node
{

inline std::string wholeObjectGroundingTarget(const std::string &description)
{
    static const std::regex directive(R"(\s*,\s*grasp\s+its\s+(body|rim|handle|side|edge|lip)\s*$)", std::regex::icase);
    return std::regex_replace(description, directive, "");
}

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

// Lenient fallback for agent-style replies (e.g. nanobot): finds the first
// "[x1,y1,x2,y2]" bracket group embedded in surrounding text and rounds
// fractional coordinates. The strict parse above stays the primary path.
// The rebuilt string lives in thread-local storage so the returned smatch
// stays valid until the next call on the same thread.
inline bool parseGroundingBoxLenient(const std::string &reply, std::smatch &match)
{
    if (parseGroundingBox(reply, match)) return true;
    static const std::regex loose(
        R"(\[\s*([0-9]{1,4}(?:\.[0-9]+)?)\s*,\s*([0-9]{1,4}(?:\.[0-9]+)?)\s*,\s*([0-9]{1,4}(?:\.[0-9]+)?)\s*,\s*([0-9]{1,4}(?:\.[0-9]+)?)\s*\])");
    static thread_local std::string rebuilt;
    std::smatch found;
    if (!std::regex_search(reply, found, loose)) return false;
    const int coords[4] = {
        static_cast<int>(std::lround(std::stod(found[1].str()))),
        static_cast<int>(std::lround(std::stod(found[2].str()))),
        static_cast<int>(std::lround(std::stod(found[3].str()))),
        static_cast<int>(std::lround(std::stod(found[4].str())))};
    for (int c : coords) {
        if (c < 0 || c > 1000) return false;
    }
    if (!(coords[0] < coords[2] && coords[1] < coords[3])) return false;
    rebuilt = "[" + std::to_string(coords[0]) + "," + std::to_string(coords[1]) +
        "," + std::to_string(coords[2]) + "," + std::to_string(coords[3]) + "]";
    return parseGroundingBox(rebuilt, match);
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
