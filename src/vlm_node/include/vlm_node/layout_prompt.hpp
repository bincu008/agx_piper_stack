#pragma once

#include <algorithm>
#include <cctype>
#include <regex>
#include <string>
#include <vector>

namespace vlm_node {
struct LayoutPrompt {
    std::vector<std::string> objects;
    std::string destination;
    bool valid = false;
};

inline std::string layoutTrim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
}

inline LayoutPrompt parseLayoutPrompt(const std::string &prompt) {
    LayoutPrompt result;
    std::smatch match;
    const std::regex syntax(
        R"(^\s*(?:place\s+)?(.+)\s+(?:onto|into|on|in)\s+(.+?)\s*[.]?\s*$)",
        std::regex::icase);
    if (!std::regex_match(prompt, match, syntax)) return result;
    const auto clean = [](std::string value) {
        value = layoutTrim(value);
        return std::regex_replace(value, std::regex(
            R"(^(?:the|an|a)\s+)", std::regex::icase), "");
    };
    result.destination = clean(match[2].str());
    if (!result.destination.empty() && result.destination.back() == '.') {
        result.destination.pop_back();
    }
    const std::string clause = std::regex_replace(match[1].str(),
        std::regex(R"(\s+then\s+|\s+and\s+|\s*,\s*(?:and\s+)?)",
            std::regex::icase), "|");
    std::size_t begin = 0;
    while (begin <= clause.size()) {
        const auto end = clause.find('|', begin);
        const auto object = clean(clause.substr(begin,
            end == std::string::npos ? end : end - begin));
        if (object.empty()) return result;
        result.objects.push_back(object);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    result.valid = result.objects.size() > 1 && !result.destination.empty();
    return result;
}
}  // namespace vlm_node
