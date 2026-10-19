#pragma once

#include <array>
#include <deque>
#include <string>

namespace vlm_node {

inline std::string captureRole(const std::string &target) {
    return target.find("placement surface") != std::string::npos ? "destination" : "object";
}

class CaptureAssociations {
    struct Entry {std::string frame; std::array<int,4> box; std::string role;};
    std::deque<Entry> entries_;
public:
    void remember(const std::string &frame,const std::array<int,4> &box,const std::string &role) {
        entries_.push_back({frame,box,role});
        if (entries_.size()>32) entries_.pop_front();
    }
    std::string find(const std::string &frame,const std::array<int,4> &box) const {
        for (auto it=entries_.rbegin();it!=entries_.rend();++it)
            if (it->frame==frame && it->box==box) return it->role;
        // Native callers may expand the grounded box before segmentation.
        for (auto it=entries_.rbegin();it!=entries_.rend();++it)
            if (it->frame==frame && box[0]<=it->box[0] && box[1]<=it->box[1] &&
                box[2]>=it->box[2] && box[3]>=it->box[3]) return it->role;
        return {};
    }
};
}  // namespace vlm_node
