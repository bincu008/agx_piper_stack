#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <Eigen/Geometry>
#include <octomap/OcTree.h>

namespace vision_node {

// Same OctoMap ray caster as next-view evaluation. The terminal margin prevents
// the target's surface/occupied voxel from counting as an intervening obstacle.
inline bool viewRayClear(const octomap::OcTree &tree, const Eigen::Vector3f &origin,
                         const Eigen::Vector3f &target, float margin) {
    if (!origin.allFinite() || !target.allFinite() || !std::isfinite(margin) || margin<0.F)
        return false;
    const Eigen::Vector3f delta=target-origin;
    const float length=delta.norm();
    if (length<=margin) return false;
    octomap::point3d hit;
    if (!tree.castRay(octomap::point3d(origin.x(),origin.y(),origin.z()),
                      octomap::point3d(delta.x(),delta.y(),delta.z()),hit,true,length-margin))
        return true;
    return (Eigen::Vector3f(hit.x(),hit.y(),hit.z())-origin).norm()>=length-margin;
}

struct VisibilityResult {
    unsigned samples=0, in_view=0, clear=0;
    float fraction=0.F;
    bool visible=false;
};

inline VisibilityResult targetVisibility(const std::vector<Eigen::Vector3f> &scene,
    const std::vector<Eigen::Vector3f> &target, const Eigen::Isometry3f &base_from_camera,
    float fx,float fy,float cx,float cy,unsigned width,unsigned height) {
    VisibilityResult result;
    // Use an immutable current-view tree, never the accumulated world: stale
    // occupied cells must not keep blocking a view after an object is moved.
    constexpr float resolution=.005F;
    octomap::OcTree target_tree(resolution),scene_tree(resolution);
    for (const auto &p:target)
        if (p.allFinite()) target_tree.updateNode(octomap::point3d(p.x(),p.y(),p.z()),true,true);
    for (const auto &p:scene) {
        if (!p.allFinite()) continue;
        const octomap::point3d point(p.x(),p.y(),p.z());
        // Exclude only the remembered target's occupied cells, not its bounding
        // box or nearby objects. This also avoids self-occluding its far rim.
        if (!target_tree.search(point)) scene_tree.updateNode(point,true,true);
    }
    scene_tree.updateInnerOccupancy();
    const auto camera_from_base=base_from_camera.inverse();
    const Eigen::Vector3f origin=base_from_camera.translation();
    const size_t count=std::min<size_t>(64,target.size());
    for (size_t i=0;i<count;++i) {
        const auto &p=target[count==1 ? 0 : i*(target.size()-1)/(count-1)];
        if (!p.allFinite()) continue;
        ++result.samples;
        const Eigen::Vector3f optical=camera_from_base*p;
        if (optical.z()<.05F || optical.z()>1.F) continue;
        const float u=fx*optical.x()/optical.z()+cx;
        const float v=fy*optical.y()/optical.z()+cy;
        if (u<0.F || v<0.F || u>=width || v>=height) continue;
        ++result.in_view;
        if (viewRayClear(scene_tree,origin,p,.010F)) ++result.clear;
    }
    if (result.samples) result.fraction=static_cast<float>(result.clear)/result.samples;
    result.visible=result.samples>0 && result.fraction>=.6F;
    return result;
}
}  // namespace vision_node
