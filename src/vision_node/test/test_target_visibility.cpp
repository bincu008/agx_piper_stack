#include <cassert>
#include <cmath>
#include "vision_node/target_visibility.hpp"

using Points=std::vector<Eigen::Vector3f>;
Points plane(float z) {
    Points points;
    for (int x=-4;x<=4;++x)
        for (int y=-4;y<=4;++y) points.emplace_back(x*.01F,y*.01F,z);
    return points;
}
int main() {
    const auto identity=Eigen::Isometry3f::Identity();
    const auto target=plane(.5F);
    auto result=vision_node::targetVisibility(target,target,identity,500,500,320,240,640,480);
    assert(result.visible && result.samples==64 && result.clear==64);
    // A separate foreground surface blocks the rays, not the target itself.
    Points blocked;
    for (int x=-30;x<=30;++x)
        for (int y=-30;y<=30;++y) blocked.emplace_back(x*.002F,y*.002F,.3F);
    blocked.insert(blocked.end(),target.begin(),target.end());
    result=vision_node::targetVisibility(blocked,target,identity,500,500,320,240,640,480);
    assert(!result.visible && result.in_view==64 && result.clear==0);
    // Known far rim samples must not be blocked by the object's known near rim.
    Points complete=target;
    auto back=plane(.52F);complete.insert(complete.end(),back.begin(),back.end());
    assert(vision_node::targetVisibility(complete,complete,identity,
        500,500,320,240,640,480).visible);
    // Clear rays do not make a point behind/outside the camera visible.
    Points outside{Eigen::Vector3f(2,0,.5F)}, behind{Eigen::Vector3f(0,0,-.5F)};
    assert(!vision_node::targetVisibility(target,outside,identity,
        500,500,320,240,640,480).visible);
    assert(!vision_node::targetVisibility(target,behind,identity,
        500,500,320,240,640,480).visible);
    Points too_far{Eigen::Vector3f(0,0,1.1F)}, too_close{Eigen::Vector3f(0,0,.03F)};
    assert(!vision_node::targetVisibility(target,too_far,identity,
        500,500,320,240,640,480).visible);
    assert(!vision_node::targetVisibility(target,too_close,identity,
        500,500,320,240,640,480).visible);
    // Use the supplied manual camera geometry, including translation/rotation.
    Eigen::Isometry3f moved=Eigen::Isometry3f::Identity();
    moved.translation()=Eigen::Vector3f(.2F,-.1F,.8F);
    moved.linear()=Eigen::AngleAxisf(1.2F,Eigen::Vector3f::UnitY()).toRotationMatrix();
    Points world;
    for (const auto &p:target) world.push_back(moved*p);
    assert(vision_node::targetVisibility(world,world,moved,500,500,320,240,640,480).visible);
    // Field-of-view coverage is part of the denominator, not just clear rays.
    Points partial;
    for(int i=0;i<10;++i)partial.emplace_back(i<5 ? 0.F : 2.F,0.F,.5F);
    result=vision_node::targetVisibility(target,partial,identity,500,500,320,240,640,480);
    assert(!result.visible && result.in_view==5 && std::abs(result.fraction-.5F)<.001F);
    octomap::OcTree tree(.005);
    tree.updateNode(octomap::point3d(0,0,.5),true);
    assert(vision_node::viewRayClear(tree,Eigen::Vector3f::Zero(),Eigen::Vector3f(0,0,.5),.01));
    assert(!vision_node::viewRayClear(tree,Eigen::Vector3f::Zero(),Eigen::Vector3f(0,0,.8),.01));
    assert(!vision_node::viewRayClear(tree,Eigen::Vector3f::Zero(),Eigen::Vector3f::Zero(),.01));
}
