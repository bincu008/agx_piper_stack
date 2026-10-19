#pragma once
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>
#include <array>
#include <limits>

namespace vamp_node {
// Inner pad normal is the jaw-closing axis (tool Y), not finger length Z.
inline float padSurfaceAngle(const Eigen::Vector3f &jaw,
    const Eigen::Vector3f &left, const Eigen::Vector3f &right) {
  return std::acos(std::clamp(std::min(std::abs(jaw.normalized().dot(left.normalized())),
    std::abs(jaw.normalized().dot(right.normalized()))), 0.0F, 1.0F));
}
// Preserve the original side of the object relative to gravity (+Z base_link).
// A small correction may become vertical, but may not tilt through to the
// opposite side. Ignore an original <1-degree lateral tilt as sensor noise.
inline bool preservesApproachSide(const Eigen::Matrix3f &original,
    const Eigen::Matrix3f &adjusted) {
  if (!original.allFinite() || !adjusted.allFinite()) return false;
  const Eigen::Vector3f before = original.col(2).normalized();
  const Eigen::Vector3f after = adjusted.col(2).normalized();
  if (before.dot(after) <= 0.0F) return false;
  Eigen::Vector3f side_before = before;
  Eigen::Vector3f side_after = after;
  side_before.z() = 0.0F;
  side_after.z() = 0.0F;
  if (side_before.norm() < 0.0174524F) return true;
  return side_before.dot(side_after) >= -1.e-6F;
}
// Positive displacement points away from the measured object center through
// the contact region. Compare the physical hand base, not the offset TCP.
inline std::optional<float> outwardHandDisplacement(
    const Eigen::Isometry3f &original, const Eigen::Isometry3f &adjusted,
    const Eigen::Vector3f &contact_center, const Eigen::Vector3f &object_center,
    float tcp_offset_m) {
  Eigen::Vector3f outward = contact_center - object_center;
  outward.z() = 0.0F;
  if (!outward.allFinite() || outward.norm() < .005F ||
      !original.matrix().allFinite() || !adjusted.matrix().allFinite()) return std::nullopt;
  outward.normalize();
  const Eigen::Vector3f before = original.translation() - original.linear().col(2)*tcp_offset_m;
  const Eigen::Vector3f after = adjusted.translation() - adjusted.linear().col(2)*tcp_offset_m;
  return (after-before).dot(outward);
}
inline bool handAdjustmentIsOutward(const Eigen::Isometry3f &original,
    const Eigen::Isometry3f &adjusted, const Eigen::Vector3f &contact_center,
    const Eigen::Vector3f &object_center, float tcp_offset_m) {
  const auto displacement = outwardHandDisplacement(original,adjusted,
    contact_center,object_center,tcp_offset_m);
  // No definite radial direction means no certified angle correction.
  return displacement && *displacement >= -.0001F;
}
// Degrees are candidate surface-angle targets; only improve the current pose.
inline std::vector<float> outwardBendTargets(float current_angle_rad) {
  std::vector<float> targets;
  if (!std::isfinite(current_angle_rad)) return targets;
  const float degrees=current_angle_rad*180.F/3.14159265358979323846F;
  for (float target : {10.F,30.F,50.F}) {
    if (target<degrees-1.e-4F) targets.push_back(target);
  }
  return targets;
}
// Keep the original physical pad bounds. Only depth errors can be corrected
// by advancing along approach; lateral errors cannot be repaired this way.
inline std::optional<float> contactPadAdvance(const Eigen::Isometry3f &pose,
    const std::array<Eigen::Vector3f,2> &contacts, float opening,
    float length, float height, float tcp_offset, float maximum_advance=.010F) {
  if (!pose.matrix().allFinite() || !std::isfinite(opening) || opening<0 ||
      !std::isfinite(length) || length<=.004F || !std::isfinite(height) || height<=0 ||
      !std::isfinite(tcp_offset)) return std::nullopt;
  float minimum_depth=std::numeric_limits<float>::infinity();
  float maximum_depth=-minimum_depth;
  for (const auto &contact : contacts) {
    if (!contact.allFinite()) return std::nullopt;
    const Eigen::Vector3f local=pose.linear().transpose()*(contact-pose.translation());
    if (std::abs(local.x())>height*.5F+.003F ||
        std::abs(std::abs(local.y())-opening*.5F)>.004F) return std::nullopt;
    const float depth=local.z()+tcp_offset;
    minimum_depth=std::min(minimum_depth,depth);maximum_depth=std::max(maximum_depth,depth);
  }
  if (minimum_depth<.002F) return std::nullopt;
  if (maximum_depth<=length-.002F) return 0.F;
  // Leave 1 mm inside the existing tip margin for the recovery IK error.
  const float advance=maximum_depth-(length-.002F)+.001F;
  if (advance>maximum_advance || minimum_depth-advance<.002F) return std::nullopt;
  return advance;
}
inline float fingerSurfaceAngle(const Eigen::Matrix3f &rotation,
    const Eigen::Vector3f &normal) {
  return std::asin(std::clamp(std::abs(rotation.col(2).dot(normal.normalized())),0.F,1.F));
}
// One hinge only: local tool X, through the original contact region. Solve
// the finger-to-surface angle analytically, retaining yaw and the hinge axis.
inline std::optional<Eigen::Isometry3f> bendOriginalGraspOutward(
    const Eigen::Isometry3f &original, const Eigen::Vector3f &normal,
    const Eigen::Vector3f &contact_center, const Eigen::Vector3f &object_center,
    float tcp_offset_m, float contact_depth_m, float surface_angle_rad) {
  if (!normal.allFinite() || normal.norm()<.5F || !original.matrix().allFinite() ||
      !std::isfinite(surface_angle_rad) || surface_angle_rad<=0.F || surface_angle_rad>=1.5707963F)
    return std::nullopt;
  if (fingerSurfaceAngle(original.linear(),normal)<=surface_angle_rad) return original;
  const Eigen::Vector3f n=normal.normalized();
  const float a=n.dot(original.linear().col(2));
  const float b=-n.dot(original.linear().col(1));
  const float radius=std::hypot(a,b);
  if (radius<1.e-6F) return original;
  const float phase=std::atan2(b,a);
  const float root=std::acos(std::clamp(std::sin(surface_angle_rad)/radius,-1.F,1.F));
  const float pi=3.14159265358979323846F;
  const Eigen::Vector3f pivot=original.translation()-original.linear().col(2)*
    (tcp_offset_m-contact_depth_m);
  float best_rotation=pi;
  std::optional<Eigen::Isometry3f> best;
  for (float angle : {phase+root,phase-root,phase+pi-root,phase-pi+root}) {
    angle=std::atan2(std::sin(angle),std::cos(angle));
    if (std::abs(angle)>=1.5707963F || std::abs(angle)>=best_rotation) continue;
    auto pose=original;
    pose.linear()=original.linear()*Eigen::AngleAxisf(angle,Eigen::Vector3f::UnitX()).toRotationMatrix();
    pose.translation()=pivot+pose.linear().col(2)*(tcp_offset_m-contact_depth_m);
    if (fingerSurfaceAngle(pose.linear(),normal)>surface_angle_rad+1.e-5F ||
        !handAdjustmentIsOutward(original,pose,contact_center,object_center,tcp_offset_m)) continue;
    best_rotation=std::abs(angle);best=pose;
  }
  return best;
}
inline bool sameOriginalHinge(const Eigen::Matrix3f &original,
    const Eigen::Matrix3f &actual) {
  return original.col(0).dot(actual.col(0))>=.9998F &&
    original.col(2).dot(actual.col(2))>0.F;
}
inline std::optional<Eigen::Matrix3f> alignContactSurfaces(
    const Eigen::Matrix3f &original, Eigen::Vector3f left,
    Eigen::Vector3f right, float tolerance_rad,
    Eigen::Matrix3f *preview_rotation = nullptr) {
  if (!left.allFinite() || !right.allFinite() || left.norm()<.5F || right.norm()<.5F)
    return std::nullopt;
  left.normalize(); right.normalize();
  const Eigen::Vector3f jaw=original.col(1);
  if (left.dot(jaw)<0) left=-left;
  if (right.dot(left)<0) right=-right;
  if ((left+right).norm()<1.e-5F) return std::nullopt;
  const Eigen::Vector3f desired=(left+right).normalized();
  const Eigen::Quaternionf correction=Eigen::Quaternionf::FromTwoVectors(jaw,desired);
  // A rejected surface pair still has a best-fit orientation for debug only.
  // Never return this preview as a valid alignment or an executable target.
  if (preview_rotation) *preview_rotation = (correction*original).eval();
  if (padSurfaceAngle(desired,left,right)>tolerance_rad) return std::nullopt;
  // Eight cheap geometry iterations find the smallest rotation into the cone.
  float lo=0,hi=1;
  for (int i=0;i<8;++i) {
    const float mid=(lo+hi)*.5F;
    const Eigen::Vector3f trial=Eigen::Quaternionf::Identity().slerp(mid,correction)*jaw;
    if (padSurfaceAngle(trial,left,right)<=tolerance_rad) hi=mid; else lo=mid;
  }
  const Eigen::Matrix3f adjusted =
    (Eigen::Quaternionf::Identity().slerp(hi,correction)*original).eval();
  if (preview_rotation) *preview_rotation = adjusted;
  if (!preservesApproachSide(original, adjusted)) return std::nullopt;
  return adjusted;
}
}  // namespace vamp_node
