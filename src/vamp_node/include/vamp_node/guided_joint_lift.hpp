#pragma once

#include "vamp_node/cartesian_grasp_approach.hpp"
#include "vamp_node/upright_motion.hpp"
#include <Eigen/Dense>
#include <chrono>

namespace vamp_node {

// Differential joint motion: height and object tilt are tasks; XY and yaw are
// free. Guidance and joint-margin centering act in the task's null space.
// No failed prefix is returned as an executable path.
inline CartesianApproachPlan planGuidedJointLift(
  const std::vector<float> &start, float rise, float tilt, float lateral_bound,
  const std::function<Eigen::Isometry3f(const std::vector<float> &)> &fk,
  const std::function<Eigen::Vector3f(const Eigen::Isometry3f &)> &guidance,
  const std::function<bool(const std::vector<float> &, const std::vector<float> &)> &edge_safe,
  float yaw_bias = 0.F, float seconds = 1.F)
{
  CartesianApproachPlan result;
  const auto fail = [&](const std::string &why) {
    result.success = false; result.failure_reason = why;
    result.waypoint_joints.clear(); result.waypoint_poses.clear(); return result;
  };
  if (!jointsWithinPiperLimits(start) || !std::isfinite(rise) || rise <= 0.F ||
      !std::isfinite(tilt) || tilt < 0.F || !std::isfinite(lateral_bound) ||
      lateral_bound <= 0.F || !std::isfinite(seconds) || seconds <= 0.F)
    return fail("GUIDED_LIFT_INVALID_INPUT");
  const auto origin = fk(start);
  if (!finiteIsometry(origin)) return fail("GUIDED_LIFT_INVALID_FK");
  const Eigen::Vector3f local_up = origin.linear().transpose()*Eigen::Vector3f::UnitZ();
  auto q = start;
  const auto started = std::chrono::steady_clock::now();
  for (size_t iteration = 0; iteration < 1536; ++iteration) {
    if (std::chrono::duration<float>(std::chrono::steady_clock::now()-started).count() > seconds)
      return fail("GUIDED_LIFT_TIME_BUDGET");
    const auto pose = fk(q);
    if (pose.translation().z()-origin.translation().z() >= rise)
      { result.success = true; return result; }
    const Eigen::Vector3f up = pose.linear()*local_up;
    Eigen::Matrix<float,4,6> jacobian;
    for (size_t j=0;j<6;++j) {
      auto perturbed = q; perturbed[j] += .001F;
      const auto next = fk(perturbed);
      jacobian(0,j) = (next.translation().z()-pose.translation().z())/.001F;
      jacobian.block<3,1>(1,j) = .08F*(next.linear()*local_up-up)/.001F;
    }
    Eigen::Matrix<float,4,1> task;
    task[0] = std::min(.0015F, rise-(pose.translation().z()-origin.translation().z())+.00002F);
    task.tail<3>() = .08F*.25F*(Eigen::Vector3f::UnitZ()-up);
    // An empty hand has no object orientation to retain.
    if (tilt >= 3.F) { jacobian.bottomRows<3>().setZero(); task.tail<3>().setZero(); }
    Eigen::Matrix<float,6,1> secondary;
    auto force = guidance(pose);
    if (!force.allFinite()) return fail("GUIDED_LIFT_INVALID_GUIDANCE");
    if (force.norm()>1.F) force.normalize();
    for (size_t j=0;j<6;++j) {
      const float center = (kPiperJointLower[j]+kPiperJointUpper[j])*.5F;
      secondary[j] = .002F*(center-q[j])/(kPiperJointUpper[j]-kPiperJointLower[j]);
      auto perturbed=q;perturbed[j]+=.001F;
      secondary[j] += .004F*force.dot((fk(perturbed).translation()-pose.translation())/.001F);
    }
    secondary[3] += yaw_bias;
    Eigen::Matrix<float,6,1> delta;
    std::array<bool,6> blocked{};
    for (size_t active=0;active<=6;++active) {
      const Eigen::Matrix<float,4,4> damped = jacobian*jacobian.transpose()+
        .00001F*Eigen::Matrix<float,4,4>::Identity();
      const Eigen::Matrix<float,6,4> inverse = jacobian.transpose()*damped.inverse();
      delta = inverse*task+(Eigen::Matrix<float,6,6>::Identity()-inverse*jacobian)*secondary;
      bool changed=false;
      for (size_t j=0;j<6;++j) {
        if (blocked[j]) { delta[j]=0.F; continue; }
        if ((q[j]>=kPiperJointUpper[j]-1.e-5F && delta[j]>0.F) ||
            (q[j]<=kPiperJointLower[j]+1.e-5F && delta[j]<0.F)) {
          blocked[j]=true; jacobian.col(j).setZero(); secondary[j]=0.F; changed=true;
        }
      }
      if (!changed) break;
    }
    if (!delta.allFinite()) return fail("GUIDED_LIFT_INVALID_STEP");
    if (delta.cwiseAbs().maxCoeff()>.004F) delta*=.004F/delta.cwiseAbs().maxCoeff();
    bool advanced=false;
    for (float scale : {1.F,.5F,.25F,.125F}) {
      auto next=q, midpoint=q;
      for (size_t j=0;j<6;++j) {
        next[j]=std::clamp(q[j]+scale*delta[j],kPiperJointLower[j],kPiperJointUpper[j]);
        midpoint[j]=.5F*(q[j]+next[j]);
      }
      const auto target=fk(next), middle=fk(midpoint);
      const auto local = [&](const Eigen::Isometry3f &p) {
        const Eigen::Vector3f d=p.translation()-origin.translation();
        return finiteIsometry(p) && d.z()>=-.001F && d.z()<=rise+.002F &&
          d.head<2>().norm()<=lateral_bound && objectRemainsUpright(p,origin,tilt);
      };
      if (!local(target) || !local(middle) ||
          target.translation().z()<pose.translation().z()-.00001F ||
          (target.translation()-pose.translation()).norm()>.002F ||
          (middle.translation()-pose.translation()).norm()>.002F ||
          (target.translation()-middle.translation()).norm()>.002F ||
          maximumJointDelta(next,q)<1.e-7F || !edge_safe(q,next)) continue;
      result.waypoint_joints.push_back(next); result.waypoint_poses.push_back(target);
      q=next;advanced=true;break;
    }
    if (!advanced) return fail("GUIDED_LIFT_NO_PROGRESS");
  }
  return fail("GUIDED_LIFT_WAYPOINT_BUDGET");
}
}  // namespace vamp_node
