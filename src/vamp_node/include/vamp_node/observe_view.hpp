#pragma once

#include <Eigen/Geometry>
#include <array>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

namespace vamp_node {

struct ObserveViewResult {
    std::vector<float> joints;
    Eigen::Isometry3f requested = Eigen::Isometry3f::Identity();
    Eigen::Isometry3f achieved = Eigen::Isometry3f::Identity();
    bool moved = false;
    bool endpoint_safe = false;
    std::string status = "NO_REACHABLE_CHANGE";
    float fraction = 0.F;
    unsigned tested = 0;
};

inline Eigen::Isometry3f relativeCameraView(
    const Eigen::Isometry3f &camera, const std::array<float, 6> &delta, float fraction)
{
    Eigen::Isometry3f result = camera;
    result.translation() += camera.linear() * Eigen::Vector3f(
        delta[0], delta[1], delta[2]) * fraction;
    result.linear() = camera.linear() * (
        Eigen::AngleAxisf(delta[5] * fraction, Eigen::Vector3f::UnitZ()) *
        Eigen::AngleAxisf(delta[4] * fraction, Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(delta[3] * fraction, Eigen::Vector3f::UnitX())).toRotationMatrix();
    return result;
}

// Solve and collision-check endpoints only. No waypoint or actuator execution.
template <typename Solve, typename Safe, typename FK>
ObserveViewResult calculateObserveView(
    const std::vector<float> &start, const Eigen::Isometry3f &camera,
    const Eigen::Isometry3f &hand_eye, const std::array<float, 6> &delta,
    Solve solve, Safe safe, FK fk)
{
    ObserveViewResult result;
    result.joints = start;
    result.requested = relativeCameraView(camera, delta, 1.F);
    result.achieved = camera;
    bool nonzero = false;
    for (float value : delta) nonzero |= std::abs(value) > 1.e-7F;
    if (!nonzero) {result.status = "NO_CHANGE"; return result;}
    const auto began = std::chrono::steady_clock::now();
    for (float fraction : {1.F, .5F, .25F}) {
        const auto desired_camera = relativeCameraView(camera, delta, fraction);
        for (float angular_tolerance : {.025F, .35F}) {
            if (std::chrono::duration<double>(
                std::chrono::steady_clock::now()-began).count() > 3.) return result;
            Eigen::Isometry3f target = desired_camera * hand_eye.inverse();
            std::vector<float> solved;
            ++result.tested;
            if (!solve(target, start, angular_tolerance, solved) || solved.size() != 6) continue;
            auto achieved = fk(solved) * hand_eye;
            // Loose wrist rotation must not shift the camera far through its lever arm.
            if ((achieved.translation()-desired_camera.translation()).norm() > .003F) {
                target.translation() = desired_camera.translation() -
                    fk(solved).linear() * hand_eye.translation();
                auto seed = solved;
                ++result.tested;
                if (!solve(target, seed, angular_tolerance, solved)) continue;
                achieved = fk(solved) * hand_eye;
            }
            if ((achieved.translation()-desired_camera.translation()).norm() > .005F ||
                Eigen::AngleAxisf(desired_camera.linear().transpose()*achieved.linear()).angle() >
                    angular_tolerance + .01F) continue;
            float change = 0.F;
            for (size_t j=0; j<start.size(); ++j) change += std::abs(solved[j]-start[j]);
            if (change < 1.e-5F) continue;
            if (!safe(solved)) {
                result.status = "TARGET_COLLISION";
                return result;  // An unsafe endpoint always means stay, not a hidden detour.
            }
            result.joints = solved;
            result.achieved = achieved;
            result.fraction = fraction;
            result.moved = result.endpoint_safe = true;
            result.status = fraction < 1.F ? "PARTIAL" :
                angular_tolerance > .03F ? "RELAXED" : "EXACT";
            return result;
        }
    }
    return result;
}
}  // namespace vamp_node
