#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>

namespace vamp_node {
enum class PipelineComponent { NumericIK, Collision, HGGD, RRTC, Count };
constexpr std::size_t component_count = static_cast<std::size_t>(PipelineComponent::Count);
struct PipelineMetrics {
    std::array<double, component_count> seconds{};
    std::array<std::uint64_t, component_count> calls{};
    std::array<unsigned, component_count> depth{};
    std::uint64_t pose_ik_calls = 0;
    std::uint64_t ik_iterations = 0;
    std::uint64_t ik_fk_evaluations = 0;
};
inline thread_local PipelineMetrics *active_pipeline_metrics = nullptr;

// Nested checks in the same category are counted, but their time is not
// counted twice. Different categories (e.g. RRTC containing collision) overlap.
class ComponentTimer {
public:
    explicit ComponentTimer(PipelineComponent component)
        : metrics_(active_pipeline_metrics), index_(static_cast<std::size_t>(component)) {
        if (metrics_) {
            ++metrics_->calls[index_];
            outer_ = metrics_->depth[index_]++ == 0;
            if (outer_) started_ = std::chrono::steady_clock::now();
        }
    }
    ~ComponentTimer() {
        if (metrics_) {
            --metrics_->depth[index_];
            if (outer_) metrics_->seconds[index_] += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started_).count();
        }
    }
    ComponentTimer(const ComponentTimer &) = delete;
    ComponentTimer &operator=(const ComponentTimer &) = delete;
private:
    PipelineMetrics *metrics_;
    std::size_t index_;
    bool outer_ = false;
    std::chrono::steady_clock::time_point started_;
};

class PipelineProfile {
public:
    PipelineProfile(bool enabled, std::string stage,
                    std::function<void(const std::string &)> output)
        : enabled_(enabled), stage_(std::move(stage)), output_(std::move(output)),
          previous_(active_pipeline_metrics), started_(std::chrono::steady_clock::now()) {
        if (enabled_) active_pipeline_metrics = &metrics_;
    }
    ~PipelineProfile() {
        if (!enabled_) return;
        active_pipeline_metrics = previous_;
        std::ostringstream result;
        result << "{\"stage\":\"" << stage_ << "\",\"total_seconds\":"
               << std::chrono::duration<double>(std::chrono::steady_clock::now()-started_).count();
        constexpr const char *names[] = {"numeric_ik", "collision", "hggd_detection", "rrtc"};
        for (std::size_t i=0; i<component_count; ++i) {
            result << ",\"" << names[i] << "_seconds\":" << metrics_.seconds[i]
                   << ",\"" << names[i] << "_calls\":" << metrics_.calls[i];
        }
        result << ",\"pose_ik_calls\":" << metrics_.pose_ik_calls
               << ",\"ik_iterations\":" << metrics_.ik_iterations
               << ",\"ik_fk_evaluations\":" << metrics_.ik_fk_evaluations << "}";
        output_(result.str());
    }
    PipelineProfile(const PipelineProfile &) = delete;
    PipelineProfile &operator=(const PipelineProfile &) = delete;
private:
    bool enabled_;
    std::string stage_;
    std::function<void(const std::string &)> output_;
    PipelineMetrics metrics_;
    PipelineMetrics *previous_;
    std::chrono::steady_clock::time_point started_;
};
}  // namespace vamp_node
