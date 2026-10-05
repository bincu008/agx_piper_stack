#include <gtest/gtest.h>
#include "vamp_node/pipeline_timing.hpp"
#include <thread>

using namespace vamp_node;

TEST(PipelineTiming, NestedTimersCountCallsWithoutLeavingContext) {
    std::string output;
    {
        PipelineProfile profile(true, "test", [&](const auto &data) { output = data; });
        {
            ComponentTimer outer(PipelineComponent::Collision);
            ComponentTimer inner(PipelineComponent::Collision);
            EXPECT_EQ(active_pipeline_metrics->calls[1], 2u);
            EXPECT_EQ(active_pipeline_metrics->depth[1], 2u);
        }
        EXPECT_EQ(active_pipeline_metrics->depth[1], 0u);
        EXPECT_GE(active_pipeline_metrics->seconds[1], 0.0);
    }
    EXPECT_EQ(active_pipeline_metrics, nullptr);
    EXPECT_NE(output.find("\"collision_calls\":2"), std::string::npos);
}

TEST(PipelineTiming, DisabledProfileDoesNotRecordAndThreadsAreIsolated) {
    bool emitted = false;
    PipelineProfile disabled(false, "disabled", [&](const auto &) { emitted = true; });
    ComponentTimer timer(PipelineComponent::NumericIK);
    EXPECT_EQ(active_pipeline_metrics, nullptr);
    {
        PipelineProfile enabled(true, "enabled", [](const auto &) {});
        std::thread other([]() { EXPECT_EQ(active_pipeline_metrics, nullptr); });
        other.join();
    }
    EXPECT_FALSE(emitted);
}
