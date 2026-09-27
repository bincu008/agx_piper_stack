#include <vector>
#include <array>
#include <utility>
#include <iostream>
#include <iomanip>

#include <vamp/collision/factory.hh>
#include <vamp/collision/filter_centervox.hh>
#include <vamp/planning/validate.hh>
#include <vamp/planning/rrtc.hh>
#include <vamp/planning/simplify.hh>
#include <vamp/robots/panda.hh>
#include <vamp/random/halton.hh>

using Robot = vamp::robots::Panda;
static constexpr const std::size_t rake = vamp::FloatVectorWidth;
using EnvironmentInput = vamp::collision::Environment<float>;
using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;
using RRTC = vamp::planning::RRTC<Robot, rake, Robot::resolution>;

// Start and goal configurations
static constexpr Robot::ConfigurationArray start = {0., -0.785, 0., -2.356, 0., 1.571, 0.785};
static constexpr Robot::ConfigurationArray goal = {2.35, 1., 0., -0.8, 0, 2.5, 0.785};

// // Spheres for the cage problem - (x, y, z) center coordinates with fixed, common radius defined below
// static const std::vector<std::array<float, 3>> problem = {
//     {0.55, 0, 0.25},
//     {0.35, 0.35, 0.25},
//     {0, 0.55, 0.25},
//     {-0.55, 0, 0.25},
//     {-0.35, -0.35, 0.25},
//     {0, -0.55, 0.25},
//     {0.35, -0.35, 0.25},
//     {0.35, 0.35, 0.8},
//     {0, 0.55, 0.8},
//     {-0.35, 0.35, 0.8},
//     {-0.55, 0, 0.8},
//     {-0.35, -0.35, 0.8},
//     {0, -0.55, 0.8},
//     {0.35, -0.35, 0.8},
// };

auto main(int, char **) -> int
{
    const std::vector<vamp::collision::Point> pc = {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {0.5, 0.5, 0.5}, {2.0, 2.0, 2.0}, {1.1, 1.1, 1.1}}; // 如果 (2,2,2) 沒被成功 filter，建 MVT 時就會有 warinig
    std::vector<vamp::collision::Point> filtered_pc;
    float r_min = 0.012; // 手臂本身是由很多 sphere 近似的，這是 panda 的 sphere 最小半徑
    float r_max = 0.06;
    vamp::collision::Point workspace_aabb_min = {-1.19, -1.19, -1.19}; // 手臂最遠可觸及的範圍會被一個 AABB 匡起來，這是 panda 的 min corner
    vamp::collision::Point workspace_aabb_max = {1.19, 1.19, 1.19}; // Assumption: panda 基座在 {0.0, 0.0, 0.0}
    vamp::collision::Point origin = {0.0, 0.0, 0.0}; // Assumption: panda 基座在 {0.0, 0.0, 0.0}
    float r_point = 0.0025; // 點雲裡的 point 半徑，不太重要
    
    // Build environment
    EnvironmentInput environment;
    
    filtered_pc = vamp::collision::filter_pointcloud_centervox(pc, 0.01, 1.19, origin, workspace_aabb_min, workspace_aabb_max);
    for (auto p : pc) {
        std::cout << p[0] << ", " << p[1] << ", " << p[2] << std::endl;
    }
    std::cout << std::endl;
    for (auto p : filtered_pc) {
        std::cout << p[0] << ", " << p[1] << ", " << p[2] << std::endl;
    }

    environment.mvt_pointclouds.emplace_back(filtered_pc, r_min, r_max, workspace_aabb_min, workspace_aabb_max, r_point);

    auto env_v = EnvironmentVector(environment);

    // Create RNG for planning
    auto rng = std::make_shared<vamp::rng::Halton<Robot>>();

    // Setup RRTC and plan
    vamp::planning::RRTCSettings rrtc_settings;
    rrtc_settings.range = 1.0;

    auto result =
        RRTC::solve(Robot::Configuration(start), Robot::Configuration(goal), env_v, rrtc_settings, rng);

    // If successful
    if (result.path.size() > 0)
    {
        // Simplify path with default settings
        vamp::planning::SimplifySettings simplify_settings;
        auto simplify_result = vamp::planning::simplify<Robot, rake, Robot::resolution>(
            result.path, env_v, simplify_settings, rng);

        // Output configurations of simplified path
        std::cout << std::fixed << std::setprecision(3);
        for (const auto &config : simplify_result.path)
        {
            const auto &array = config.to_array();
            for (auto i = 0U; i < Robot::dimension; ++i)
            {
                std::cout << array[i] << ", ";
            }

            std::cout << std::endl;
        }
    }
    std::cout << "Planning failed" << std::endl;

    return 0;
}
