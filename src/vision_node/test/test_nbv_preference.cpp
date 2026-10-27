#include "vision_node/nbv_preference.hpp"
#include <cstdlib>
#include <iostream>
#define REQUIRE(condition) do { if (!(condition)) { std::cerr << #condition << " failed\n"; return EXIT_FAILURE; } } while (false)
int main() {
  const Eigen::Vector3f target(.4F,-.1F,0.F), current(.5F,-.1F,.05F);
  const auto direction=vision_node::baseFacingViewDirection(target);
  REQUIRE(direction.z()>.96F);
  REQUIRE(direction.head<2>().dot(target.head<2>())<0.F);
  const Eigen::Vector3f good=target+.45F*direction;
  const Eigen::Vector3f side=target+Eigen::Vector3f(.45F,0.F,0.F);
  REQUIRE(vision_node::nextViewCost(good,target,current)<
         vision_node::nextViewCost(side,target,current));
  const Eigen::Vector3f opposite=target+Eigen::Vector3f(-.45F*direction.x(),-.45F*direction.y(),.45F*direction.z());
  REQUIRE(vision_node::nextViewCost(good,target,current)<
         vision_node::nextViewCost(opposite,target,current));
  const auto samples=vision_node::nextViewSamples(target,.45F);
  REQUIRE(samples.size()==156);
  for (const auto &p:samples) REQUIRE(std::abs((p-target).norm()-.45F)<1.e-5F);
  REQUIRE(vision_node::nextViewSamples(target,0.F).empty());
  REQUIRE(vision_node::baseFacingViewDirection(Eigen::Vector3f::Zero()).allFinite());
}
