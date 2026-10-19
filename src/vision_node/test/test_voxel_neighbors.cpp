#include "vision_node/voxel_neighbors.hpp"
#include <stdexcept>

int main()
{
  using vision_node::supportedVoxels;
  if (supportedVoxels({{0,0,0}}) != std::vector<bool>{false}) throw std::runtime_error("singleton");
  if (supportedVoxels({{0,0,0},{1,0,0}}) != std::vector<bool>{false,false}) throw std::runtime_error("pair");
  if (supportedVoxels({{0,0,0},{1,0,0},{2,0,0}}) != std::vector<bool>{false,true,false})
    throw std::runtime_error("one pass must retain center without recursive erosion");
  if (supportedVoxels({{0,0,0},{1,1,1},{1,0,0},{10,10,10}}) !=
      std::vector<bool>{true,true,true,false}) throw std::runtime_error("diagonal neighbors");
  if (supportedVoxels({{0,0,0},{0,0,0},{1,0,0}}) != std::vector<bool>{false,false,false})
    throw std::runtime_error("duplicate cells do not count as neighbors");
}
