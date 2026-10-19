#pragma once
#include <array>
#include <vector>
#include <unordered_set>
#include <cstddef>

namespace vision_node
{
using VoxelCell = std::array<int, 3>;
struct VoxelCellHash
{
  std::size_t operator()(const VoxelCell &cell) const
  {
    std::size_t hash = 0;
    for (const auto value : cell) hash ^= std::hash<int>{}(value)+0x9e3779b9+(hash<<6)+(hash>>2);
    return hash;
  }
};
inline std::vector<bool> supportedVoxels(const std::vector<VoxelCell> &cells)
{
  const std::unordered_set<VoxelCell, VoxelCellHash> occupied(cells.begin(), cells.end());
  std::vector<bool> keep;
  keep.reserve(cells.size());
  for (const auto &cell : cells) {
    int neighbors = 0;
    for (int x=-1; x<=1 && neighbors<2; ++x)
      for (int y=-1; y<=1 && neighbors<2; ++y)
        for (int z=-1; z<=1 && neighbors<2; ++z) {
          if (x==0 && y==0 && z==0) continue;
          neighbors += occupied.count({cell[0]+x,cell[1]+y,cell[2]+z});
        }
    keep.push_back(neighbors>=2);
  }
  return keep;
}
}
