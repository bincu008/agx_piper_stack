#include "vamp_node/analytical_grasp_generator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <limits>
#include <tuple>

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

namespace vamp_node {
namespace {

struct VoxelAccumulator {
  Eigen::Vector3f sum = Eigen::Vector3f::Zero();
  int count = 0;
};

bool finiteUnit(const Eigen::Vector3f &value) {
  return value.allFinite() && value.squaredNorm() > 1.0e-8F;
}

Eigen::Vector3f stableNormal(const std::vector<Eigen::Vector3f> &points,
                             std::size_t index, float radius, int minimum,
                             const Eigen::Vector3f &camera_origin,
                             int *support) {
  std::vector<Eigen::Vector3f> neighbourhood;
  const float radius_sq = radius * radius;
  for (const auto &point : points) {
    if ((point - points[index]).squaredNorm() <= radius_sq) {
      neighbourhood.push_back(point);
    }
  }
  if (support != nullptr) *support = static_cast<int>(neighbourhood.size());
  if (neighbourhood.size() < static_cast<std::size_t>(minimum)) {
    return Eigen::Vector3f::Zero();
  }
  Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
  for (const auto &point : neighbourhood) centroid += point;
  centroid /= static_cast<float>(neighbourhood.size());
  Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
  for (const auto &point : neighbourhood) {
    const Eigen::Vector3f delta = point - centroid;
    covariance.noalias() += delta * delta.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
  if (solver.info() != Eigen::Success) return Eigen::Vector3f::Zero();
  Eigen::Vector3f normal = solver.eigenvectors().col(0);
  if (!finiteUnit(normal)) return Eigen::Vector3f::Zero();
  normal.normalize();
  if (normal.dot(camera_origin - points[index]) < 0.0F) normal = -normal;
  return normal;
}

}  // namespace

AnalyticalGraspGenerator::AnalyticalGraspGenerator(AnalyticalGraspParameters parameters)
    : parameters_(parameters) {}

std::vector<GraspProposal> AnalyticalGraspGenerator::generate(
    const std::vector<Eigen::Vector3f> &target_points,
    const Eigen::Vector3f &seed,
    const Eigen::Vector3f &camera_origin,
    AnalyticalGraspDiagnostics *diagnostics) const {
  AnalyticalGraspDiagnostics local;
  local.isolated_points = target_points.size();
  std::map<std::tuple<int, int, int>, VoxelAccumulator> voxels;
  const float voxel = std::max(0.001F, parameters_.voxel_size_m);
  for (const auto &point : target_points) {
    if (!point.allFinite()) continue;
    const auto key = std::make_tuple(
      static_cast<int>(std::floor(point.x() / voxel)),
      static_cast<int>(std::floor(point.y() / voxel)),
      static_cast<int>(std::floor(point.z() / voxel)));
    auto &entry = voxels[key];
    entry.sum += point;
    ++entry.count;
  }
  std::vector<Eigen::Vector3f> points;
  points.reserve(voxels.size());
  for (const auto &[key, entry] : voxels) {
    (void)key;
    points.push_back(entry.sum / static_cast<float>(entry.count));
  }
  local.downsampled_points = points.size();
  if (points.size() < static_cast<std::size_t>(parameters_.min_normal_neighbors)) {
    if (diagnostics != nullptr) *diagnostics = local;
    return {};
  }

  std::vector<Eigen::Vector3f> normals(points.size(), Eigen::Vector3f::Zero());
  std::vector<int> support(points.size(), 0);
  for (std::size_t i = 0; i < points.size(); ++i) {
    normals[i] = stableNormal(points, i, parameters_.normal_radius_m,
                              parameters_.min_normal_neighbors,
                              camera_origin, &support[i]);
    if (finiteUnit(normals[i])) ++local.normal_points;
  }

  struct Pair {
    std::size_t left;
    std::size_t right;
    float alignment;
    AnalyticalProposalOrigin origin;
    bool model_supported = false;
    Eigen::Vector3f model_left = Eigen::Vector3f::Zero();
    Eigen::Vector3f model_right = Eigen::Vector3f::Zero();
  };
  std::vector<Pair> local_pairs;
  const std::size_t stride = std::max<std::size_t>(1, points.size() / 160);
  for (std::size_t i = 0; i < points.size(); i += stride) {
    if (!finiteUnit(normals[i])) continue;
    int best = -1;
    float best_distance = std::numeric_limits<float>::infinity();
    for (std::size_t j = 0; j < points.size(); ++j) {
      if (i == j || !finiteUnit(normals[j])) continue;
      const Eigen::Vector3f delta = points[j] - points[i];
      const float opening = delta.norm() + parameters_.opening_clearance_m;
      if (opening < parameters_.minimum_opening_m ||
          opening > parameters_.maximum_opening_m) continue;
      const float alignment = -normals[i].dot(normals[j]);
      if (alignment < 0.70F) continue;
      if (delta.norm() < best_distance) {
        best = static_cast<int>(j);
        best_distance = delta.norm();
      }
    }
    if (best >= 0) local_pairs.push_back({i, static_cast<std::size_t>(best),
                                           -normals[i].dot(normals[best]),
                                           AnalyticalProposalOrigin::kLocalPca});
  }

  // Local PCA cross-sections cover every part of the connected target cloud.
  // This is deliberately geometric rather than semantic: a visible handle is
  // a small tube-like neighbourhood and can therefore supply its own contacts.
  std::vector<std::size_t> anchors;
  anchors.reserve(static_cast<std::size_t>(parameters_.local_pca_searches));
  std::size_t first_anchor = 0;
  float first_distance = std::numeric_limits<float>::infinity();
  for (std::size_t index = 0; index < points.size(); ++index) {
    const float distance = (points[index] - seed).squaredNorm();
    if (distance < first_distance) {
      first_distance = distance;
      first_anchor = index;
    }
  }
  anchors.push_back(first_anchor);
  while (anchors.size() < static_cast<std::size_t>(parameters_.local_pca_searches) &&
         anchors.size() < points.size()) {
    std::size_t best = points.size();
    float best_distance = -1.0F;
    for (std::size_t index = 0; index < points.size(); ++index) {
      float nearest = std::numeric_limits<float>::infinity();
      for (const auto anchor : anchors) {
        nearest = std::min(nearest, (points[index] - points[anchor]).squaredNorm());
      }
      if (nearest > best_distance + 1.0e-9F) {
        best = index;
        best_distance = nearest;
      }
    }
    if (best == points.size()) break;
    anchors.push_back(best);
  }
  const float local_radius = std::max(0.025F, parameters_.normal_radius_m * 3.0F);
  const float local_radius_sq = local_radius * local_radius;
  for (const auto anchor : anchors) {
    std::vector<std::size_t> neighbourhood;
    Eigen::Vector3f local_centroid = Eigen::Vector3f::Zero();
    for (std::size_t index = 0; index < points.size(); ++index) {
      if ((points[index] - points[anchor]).squaredNorm() > local_radius_sq) continue;
      neighbourhood.push_back(index);
      local_centroid += points[index];
    }
    if (neighbourhood.size() < static_cast<std::size_t>(parameters_.min_normal_neighbors)) continue;
    local_centroid /= static_cast<float>(neighbourhood.size());
    Eigen::Matrix3f local_covariance = Eigen::Matrix3f::Zero();
    for (const auto index : neighbourhood) {
      const Eigen::Vector3f delta = points[index] - local_centroid;
      local_covariance.noalias() += delta * delta.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> local_pca(local_covariance);
    if (local_pca.info() != Eigen::Success) continue;
    // The two axes perpendicular to the local major tangent define possible
    // tube/body cross-sections. Select the narrowest physically legal span.
    float selected_opening = std::numeric_limits<float>::infinity();
    std::size_t selected_left = 0;
    std::size_t selected_right = 0;
    for (int axis_index : {0, 1}) {
      const Eigen::Vector3f axis = local_pca.eigenvectors().col(axis_index).normalized();
      int minimum = -1;
      int maximum = -1;
      float minimum_projection = std::numeric_limits<float>::infinity();
      float maximum_projection = -std::numeric_limits<float>::infinity();
      for (const auto index : neighbourhood) {
        const float projection = axis.dot(points[index] - local_centroid);
        if (projection < minimum_projection) {
          minimum_projection = projection;
          minimum = static_cast<int>(index);
        }
        if (projection > maximum_projection) {
          maximum_projection = projection;
          maximum = static_cast<int>(index);
        }
      }
      if (minimum < 0 || maximum < 0 || minimum == maximum) continue;
      const float opening = (points[maximum] - points[minimum]).norm() +
        parameters_.opening_clearance_m;
      if (opening >= parameters_.minimum_opening_m &&
          opening <= parameters_.maximum_opening_m && opening < selected_opening) {
        selected_opening = opening;
        selected_left = static_cast<std::size_t>(maximum);
        selected_right = static_cast<std::size_t>(minimum);
      }
    }
    if (std::isfinite(selected_opening)) {
      local_pairs.push_back({selected_left, selected_right, 0.80F,
                             AnalyticalProposalOrigin::kLocalPca});
    }
  }

  // Fit circular cross-sections in deterministic global-PCA body bands.  The
  // previous extent-pair implementation selected rim/silhouette corners from
  // partial cup clouds, producing edge grasps rather than opposing body walls.
  Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
  for (const auto &point : points) centroid += point;
  centroid /= static_cast<float>(points.size());
  Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
  for (const auto &point : points) {
    const Eigen::Vector3f delta = point - centroid;
    covariance.noalias() += delta * delta.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> pca(covariance);
  std::array<Eigen::Vector3f, 3> local_axes{
    Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY(), Eigen::Vector3f::UnitZ()};
  std::vector<Pair> global_pairs;
  if (pca.info() == Eigen::Success) {
    for (int axis_index = 0; axis_index < 3; ++axis_index) {
      Eigen::Vector3f axis = pca.eigenvectors().col(axis_index).normalized();
      local_axes[axis_index] = axis;
    }
    // Fixed interior bands avoid spending the proposal budget on rim ends.
    const Eigen::Vector3f dominant_axis = local_axes[2];
    const Eigen::Vector3f plane_u = local_axes[0];
    const Eigen::Vector3f plane_v = local_axes[1];
    float dominant_min = std::numeric_limits<float>::infinity();
    float dominant_max = -std::numeric_limits<float>::infinity();
    for (const auto &point : points) {
      const float projection = dominant_axis.dot(point - centroid);
      dominant_min = std::min(dominant_min, projection);
      dominant_max = std::max(dominant_max, projection);
    }
    const int bands = std::max(1, std::min(10, parameters_.global_pca_searches));
    const int directions = std::max(1, (parameters_.global_pca_searches + bands - 1) / bands);
    const float band_half_width = std::max(0.010F,
      (dominant_max - dominant_min) / static_cast<float>(bands * 2));
    for (int band = 0; band < bands; ++band) {
      const float fraction = 0.15F + 0.70F *
        (static_cast<float>(band) + 0.5F) / static_cast<float>(bands);
      const float band_center = dominant_min + fraction * (dominant_max - dominant_min);
      std::vector<std::size_t> band_points;
      Eigen::Vector3f band_centroid = Eigen::Vector3f::Zero();
      for (std::size_t index = 0; index < points.size(); ++index) {
        const Eigen::Vector3f relative = points[index] - centroid;
        if (std::abs(dominant_axis.dot(relative) - band_center) > band_half_width) continue;
        band_points.push_back(index);
        band_centroid += points[index];
      }
      if (band_points.size() < static_cast<std::size_t>(parameters_.min_normal_neighbors * 2)) continue;
      band_centroid /= static_cast<float>(band_points.size());
      Eigen::MatrixXf fit_matrix(static_cast<Eigen::Index>(band_points.size()), 3);
      Eigen::VectorXf fit_rhs(static_cast<Eigen::Index>(band_points.size()));
      for (std::size_t row = 0; row < band_points.size(); ++row) {
        const Eigen::Vector3f delta = points[band_points[row]] - band_centroid;
        const float u = plane_u.dot(delta);
        const float v = plane_v.dot(delta);
        fit_matrix(static_cast<Eigen::Index>(row), 0) = 2.0F * u;
        fit_matrix(static_cast<Eigen::Index>(row), 1) = 2.0F * v;
        fit_matrix(static_cast<Eigen::Index>(row), 2) = 1.0F;
        fit_rhs(static_cast<Eigen::Index>(row)) = u * u + v * v;
      }
      const Eigen::Vector3f circle = fit_matrix.colPivHouseholderQr().solve(fit_rhs);
      if (!circle.allFinite()) continue;
      const float radius_sq = circle.z() + circle.x() * circle.x() + circle.y() * circle.y();
      if (!std::isfinite(radius_sq) || radius_sq <= 0.0F) continue;
      const float radius = std::sqrt(radius_sq);
      if (radius < 0.006F ||
          2.0F * radius + parameters_.opening_clearance_m > parameters_.maximum_opening_m) continue;
      const Eigen::Vector3f cross_center = band_centroid +
        plane_u * circle.x() + plane_v * circle.y();
      for (int direction = 0; direction < directions; ++direction) {
        if (global_pairs.size() >= static_cast<std::size_t>(parameters_.global_pca_searches)) break;
        const float angle = static_cast<float>(2.0 * M_PI * direction / directions);
        const Eigen::Vector3f closing_axis =
          (std::cos(angle) * plane_u + std::sin(angle) * plane_v).normalized();
        const Eigen::Vector3f model_left = cross_center + radius * closing_axis;
        const Eigen::Vector3f model_right = cross_center - radius * closing_axis;
        const float opening = (model_left - model_right).norm() +
          parameters_.opening_clearance_m;
        if (opening >= parameters_.minimum_opening_m &&
            opening <= parameters_.maximum_opening_m) {
          std::size_t nearest_left = band_points.front();
          std::size_t nearest_right = band_points.front();
          float left_distance = std::numeric_limits<float>::infinity();
          float right_distance = std::numeric_limits<float>::infinity();
          for (const auto index : band_points) {
            const float candidate_left = (points[index] - model_left).squaredNorm();
            const float candidate_right = (points[index] - model_right).squaredNorm();
            if (candidate_left < left_distance) { left_distance = candidate_left; nearest_left = index; }
            if (candidate_right < right_distance) { right_distance = candidate_right; nearest_right = index; }
          }
          global_pairs.push_back({nearest_left, nearest_right, 0.75F,
                                  AnalyticalProposalOrigin::kGlobalPca, true,
                                  model_left, model_right});
        }
      }
    }
  }
  local.local_contact_pairs = local_pairs.size();
  local.global_contact_pairs = global_pairs.size();
  local.contact_pairs = local.local_contact_pairs + local.global_contact_pairs;

  // Select contact families before expanding fixed approach rolls.  Expanding
  // every roll for the first contact pair made a single cup midpoint consume
  // the proposal budget.  This is deterministic farthest-point sampling: the
  // first family is the best normal/seed match, then each later family is the
  // farthest remaining contact midpoint, with stable quality/index ties.
  const auto pair_left = [&points](const Pair &pair) {
    return pair.model_supported ? pair.model_left : points[pair.left];
  };
  const auto pair_right = [&points](const Pair &pair) {
    return pair.model_supported ? pair.model_right : points[pair.right];
  };
  const auto pair_center = [&pair_left, &pair_right](const Pair &pair) {
    return (pair_left(pair) + pair_right(pair)) * 0.5F;
  };
  const auto pair_priority = [&pair_center, &seed](const Pair &pair) {
    return pair.alignment - 0.20F * std::min(1.0F,
      (pair_center(pair) - seed).norm() / 0.14F);
  };
  const auto select_spatial_families = [&](const std::vector<Pair> &pairs, const int limit) {
    std::vector<Pair> selected_pairs;
    std::vector<bool> pair_selected(pairs.size(), false);
    const std::size_t family_limit = std::min(
      pairs.size(), static_cast<std::size_t>(std::max(1, limit)));
    for (std::size_t family = 0; family < family_limit; ++family) {
      std::size_t best_index = pairs.size();
      float best_distance = -1.0F;
      float best_priority = -std::numeric_limits<float>::infinity();
      bool best_is_separated = false;
      for (std::size_t index = 0; index < pairs.size(); ++index) {
        if (pair_selected[index]) continue;
        const Eigen::Vector3f center = pair_center(pairs[index]);
        float nearest_distance = std::numeric_limits<float>::infinity();
        for (const auto &chosen : selected_pairs) {
        nearest_distance = std::min(nearest_distance,
          (center - pair_center(chosen)).norm());
        }
        const bool is_separated = selected_pairs.empty() ||
        nearest_distance >= parameters_.min_contact_center_separation_m;
        const float priority = pair_priority(pairs[index]);
        const bool choose_first = selected_pairs.empty() &&
        (best_index == pairs.size() || priority > best_priority + 1.0e-6F);
        const bool choose_spatial = !selected_pairs.empty() &&
        (best_index == pairs.size() ||
         (is_separated && !best_is_separated) ||
         (is_separated == best_is_separated &&
          (nearest_distance > best_distance + 1.0e-6F ||
           (std::abs(nearest_distance - best_distance) <= 1.0e-6F &&
            priority > best_priority + 1.0e-6F))));
        if (choose_first || choose_spatial) {
        best_index = index;
        best_distance = nearest_distance;
        best_priority = priority;
        best_is_separated = is_separated;
        }
      }
      if (best_index == pairs.size()) break;
      pair_selected[best_index] = true;
      selected_pairs.push_back(pairs[best_index]);
    }
    return selected_pairs;
  };
  const auto selected_local_pairs = select_spatial_families(
    local_pairs, parameters_.local_pca_searches);
  const auto selected_global_pairs = select_spatial_families(
    global_pairs, parameters_.global_pca_searches);
  std::vector<Pair> selected_pairs;
  selected_pairs.reserve(selected_local_pairs.size() + selected_global_pairs.size());
  for (std::size_t index = 0;
       index < std::max(selected_local_pairs.size(), selected_global_pairs.size()); ++index) {
    if (index < selected_local_pairs.size()) selected_pairs.push_back(selected_local_pairs[index]);
    if (index < selected_global_pairs.size()) selected_pairs.push_back(selected_global_pairs[index]);
  }
  local.selected_contact_families = selected_pairs.size();
  local.selected_local_families = selected_local_pairs.size();
  local.selected_global_families = selected_global_pairs.size();

  std::vector<GraspProposal> proposals;
  const float duplicate_cosine = std::cos(
    parameters_.dedup_angle_degrees * static_cast<float>(M_PI) / 180.0F);
  const auto add = [&](GraspProposal proposal) {
    if (proposals.size() >= static_cast<std::size_t>(parameters_.maximum_proposals)) return;
    for (const auto &existing : proposals) {
      const Eigen::Vector3f old_center = (existing.contact_left + existing.contact_right) * 0.5F;
      const Eigen::Vector3f new_center = (proposal.contact_left + proposal.contact_right) * 0.5F;
      if ((old_center - new_center).norm() <= parameters_.dedup_position_m &&
          std::abs(existing.closing.dot(proposal.closing)) >= duplicate_cosine &&
          existing.approach.dot(proposal.approach) >= duplicate_cosine) {
        ++local.deduplicated;
        return;
      }
    }
    proposals.push_back(std::move(proposal));
  };

  for (std::size_t family_index = 0; family_index < selected_pairs.size(); ++family_index) {
    const auto &pair = selected_pairs[family_index];
    const Eigen::Vector3f left = pair_left(pair);
    const Eigen::Vector3f right = pair_right(pair);
    Eigen::Vector3f closing = left - right;
    const float measured_opening = closing.norm();
    const float opening = measured_opening + parameters_.opening_clearance_m;
    if (!finiteUnit(closing) || opening < parameters_.minimum_opening_m ||
        opening > parameters_.maximum_opening_m) continue;
    closing.normalize();
    const Eigen::Vector3f center = (left + right) * 0.5F;
    Eigen::Vector3f surface_tangent = local_axes[0];
    for (const auto &axis : local_axes) {
      if (std::abs(axis.dot(closing)) < std::abs(surface_tangent.dot(closing))) {
        surface_tangent = axis;
      }
    }
    const std::array<Eigen::Vector3f, 3> approach_bases = {
      Eigen::Vector3f(0.0F, 0.0F, -1.0F),
      center - camera_origin,
      surface_tangent};
    std::vector<Eigen::Vector3f> usable_approach_bases;
    for (auto approach_base : approach_bases) {
      approach_base -= closing * approach_base.dot(closing);
      if (!finiteUnit(approach_base)) continue;
      approach_base.normalize();
      bool duplicate_base = false;
      for (const auto &existing : usable_approach_bases) {
        if (existing.dot(approach_base) > 0.995F) {
          duplicate_base = true;
          break;
        }
      }
      if (!duplicate_base) usable_approach_bases.push_back(approach_base);
    }
    std::vector<float> rolls{0.0F};
    const float roll_step = std::max(1.0F, parameters_.roll_step_degrees);
    for (float roll = roll_step; roll <= 45.001F; roll += roll_step) {
      rolls.push_back(-roll);
      rolls.push_back(roll);
    }
    int variants_emitted = 0;
    for (const float roll : rolls) {
      for (const auto &approach_base : usable_approach_bases) {
        if (variants_emitted >= parameters_.maximum_approach_variants_per_pair ||
            proposals.size() >= static_cast<std::size_t>(parameters_.maximum_proposals)) {
          break;
        }
        Eigen::Vector3f approach = Eigen::AngleAxisf(
          roll * static_cast<float>(M_PI) / 180.0F, closing) * approach_base;
        approach -= closing * approach.dot(closing);
        if (!finiteUnit(approach)) continue;
        approach.normalize();
        Eigen::Vector3f height = approach.cross(closing);
        if (!finiteUnit(height)) continue;
        height.normalize();
        const float normal_quality = std::clamp(pair.alignment, 0.0F, 1.0F);
        const float support_quality = pair.model_supported ? 0.75F : std::min(1.0F,
          static_cast<float>(support[pair.left] + support[pair.right]) /
            static_cast<float>(parameters_.min_normal_neighbors * 4));
        const float seed_quality = 1.0F - std::clamp(
          (center - seed).norm() / 0.14F, 0.0F, 1.0F);
        const float opening_quality = 1.0F - std::abs(
          opening - 0.5F * (parameters_.minimum_opening_m + parameters_.maximum_opening_m)) /
          (0.5F * (parameters_.maximum_opening_m - parameters_.minimum_opening_m));
        const float normal_symmetry = pair.model_supported ? 0.75F : finiteUnit(normals[pair.left]) &&
          finiteUnit(normals[pair.right]) ? 1.0F - std::clamp(
          (normals[pair.left] + normals[pair.right]).norm() * 0.5F,
          0.0F, 1.0F) : 0.50F;
        int finger_body_points = 0;
        const float half_opening = opening * 0.5F;
        for (const auto &point : points) {
          const Eigen::Vector3f local = point - center;
          const float side = std::abs(local.dot(closing));
          const float vertical = std::abs(local.dot(height));
          const float depth = local.dot(approach);
          if (side > half_opening &&
              side <= half_opening + parameters_.finger_width_m &&
              vertical <= parameters_.finger_height_m * 0.5F &&
              depth >= -parameters_.contact_depth_m &&
              depth <= parameters_.finger_length_m - parameters_.contact_depth_m) {
            ++finger_body_points;
          }
        }
        const float finger_clearance_quality = 1.0F / (1.0F +
          static_cast<float>(finger_body_points) / 8.0F);
        GraspProposal proposal;
        proposal.sample = center;
        proposal.hand_base = center - approach * parameters_.contact_depth_m;
        proposal.approach = approach;
        proposal.closing = closing;
        proposal.height_axis = height;
        proposal.contact_left = left;
        proposal.contact_right = right;
        proposal.opening_m = opening;
        proposal.quality = 0.25F * normal_quality + 0.15F * normal_symmetry +
          0.15F * support_quality + 0.20F * seed_quality +
          0.15F * finger_clearance_quality +
          0.10F * std::max(0.0F, opening_quality);
        proposal.has_contacts = true;
        proposal.model_supported_contacts = pair.model_supported;
        proposal.half_antipodal = pair.alignment >= 0.70F;
        proposal.full_antipodal = pair.alignment >= 0.90F;
        proposal.source = GraspProposalSource::kAnalytical;
        proposal.analytical_origin = pair.origin;
        proposal.family_index = family_index;
        proposal.variant_index = static_cast<std::size_t>(variants_emitted);
        const std::size_t proposal_count_before = proposals.size();
        add(std::move(proposal));
        if (proposals.size() > proposal_count_before) ++variants_emitted;
      }
      if (variants_emitted >= parameters_.maximum_approach_variants_per_pair ||
          proposals.size() >= static_cast<std::size_t>(parameters_.maximum_proposals)) break;
    }
    if (proposals.size() >= static_cast<std::size_t>(parameters_.maximum_proposals)) break;
  }
  // Validation consumes one variant from every local/global family before it
  // returns for a second variant, preventing a single point from dominating.
  std::stable_sort(proposals.begin(), proposals.end(),
    [](const GraspProposal &left, const GraspProposal &right) {
      if (left.variant_index != right.variant_index) {
        return left.variant_index < right.variant_index;
      }
      return left.family_index < right.family_index;
    });
  local.generated = proposals.size();
  if (diagnostics != nullptr) *diagnostics = local;
  return proposals;
}

}  // namespace vamp_node
