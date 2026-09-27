"""Point-cloud preprocessing and iterative AnyPlace pose inference."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from .model import native_farthest_point_indices


@dataclass
class PlacementCandidate:
    """One inferred object pose in the destination cloud frame."""

    rotation: np.ndarray
    translation: np.ndarray
    score: float


def downsample_points(points: np.ndarray, count: int) -> np.ndarray:
    """Clean and deterministically farthest-point sample an XYZ cloud."""
    clean = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    clean = clean[np.isfinite(clean).all(axis=1)]
    if not len(clean):
        raise ValueError("point cloud contains no finite XYZ samples")
    tensor = torch.from_numpy(clean).unsqueeze(0)
    indices = native_farthest_point_indices(tensor, count)[0].cpu().numpy()
    sampled = clean[indices]
    if len(sampled) < count:
        repeats = int(math.ceil(count / len(sampled)))
        sampled = np.tile(sampled, (repeats, 1))[:count]
    return sampled.astype(np.float32, copy=False)


def make_dummy_box_cloud(
        dimensions=(0.04, 0.03, 0.02), point_count=200,
        seed=7) -> np.ndarray:
    """Sample a centered rectangular object's six surfaces."""
    rng = np.random.default_rng(seed)
    dimensions = np.asarray(dimensions, dtype=np.float32)
    if dimensions.shape != (3,) or np.any(dimensions <= 0):
        raise ValueError("dummy object dimensions must contain three positive values")
    points = np.empty((point_count, 3), dtype=np.float32)
    faces = rng.integers(0, 6, size=point_count)
    points[:] = rng.uniform(-0.5, 0.5, size=(point_count, 3)) * dimensions
    for index, face in enumerate(faces):
        axis = face // 2
        points[index, axis] = dimensions[axis] * (0.5 if face % 2 else -0.5)
    return points


def random_rotation_matrices(count: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample SO(3) using normalized random quaternions."""
    quaternion = rng.normal(size=(count, 4))
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    w, x, y, z = quaternion.T
    matrices = np.empty((count, 3, 3), dtype=np.float32)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - z * w)
    matrices[:, 0, 2] = 2 * (x * z + y * w)
    matrices[:, 1, 0] = 2 * (x * y + z * w)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - x * w)
    matrices[:, 2, 0] = 2 * (x * z - y * w)
    matrices[:, 2, 1] = 2 * (y * z + x * w)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def diffusion_timestep_schedule(iterations: int, levels=5, factor=20) -> list[int]:
    """Reproduce AnyPlace's timestep allocation across refinement iterations."""
    if iterations <= 0:
        return []
    weights = factor ** np.arange(1, levels + 1, dtype=np.float64)
    nominal = np.ceil(weights / weights.sum() * iterations).astype(int)
    scale = iterations / nominal.sum()
    nominal = np.ceil(scale * nominal).astype(int)[::-1]
    nominal[0] -= nominal.sum() - iterations
    per_level = nominal[::-1]
    schedule = []
    for index, count in enumerate(per_level):
        schedule.extend([max(levels - index - 1, 0)] * int(count))
    return schedule[:iterations]


def infer_placements(
        model, destination_points: np.ndarray, object_points: np.ndarray,
        candidate_count=8, refinement_steps=50, seed=13,
        device="cpu", orientation_policy="free") -> tuple[list[PlacementCandidate], float]:
    """Run the AnyPlace low-level network on destination/object point clouds."""
    started = time.perf_counter()
    destination = downsample_points(destination_points, 400)
    child = downsample_points(object_points, 200)
    destination_mean = destination.mean(axis=0)
    destination_centered = destination - destination_mean
    child_centered = child - child.mean(axis=0)

    rng = np.random.default_rng(seed)
    candidate_count = max(1, int(candidate_count))
    sample_indices = rng.choice(
        len(destination), size=candidate_count,
        replace=len(destination) < candidate_count)
    child_means = destination[sample_indices].copy()
    policy = str(orientation_policy).strip().lower()
    if policy not in ("free", "upright"):
        raise ValueError(f"unsupported orientation policy: {orientation_policy}")
    if policy == "upright":
        # The cached object cloud is expressed in base_link. A yaw-only
        # rotation preserves its current gravity-relative upright axis while
        # still allowing AnyPlace to choose its heading on the destination.
        yaw = rng.uniform(-np.pi, np.pi, size=candidate_count)
        rotations = np.zeros((candidate_count, 3, 3), dtype=np.float32)
        rotations[:, 0, 0] = np.cos(yaw)
        rotations[:, 0, 1] = -np.sin(yaw)
        rotations[:, 1, 0] = np.sin(yaw)
        rotations[:, 1, 1] = np.cos(yaw)
        rotations[:, 2, 2] = 1.0
    else:
        rotations = random_rotation_matrices(candidate_count, rng)

    destination_tensor = torch.from_numpy(destination_centered).to(device)
    destination_tensor = destination_tensor.unsqueeze(0).repeat(candidate_count, 1, 1)
    destination_means = torch.from_numpy(destination_mean).to(device)
    destination_means = destination_means.unsqueeze(0).repeat(candidate_count, 1)
    original_child_tensor = torch.from_numpy(child_centered).to(device)
    original_child_tensor = original_child_tensor.unsqueeze(0).repeat(
        candidate_count, 1, 1)
    centered_child = original_child_tensor
    rotation_tensor = torch.from_numpy(rotations).to(device)
    centered_child = torch.bmm(rotation_tensor, centered_child.transpose(1, 2)).transpose(1, 2)
    child_mean_tensor = torch.from_numpy(child_means).to(device)

    schedule = diffusion_timestep_schedule(int(refinement_steps))
    with torch.inference_mode():
        for timestep_value in schedule:
            timestep = torch.full(
                (candidate_count, 1), float(timestep_value),
                dtype=destination_tensor.dtype, device=device)
            timestep_embedding = model.pos_emb(timestep)
            model_input = {
                "parent_start_pcd": destination_tensor,
                "child_start_pcd": centered_child,
                "parent_start_pcd_mean": destination_means,
                "child_start_pcd_mean": child_mean_tensor,
                "timestep_emb": timestep_embedding,
            }
            rotation_delta = model(model_input)["rot_mat"][:, 0]
            rotation_tensor = torch.bmm(rotation_delta, rotation_tensor)
            if policy == "upright":
                # Project the learned update onto SO(2): roll and pitch are
                # hard constrained, yaw remains free. Translation is then
                # refined by the network against this upright geometry.
                yaw = torch.atan2(rotation_tensor[:, 1, 0],
                                  rotation_tensor[:, 0, 0])
                cosine, sine = torch.cos(yaw), torch.sin(yaw)
                projected = torch.zeros_like(rotation_tensor)
                projected[:, 0, 0] = cosine
                projected[:, 0, 1] = -sine
                projected[:, 1, 0] = sine
                projected[:, 1, 1] = cosine
                projected[:, 2, 2] = 1.0
                rotation_tensor = projected
                centered_child = torch.bmm(
                    rotation_tensor,
                    original_child_tensor.transpose(1, 2)).transpose(1, 2)
            else:
                centered_child = torch.bmm(
                    rotation_delta,
                    centered_child.transpose(1, 2)).transpose(1, 2)
            model_input["child_start_pcd"] = centered_child
            translation_delta = model(model_input)["trans"][:, 0]
            child_mean_tensor = child_mean_tensor + translation_delta

    # A geometry-only display ranking: candidates close to the destination
    # surface are shown first. It does not alter the network output poses.
    destination_cpu = torch.from_numpy(destination).float()
    transformed = torch.bmm(
        rotation_tensor.cpu(), torch.from_numpy(child_centered).float().unsqueeze(0)
        .repeat(candidate_count, 1, 1).transpose(1, 2)).transpose(1, 2)
    transformed += child_mean_tensor.cpu()[:, None, :]
    distances = torch.cdist(transformed, destination_cpu.unsqueeze(0).expand(
        candidate_count, -1, -1)).min(dim=-1).values
    contact_count = max(1, int(0.1 * distances.shape[1]))
    scores = -torch.topk(
        distances, contact_count, dim=1, largest=False).values.mean(dim=1)

    output = [
        PlacementCandidate(
            rotation_tensor[index].cpu().numpy(),
            child_mean_tensor[index].cpu().numpy(),
            float(scores[index]))
        for index in range(candidate_count)]
    output.sort(key=lambda item: item.score, reverse=True)
    return output, time.perf_counter() - started
