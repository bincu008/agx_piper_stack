"""CPU-portable AnyPlace diffusion transformer.

The module hierarchy intentionally matches the released AnyPlace checkpoint.
Only the low-level ``NSMTransformerSingleTransformationRegression`` network is
included.  CUDA/PyG sampling has been replaced by native PyTorch.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
import torch.nn.functional as functional


class SinusoidalPosEmb:
    """Parameter-free timestep embedding used by the released checkpoint."""

    def __init__(self, dim: int, max_pos: int | None = None):
        self.dim = dim
        self.max_pos = max_pos

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        if self.max_pos is not None:
            value = torch.clamp(value, 0, self.max_pos)
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=value.device, dtype=value.dtype) * -scale)
        embedded = value[:, None] * frequencies[None, :]
        return torch.cat((embedded.sin(), embedded.cos()), dim=-1)


def native_farthest_point_indices(
        points: torch.Tensor, sample_count: int) -> torch.Tensor:
    """Return deterministic greedy FPS indices for a ``B x N x 3`` tensor."""
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape B x N x 3")
    batch_size, point_count, _ = points.shape
    if point_count == 0:
        raise ValueError("cannot sample an empty point cloud")
    count = min(int(sample_count), point_count)
    if count == point_count:
        return torch.arange(point_count, device=points.device).expand(batch_size, -1)

    indices = torch.empty(
        (batch_size, count), dtype=torch.long, device=points.device)
    minimum_distance = torch.full(
        (batch_size, point_count), float("inf"),
        dtype=points.dtype, device=points.device)
    # Starting far from the centroid is deterministic and gives better coverage
    # than always using input point zero.
    centroid = points.mean(dim=1, keepdim=True)
    farthest = ((points - centroid) ** 2).sum(dim=-1).argmax(dim=1)
    batch = torch.arange(batch_size, device=points.device)
    for index in range(count):
        indices[:, index] = farthest
        selected = points[batch, farthest].unsqueeze(1)
        distance = ((points - selected) ** 2).sum(dim=-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        farthest = minimum_distance.argmax(dim=1)
    return indices


class FPSDownSample:
    """Native replacement for the original ``torch_cluster.fps`` wrapper."""

    def __init__(self, num_group: int):
        self.num_group = int(num_group)

    def forward(
            self, points: torch.Tensor,
            return_idx: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        indices = native_farthest_point_indices(points, self.num_group)
        sampled = torch.gather(
            points, 1, indices[:, :, None].expand(-1, -1, points.shape[-1]))
        return (sampled, indices) if return_idx else sampled


def _clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


def _attention(query, key, value, mask=None):
    dimension = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(dimension)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention = functional.softmax(scores, dim=-1)
    return torch.matmul(attention, value), attention


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, value):
        mean = value.mean(-1, keepdim=True)
        std = value.std(-1, keepdim=True)
        return self.a_2 * (value - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.norm = LayerNorm(size)

    def forward(self, value, sublayer):
        return value + sublayer(self.norm(value))


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count: int, model_dimension: int):
        super().__init__()
        if model_dimension % head_count:
            raise ValueError("model dimension must be divisible by head count")
        self.d_k = model_dimension // head_count
        self.h = head_count
        self.linears = _clones(nn.Linear(model_dimension, model_dimension), 4)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        query, key, value = [
            layer(item).view(batch_size, -1, self.h, self.d_k).transpose(1, 2).contiguous()
            for layer, item in zip(self.linears, (query, key, value))]
        output, self.attn = _attention(query, key, value, mask)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.d_k)
        return self.linears[-1](output)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, model_dimension: int, hidden_dimension: int):
        super().__init__()
        self.w_1 = nn.Linear(model_dimension, hidden_dimension)
        self.norm = nn.Sequential()
        self.w_2 = nn.Linear(hidden_dimension, model_dimension)

    def forward(self, value):
        hidden = self.norm(functional.relu(self.w_1(value)).transpose(2, 1).contiguous())
        return self.w_2(hidden.transpose(2, 1).contiguous())


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = _clones(SublayerConnection(size), 2)
        self.size = size

    def forward(self, value, mask):
        value = self.sublayer[0](
            value, lambda normalized: self.self_attn(
                normalized, normalized, normalized, mask))
        return self.sublayer[1](value, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = _clones(SublayerConnection(size), 3)

    def forward(self, value, memory, src_mask, target_mask):
        value = self.sublayer[0](
            value, lambda normalized: self.self_attn(
                normalized, normalized, normalized, target_mask))
        value = self.sublayer[1](
            value, lambda normalized: self.src_attn(
                normalized, memory, memory, src_mask))
        return self.sublayer[2](value, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, count):
        super().__init__()
        self.layers = _clones(layer, count)
        self.norm = LayerNorm(layer.size)

    def forward(self, value, mask):
        for layer in self.layers:
            value = layer(value, mask)
        return self.norm(value)


class Decoder(nn.Module):
    def __init__(self, layer, count):
        super().__init__()
        self.layers = _clones(layer, count)
        self.norm = LayerNorm(layer.size)

    def forward(self, value, memory, src_mask, target_mask):
        for layer in self.layers:
            value = layer(value, memory, src_mask, target_mask)
        return self.norm(value)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, target_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, src_mask=None, target_mask=None):
        memory = self.encoder(self.src_embed(source), src_mask)
        decoded = self.decoder(self.tgt_embed(target), memory, src_mask, target_mask)
        return self.generator(decoded)


class Transformer(nn.Module):
    """Transformer with names matching AnyPlace ``nsm_transformer.Transformer``."""

    def __init__(self, hidden_dim=256, n_heads=1, n_blocks=4):
        super().__init__()
        feature_dim = hidden_dim + 2
        attention = MultiHeadedAttention(n_heads, feature_dim)
        feed_forward = PositionwiseFeedForward(feature_dim, hidden_dim)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(
                feature_dim, copy.deepcopy(attention), copy.deepcopy(feed_forward)), n_blocks),
            Decoder(DecoderLayer(
                feature_dim, copy.deepcopy(attention), copy.deepcopy(attention),
                copy.deepcopy(feed_forward)), n_blocks),
            nn.Sequential(), nn.Sequential(), nn.Sequential())

    def forward(self, source, target):
        source = source.transpose(2, 1).contiguous()
        target = target.transpose(2, 1).contiguous()
        return self.model(target, source).transpose(2, 1).contiguous()


class NSMTransformerSingleTransformationRegression(nn.Module):
    """The released AnyPlace low-level diffusion correction network."""

    def __init__(
            self, feat_dim=3, hidden_dim=256, n_heads=1, n_blocks=4,
            n_pts=1024, pn_pts=None, cn_pts=None, n_queries=1,
            pooling="max", use_timestep_emb=True, max_timestep=5,
            timestep_pool_method="meanpool", **_unused):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = hidden_dim
        self.n_pts = n_pts
        self.pn_pts = pn_pts
        self.cn_pts = cn_pts
        self.pcd_emb = nn.Linear(3, hidden_dim // 2)
        self.des_emb = nn.Linear(feat_dim, hidden_dim // 2)
        self.n_queries = n_queries
        self.transformer = Transformer(hidden_dim, n_heads, n_blocks)
        pooled_dim = hidden_dim + 2
        self.out_trans = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.out_vec1 = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.out_vec2 = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.pooling = pooling
        self.pos_emb_max_timestep = max_timestep
        self.pos_emb = SinusoidalPosEmb(hidden_dim + 2, max_timestep)
        self.timestep_emb_proj = nn.Linear(hidden_dim + 2, hidden_dim + 2)
        self.use_timestep_emb = use_timestep_emb
        self.timestep_pool_method = timestep_pool_method
        parent_count = n_pts if pn_pts is None else pn_pts
        child_count = n_pts if cn_pts is None else cn_pts
        self.fps_ds_p = FPSDownSample(parent_count)
        self.fps_ds_c = FPSDownSample(child_count)
        self.register_buffer(
            "onehot_a", torch.tensor([[[1.0, 0.0]]]), persistent=False)
        self.register_buffer(
            "onehot_b", torch.tensor([[[0.0, 1.0]]]), persistent=False)

    def _process_input(self, model_input):
        parent_centered = model_input["parent_start_pcd"]
        child_centered = model_input["child_start_pcd"]
        offset = (
            model_input["parent_start_pcd_mean"]
            - model_input["child_start_pcd_mean"])
        batch_size, parent_count, _ = parent_centered.shape
        child_count = child_centered.shape[1]
        diameter = (
            parent_centered.max(1).values.max(-1).values
            - parent_centered.min(1).values.min(-1).values)
        scaling = 1.0 / torch.clamp(diameter / 2.0, min=1e-6)
        parent_scaled = parent_centered * scaling[:, None, None]
        child_scaled = child_centered * scaling[:, None, None]
        child_relative = child_scaled - offset[:, None, :] * scaling[:, None, None]
        parent_features = torch.cat((
            self.pcd_emb(parent_scaled), self.des_emb(parent_scaled),
            self.onehot_a.expand(batch_size, parent_count, -1)), dim=-1)
        child_features = torch.cat((
            self.pcd_emb(child_relative), self.des_emb(child_scaled),
            self.onehot_b.expand(batch_size, child_count, -1)), dim=-1)
        return parent_scaled, child_relative, parent_features, child_features

    def forward(self, model_input):
        parent, child, parent_features, child_features = self._process_input(model_input)
        parent, parent_indices = self.fps_ds_p.forward(parent, return_idx=True)
        child, child_indices = self.fps_ds_c.forward(child, return_idx=True)
        parent_features = torch.gather(
            parent_features, 1,
            parent_indices[:, :, None].expand(-1, -1, parent_features.shape[-1]))
        child_features = torch.gather(
            child_features, 1,
            child_indices[:, :, None].expand(-1, -1, child_features.shape[-1]))
        batch_size = parent.shape[0]
        if "timestep_emb" in model_input and self.use_timestep_emb:
            child_features = torch.cat((
                child_features, model_input["timestep_emb"].reshape(
                    batch_size, 1, -1)), dim=1)
        per_point = self.transformer(
            child_features.transpose(2, 1), parent_features.transpose(2, 1)).transpose(2, 1)
        if self.pooling == "mean":
            pooled = per_point.mean(dim=1, keepdim=True)
        else:
            pooled = per_point.max(dim=1, keepdim=True).values
        if "timestep_emb" in model_input and self.use_timestep_emb:
            timestep = self.timestep_emb_proj(
                model_input["timestep_emb"]).reshape(batch_size, 1, -1)
            if self.timestep_pool_method == "concat":
                pooled = torch.cat((pooled, timestep), dim=-1)
            else:
                pooled = torch.stack((pooled, timestep), dim=1).mean(dim=1)

        translation = self.out_trans(pooled)
        vector_one = functional.normalize(self.out_vec1(pooled), dim=-1)
        raw_vector_two = self.out_vec2(pooled)
        vector_two = raw_vector_two - (
            vector_one * raw_vector_two).sum(-1, keepdim=True) * vector_one
        vector_two = functional.normalize(vector_two, dim=-1)
        vector_three = torch.cross(vector_one, vector_two, dim=-1)
        rotation = torch.stack((vector_one, vector_two, vector_three), dim=-1)
        return {"trans": translation, "rot_mat": rotation}


def build_anyplace_model() -> NSMTransformerSingleTransformationRegression:
    """Construct the architecture recorded in the released multitask checkpoint."""
    return NSMTransformerSingleTransformationRegression(
        feat_dim=3, hidden_dim=256, n_heads=1, n_blocks=4,
        n_pts=1024, pn_pts=None, cn_pts=None, n_queries=1,
        pooling="max", use_timestep_emb=True, max_timestep=5,
        timestep_pool_method="meanpool")


def load_anyplace_checkpoint(path: str, device="cpu"):
    """Load only the low-level pose network from an official checkpoint."""
    model = build_anyplace_model().to(device)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("refine_pose_model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
