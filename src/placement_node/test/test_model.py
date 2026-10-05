import torch

from placement_node.model import build_anyplace_model, native_farthest_point_indices


def test_released_architecture_parameter_count():
    model = build_anyplace_model()
    assert sum(parameter.numel() for parameter in model.parameters()) == 4_548_759


def test_native_fps_is_cpu_and_deterministic():
    points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]])
    first = native_farthest_point_indices(points, 3)
    second = native_farthest_point_indices(points, 3)
    assert first.device.type == "cpu"
    assert torch.equal(first, second)
    assert len(torch.unique(first)) == 3


def test_model_cpu_forward_shapes():
    model = build_anyplace_model().eval()
    model.fps_ds_p.num_group = 16
    model.fps_ds_c.num_group = 12
    model_input = {
        "parent_start_pcd": torch.randn(1, 16, 3),
        "child_start_pcd": torch.randn(1, 12, 3),
        "parent_start_pcd_mean": torch.zeros(1, 3),
        "child_start_pcd_mean": torch.zeros(1, 3),
        "timestep_emb": model.pos_emb(torch.ones(1, 1)),
    }
    with torch.inference_mode():
        output = model(model_input)
    assert output["rot_mat"].shape == (1, 1, 3, 3)
    assert output["trans"].shape == (1, 1, 3)
    assert torch.isfinite(output["rot_mat"]).all()
    assert torch.isfinite(output["trans"]).all()
