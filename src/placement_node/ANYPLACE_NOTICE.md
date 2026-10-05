# Upstream notice

The network architecture and diffusion inference logic in this package are an
ARM64/CPU adaptation of the AnyPlace low-level placement network:

- Repository: https://github.com/ac-rad/anyplace
- Reviewed revision: `3049f78ad226ba0d9e54c63e2ca7ad7bbcfaa45e`
- Authors: Yuchi Zhao et al.
- License: MIT

This package intentionally excludes Molmo, SAM2, AnyGrasp, IsaacLab, the
energy-model baseline, training code, and the original simulation/evaluation
wrapper. `torch-cluster` farthest-point sampling was replaced with native
PyTorch and all computational tensors remain in the incoming camera frame.
