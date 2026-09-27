#!/usr/bin/env bash
set -euo pipefail
model_dir="${1:-$(cd "$(dirname "$0")/../models/hggd_mcu" && pwd)}"
cd "$model_dir"
printf '%s\n' \
 '08b9dd3c67e0360084d5f66c4bc86a21bd78649fc96903a72d8e6fd13c34f881  resnet.onnx' \
 '57049ea3243cc9786ae67264a7b6caa131233438c54e3e6966dc184cc2ea61c0  anchornet.onnx' \
 'dd1217444f0ed1cdb369b701492669c8d2ab561f40d30387fd6719aba0a92ff5  pointnet.onnx' \
 '37804865c5e5ffcd0fd1ede046bcd8189f2c06bb408851cca953012bc2817dfb  localnet.onnx' | sha256sum --check
