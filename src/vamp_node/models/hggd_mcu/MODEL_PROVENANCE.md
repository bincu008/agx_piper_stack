# HGGD-MCU ONNX models

Source: https://github.com/ThomasVroom/HGGD-MCU.git, commit `633023950508ecdf6779b54d17c2071f475a5dfd` (MIT).

The CPU exports use RGB-D input `[1,4,320,160]`; the unusual layout is
`[N,C,width,height]`. ResNet outputs five feature levels consumed by AnchorNet.
AnchorNet emits `loc_map [1,1,320,160]`, six angle-anchor maps and
`perpoint_features [1,32,40,20]`. PointNet consumes `[centers,35,512]` and
LocalNet consumes PointNet features plus `[centers,3]` grasp info.

SHA-256: resnet `08b9dd3c67e0360084d5f66c4bc86a21bd78649fc96903a72d8e6fd13c34f881`;
anchornet `57049ea3243cc9786ae67264a7b6caa131233438c54e3e6966dc184cc2ea61c0`;
pointnet `dd1217444f0ed1cdb369b701492669c8d2ab561f40d30387fd6719aba0a92ff5`;
localnet `37804865c5e5ffcd0fd1ede046bcd8189f2c06bb408851cca953012bc2817dfb`.
