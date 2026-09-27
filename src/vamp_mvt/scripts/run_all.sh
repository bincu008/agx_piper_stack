ROBOT="fetch"
# METHOD="capt"
METHOD="mvt"
FILTER="centervox"
# FILTER="scdf"
VOXEL_FILTER_R="0.031"

PROBLEMS=(
    "table_pick"
    "table_under_pick"
    "box"
    "bookshelf_small"
    "bookshelf_tall"
    "bookshelf_thin"
    "cage"
)

for PROB in "${PROBLEMS[@]}"; do
    python scripts/evaluate_mbm.py \
        --robot ${ROBOT} \
        --planner rrtc \
        --problem "${PROB}" \
        --pointcloud true \
        --pc_repr "${METHOD}" \
        --filter_type "${FILTER}" \
        --voxel_filter_size "${VOXEL_FILTER_R}"
done