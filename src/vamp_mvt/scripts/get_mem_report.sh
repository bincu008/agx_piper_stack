#!/bin/bash

ROBOT="fetch"
# ROBOT="panda"
# ROBOT="fetch"
METHOD="mvt"
# METHOD="capt"
FILTER="centervox" #scdf
VOXEL_FILTER_R="0.03"

PROBLEM_IDX="1"
PROBLEMS=(
    # "table_pick"
    # "table_under_pick"
    "box"
    # "bookshelf_small"
    # "bookshelf_tall"
    # "bookshelf_thin"
    # "cage"
)

for PROB in "${PROBLEMS[@]}"; do
    python scripts/evaluate_mbm.py \
        --robot ${ROBOT} \
        --planner rrtc \
        --problem "${PROB}" \
        --problem_index "${PROBLEM_IDX}" \
        --pointcloud true \
        --pc_repr "${METHOD}" \
        --filter_type "${FILTER}" \
        --voxel_filter_size "${VOXEL_FILTER_R}"

    if ls scripts/log/${ROBOT}_${METHOD}_report* 1> /dev/null 2>&1; then
        echo "Renaming reports for ${PROB}..."
        for file in scripts/log/${ROBOT}_${METHOD}_report*; do
            filename=$(basename "$file")
            mv "$file" "scripts/log/${PROB}_${PROBLEM_IDX}_${filename}"
        done
    else
        echo "Warning: No report files found for ${PROB}."
    fi
    
    echo "-----------------------------------"
done