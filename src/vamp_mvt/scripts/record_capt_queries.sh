#!/bin/bash

ROBOT="fetch"
METHOD="capt"
FILTER="scdf"
VOXEL_FILTER_R="0.031"

PROBLEMS=(
    # "table_pick"
    # "table_under_pick"
    # "box"
    # "bookshelf_small"
    # "bookshelf_tall"
    # "bookshelf_thin"
    "cage"
)

for PROB in "${PROBLEMS[@]}"; do
    # i 從 0 到 99
    for i in {0..99}; do
        
        SAVE_DIR="../nanoflann_dataset/${PROB}_${ROBOT}_capt_q/${i}"
        
        OUTPUT=$(python scripts/evaluate_mbm.py \
            --robot ${ROBOT} \
            --planner rrtc \
            --problem "${PROB}" \
            --pointcloud true \
            --pc_repr "${METHOD}" \
            --filter_type "${FILTER}" \
            --voxel_filter_size "${VOXEL_FILTER_R}" \
            --problem_index "${i}")

        # Skip invalid problems
        if [[ $OUTPUT == *"invalid problem"* ]]; then
            echo "Skipping ${PROB} index ${i}: Invalid problem"
            continue
        fi

        mkdir -p "${SAVE_DIR}"

        # Search "Collide, c: " and extract content after "c: "
        COLLIDE_DATA=$(echo "$OUTPUT" | grep "^Collide," | sed 's/^Collide,//')
        if [ ! -z "$COLLIDE_DATA" ]; then
            # Overwrite the file
            echo "$COLLIDE_DATA" > "${SAVE_DIR}/collide.txt"
        fi

        # Search "Safe, c: " and extract content after "c: "
        SAFE_DATA=$(echo "$OUTPUT" | grep "^Safe," | sed 's/^Safe,//')
        if [ ! -z "$SAFE_DATA" ]; then
            echo "$SAFE_DATA" > "${SAVE_DIR}/safe.txt"
        fi

    done
done

echo "Done"