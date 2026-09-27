#!/bin/bash

# Before run this script with PREPARE_PC != "true", make sure two things:
# 1. there's `std::cout << centers[0] << " " << centers[1] << " " << centers[2] << " " << radii << std::endl;` 
# in collides_simd() in capt.hh
# 2. there's no other output log during execution of planner

ROBOT="ur5"
PROBLEM_NAMES=("box") # "bookshelf_thin" "bookshelf_small" "table_pick" "table_three_pick" "table_under_pick"
PREPARE_PC="false"

if [[ "$PREPARE_PC" == "true" ]]; then # prepare filtered point cloud dataset (filtered by SCDF with 0.02 radius)
    for PROBLEM_NAME in "${PROBLEM_NAMES[@]}"; do
        echo "Processing problem: $PROBLEM_NAME"
        
        # Create output directory   
        mkdir -p "../nanoflann_dataset/${PROBLEM_NAME}_${ROBOT}/filtered_pc"

        for i in {0..99}; do  # modify here for different problem indices
            python scripts/prepare_nanoflann_dataset.py \
                --pc_repr capt \
                --robot $ROBOT \
                --problem $PROBLEM_NAME \
                --problem_index $i \
                --prepare_pc true
        done
        echo "Removing empty files for problem $PROBLEM_NAME"
        find "../nanoflann_dataset/${PROBLEM_NAME}_${ROBOT}/filtered_pc/" -name "*.txt" -size 0 -delete
    done
else # prepare cc queries dataset
    for PROBLEM_NAME in "${PROBLEM_NAMES[@]}"; do
        echo "Processing problem: $PROBLEM_NAME"
        
        # Create output directory
        mkdir -p "../nanoflann_dataset/${PROBLEM_NAME}_${ROBOT}/cc_query"
        
        for i in {0..99}; do # modify here for different problem indices
            python scripts/prepare_nanoflann_dataset.py \
                --pc_repr capt \
                --robot $ROBOT \
                --problem $PROBLEM_NAME \
                --problem_index $i \
                > "../nanoflann_dataset/${PROBLEM_NAME}_${ROBOT}/cc_query/${i}.txt"
        done
        echo "Removing empty files for problem $PROBLEM_NAME"
        find "../nanoflann_dataset/${PROBLEM_NAME}_${ROBOT}/cc_query/" -name "*.txt" -size 0 -delete
    done
fi
echo "Dataset generation complete!"