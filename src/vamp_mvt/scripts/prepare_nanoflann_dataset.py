import pickle
import time
from tabulate import tabulate
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from typing import Union, List
import os
import numpy as np

from fire import Fire
import vamp
from vamp import pointcloud as vpc

def save_filtered_pc_to_txt(filtered_pc, name, robot, i):
    """
    Save filtered point cloud to plain text format
    
    Args:
        filtered_pc: Point cloud data (numpy array or similar)
        name: Dataset/experiment name
        robot: Robot identifier
        i: Sample/iteration index
    """
    # Create directory structure
    save_dir = f"../nanoflann_dataset/{name}_{robot}/filtered_pc"
    os.makedirs(save_dir, exist_ok=True)
    
    # Define file path
    file_path = os.path.join(save_dir, f"{i}.txt")
    
    # Convert to numpy array if needed
    if not isinstance(filtered_pc, np.ndarray):
        points = np.array(filtered_pc)
    else:
        points = filtered_pc
    
    # Ensure points are in correct format (N x 3)
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    
    # Save as plain text
    np.savetxt(file_path, points, fmt='%.6f')
    
    print(f"Saved filtered point cloud to: {file_path}")
    return file_path

def main(
    robot: str = "panda",                  # Robot to plan for
    planner: str = "rrtc",                 # Planner name to use
    dataset: str = "problems.pkl",         # Pickled dataset to use
    problem: Union[str, List[str]] = [],   # Problem name or list of problems to  gen dataset
    problem_index: int = None,             # Problem index to gen dataset
    sampler: str = "halton",               # Sampler to use.
    skip_rng_iterations: int = 0,          # Skip a number of RNG iterations
    pc_repr: str = None,                   # Pointcloud representation, required if pointcloud=True
    samples_per_object: int = 10000,       # If pointcloud, samples per object to use
    filter_type: str = "scdf",             # Filter type for pointcloud filtering
    filter_radius: float = 0.02,           # Filter radius for pointcloud filtering, required if filter_type="scdf"
    voxel_filter_size: float = 0.0303,     # Voxel filter size for pointcloud filtering, required if filter_type="centervox"
    filter_cull: bool = True,              # Cull pointcloud around robot by maximum distance, unused if filter_type="centervox"
    prepare_pc: bool = False,              # If `true`, save filtered point cloud into text files
    **kwargs,
    ):

    if robot not in vamp.ROBOT_JOINTS:
        raise RuntimeError(f"Robot {robot} does not exist in VAMP!")

    if not pc_repr:
        raise ValueError("pc_repr (pointcloud representation) is required when pointcloud=True. Available repr: capt, mvt")
    else:
        if pc_repr not in ["capt", "mvt"]:
            raise ValueError("pc_repr must be one of: 'capt', 'mvt'")

    if filter_type not in ["scdf", "centervox"]:
        raise ValueError("filter_type must be one of: 'scdf', 'centervox'\n\t" \
                            "scdf: Space-filling Curve Distance Filter\n\t" \
                            "centervox: Center-Selective Voxel Filter")
    else:
        if filter_type == "scdf" and not filter_radius:
            raise ValueError("filter_radius is required when filter_type=scdf")
        
        if filter_type == "centervox" and not voxel_filter_size:
            raise ValueError("voxel_filter_size is required when filter_type=centervox")

    problems_dir = Path(__file__).parent.parent / 'resources' / robot / 'problems'
    with open(problems_dir.parent / dataset, 'rb') as f:
        problems = pickle.load(f)

    problem_names = list(problems['problems'].keys())
    if isinstance(problem, str):
        problem = [problem]

    if not problem:
        problem = problem_names
    else:
        for problem_name in problem:
            if problem_name not in problem_names:
                raise RuntimeError(
                    f"Problem `{problem_name}` not available! Available problems: {problem_names}"
                    )

    if problem_index is None:
        raise ValueError("A specified problem_index is required")

    (vamp_module, planner_func, plan_settings,
     simp_settings) = vamp.configure_robot_and_planner_with_kwargs(robot, planner, **kwargs)

    sampler = getattr(vamp_module, sampler)()

    total_problems = 0
    valid_problems = 0
    failed_problems = 0

    tick = time.perf_counter()
    results = []
    for name, pset in problems['problems'].items():
        if name not in problem:
            continue

        data = pset[problem_index]
            
        total_problems += 1

        if data['valid']: # if the problem is marked invalid, no point cloud or cc queries will be saved to file
            valid_problems += 1

            (env, original_pc, filtered_pc, filter_time, build_time) = vpc.problem_dict_to_pointcloud(
                robot,
                pc_repr,
                data,
                samples_per_object,
                filter_type,
                filter_radius,
                voxel_filter_size,
                filter_cull
                )

            if prepare_pc:
                save_filtered_pc_to_txt(filtered_pc, name, robot, problem_index)

            else:
                sampler.reset()
                sampler.skip(skip_rng_iterations)
                
                result = planner_func(data['start'], data['goals'], env, plan_settings, sampler)
                
if __name__ == "__main__":
    Fire(main)
