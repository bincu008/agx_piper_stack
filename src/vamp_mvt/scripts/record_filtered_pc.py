import pickle
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Union, List
from fire import Fire

import vamp
from vamp import pointcloud as vpc

def export_filtered_pointclouds(
    robot: str = "ur5",                  # Robot to plan for
    planner: str = "rrtc",                 # Planner name to use
    dataset: str = "problems.pkl",         # Pickled dataset to use
    problem: Union[str, List[str]] = ["table_pick", "table_under_pick", "box", "bookshelf_small", "bookshelf_tall", "bookshelf_thin", "cage"],   # Problem name or list of problems to evaluate
    problem_index: Union[int, List[int]] = None,# Problem index or list of indices to evaluate
    trials: int = 1,                       # Number of trials to evaluate each instance
    sampler: str = "halton",               # Sampler to use.
    skip_rng_iterations: int = 0,          # Skip a number of RNG iterations
    print_failures: bool = False,          # Print out failures and invalid problems
    pointcloud: bool = True,              # Use pointcloud rather than primitive geometry
    pc_repr: str = "mvt",                   # Pointcloud representation, required if pointcloud=True
    samples_per_object: int = 10000,       # If pointcloud, samples per object to use
    filter_type: str = "scdf",             # Filter type for pointcloud filtering
    filter_radius: float = 0.02,           # Filter radius for pointcloud filtering, required if filter_type="scdf"
    voxel_filter_size: float = 0.031,     # Voxel filter size for pointcloud filtering, required if filter_type="centervox"
    filter_cull: bool = True,              # Cull pointcloud around robot by maximum distance, unused if filter_type="centervox"
    output_base_dir: str = "../nanoflann_dataset/pointcloud",
    **kwargs
):
    """
    Save filtered point cloud into file: {output_base_dir}/{scene_name}_{robot}/{problem_index}.txt
    """
    if robot not in vamp.robots:
        raise RuntimeError(f"Robot {robot} does not exist in VAMP!")

    base_path = Path(__file__).parent
    problems_dir = base_path.parent / 'resources' / robot / 'problems'
    dataset_path = problems_dir.parent / dataset
    
    with open(dataset_path, 'rb') as f:
        problems = pickle.load(f)

    (vamp_module, planner_func, plan_settings,
     simp_settings) = vamp.configure_robot_and_planner_with_kwargs(robot, planner, **kwargs)

    r_min, r_max = vamp_module.min_max_radii()

    output_root = Path(output_base_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Starting Pointcloud Export (SCDF, Radius: {filter_radius})...")

    with open(dataset_path, 'rb') as f:
        problems = pickle.load(f)

    target_problems = [problem] if isinstance(problem, str) else problem
    
    available_scenes = problems['problems'].keys()
    
    scenes_to_process = [p for p in target_problems if p in available_scenes]
    
    if not scenes_to_process:
        print(f"Error: {target_problems} not exiists in problem dataset")
        print(f"Problem dataset: {list(available_scenes)}")
        return

    for scene_name in scenes_to_process:
        pset = problems['problems'][scene_name]
        print(f"Processing Scene: {scene_name}")
        
        scene_dir = output_root / f"{scene_name}_{robot}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        for i, data in tqdm(enumerate(pset), total=len(pset), desc=f"Exporting {scene_name}"):
            if not data['valid']:
                continue

            # Filtering (Use scdf and specified radius)
            # vpc.problem_dict_to_pointcloud 會回傳: (env, original_pc, filtered_pc, filter_time, build_time)
            _, _, filtered_pc, _, _ = vpc.problem_dict_to_pointcloud(
                robot=robot,
                r_min=r_min,
                r_max=r_max,
                pointcloud_repr="mvt",  # Don't care environment representation
                problem=data,
                samples_per_object=samples_per_object,
                filter_type="scdf",
                filter_radius=filter_radius,
                voxel_filter_size=0.0, # Don't care centervox parameter
                filter_cull=True
            )

            output_file = scene_dir / f"{i}.txt"
            
            np.savetxt(output_file, filtered_pc, fmt='%.6f', delimiter=' ')

    print(f"\nSuccessfully exported all pointclouds to: {output_root}")

if __name__ == "__main__":
    Fire(export_filtered_pointclouds)