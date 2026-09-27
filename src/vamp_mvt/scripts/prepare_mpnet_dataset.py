import pickle
import numpy as np
from tabulate import tabulate
import os
import yaml
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Union, List

from fire import Fire
import vamp
from vamp import pointcloud as vpc

def prepare_mpnet_dataset(points, robot_name, scene_name, variation_number):
    """
    Prepare and store point cloud data for MPnet training dataset.
    
    Args:
        points (list): List of points in point cloud, each point should be [x, y, z]
        robot_name (str): Name of the robot
        scene_name (str): Name of the scene
        variation_number (int): Variation number for the dataset
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert points to numpy array and ensure shape is (N, 3)
        points_array = np.array(points)
        
        # Validate input shape
        if len(points_array.shape) != 2 or points_array.shape[1] != 3:
            print(f"Error: Points must have shape (N, 3), got {points_array.shape}")
            return False
        
        # Create directory path
        scene_name = scene_name.strip('[]').strip("'")
        
        # Create directory paths for both obc and path
        base_dir = Path(f"mpnet_dataset/{scene_name}_{robot_name}")
        obc_dir = base_dir / "obc"
        path_dir = base_dir / "path"

        # Create directories
        obc_dir.mkdir(parents=True, exist_ok=True)
        path_dir.mkdir(parents=True, exist_ok=True)

        # Save point cloud data
        obc_filename = f"obc_0_{variation_number}.npy"
        obc_filepath = obc_dir / obc_filename
        np.save(obc_filepath, points_array)

        # Load and process reference path
        yaml_path = Path(f"additional_problems/{scene_name}_{robot_name}/path{variation_number + 1:04d}.yaml")
        
        if yaml_path.exists():
            with open(yaml_path, 'r') as file:
                yaml_data = yaml.safe_load(file)
            
            # Extract positions from joint trajectory
            if 'joint_trajectory' in yaml_data and 'points' in yaml_data['joint_trajectory']:
                trajectory_points = yaml_data['joint_trajectory']['points']
                positions_list = []
                
                for point in trajectory_points:
                    if 'positions' in point:
                        positions_list.append(point['positions'])
                
                if positions_list:
                    # Convert to numpy array with shape (1, sequence_length, DOF)
                    positions_array = np.array(positions_list)  # Shape: (sequence_length, DOF)
                    positions_array = np.expand_dims(positions_array, axis=0)  # Shape: (1, sequence_length, DOF)
                    
                    # Save path data
                    path_filename = f"path_0_{variation_number}.npy"
                    path_filepath = path_dir / path_filename
                    np.save(path_filepath, positions_array)
                    
                    # print(f"Successfully saved point cloud to: {obc_filepath}")
                    # print(f"Successfully saved reference path to: {path_filepath}")
                    # print(f"Point cloud shape: {points_array.shape}")
                    # print(f"Reference path shape: {positions_array.shape}")
                else:
                    print(f"Warning: No valid positions found in {yaml_path}")
            else:
                print(f"Warning: Invalid YAML structure in {yaml_path}")
        else:
            print(f"Warning: Reference path file not found: {yaml_path}")
            print("Continuing with point cloud data only...")
        
        return True
        
    except Exception as e:
        print(f"Error preparing MPnet dataset: {str(e)}")
        return False


def main(
    robot: str = "fetch",                  # Robot to plan for
    planner: str = "rrtc",                 # Planner name to use
    dataset: str = "problems.pkl",         # Pickled dataset to use
    problem: Union[str, List[str]] = [],   # Problem name or list of problems to evaluate
    sampler: str = "halton",               # Sampler to use.
    pc_repr: str = None,                   # Pointcloud representation, required if pointcloud=True
    samples_per_object: int = 10000,       # If pointcloud, samples per object to use
    filter_type: str = None,               # Filter type for pointcloud filtering
    filter_radius: float = 0.02,           # Filter radius for pointcloud filtering, required if filter_type="scdf"
    voxel_filter_size: float = 0.0303,     # Voxel filter size for pointcloud filtering, required if filter_type="centervox"
    filter_cull: bool = True,              # Cull pointcloud around robot by maximum distance, unused if filter_type="centervox"
    **kwargs,
    ):

    if robot not in vamp.ROBOT_JOINTS:
        raise RuntimeError(f"Robot {robot} does not exist in VAMP!")

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

    (vamp_module, planner_func, plan_settings,
     simp_settings) = vamp.configure_robot_and_planner_with_kwargs(robot, planner, **kwargs)

    sampler = getattr(vamp_module, sampler)()

    total_problems = 0
    valid_problems = 0
    results = []
    
    for name, pset in problems['problems'].items():
        if name not in problem:
            continue

        invalids = []
        print(f"Evaluating {robot} on {name}: ")
        for i, data in tqdm(enumerate(pset)):
            total_problems += 1

            # Only variations be classified as "valid" by vamp will be included in mpnet dataset
            if not data['valid']:
                invalids.append(i)
                continue 

            valid_problems += 1

            (env, original_pc, filtered_pc, filter_time, build_time) = vpc.problem_dict_to_pointcloud(
                robot,
                pc_repr,
                data,
                samples_per_object,
                filter_type,
                filter_radius,
                voxel_filter_size,
                filter_cull,
                )
            
            if pc_repr == "capt":
                pointcloud_results = {
                    'original_pointcloud_size': len(original_pc),
                    'filtered_pointcloud_size': len(filtered_pc),
                    'filter_time': pd.Timedelta(nanoseconds = filter_time),
                    'capt_build_time': pd.Timedelta(nanoseconds = build_time)
                    }
            else: 
                pointcloud_results = {
                    'original_pointcloud_size': len(original_pc),
                    'filtered_pointcloud_size': len(filtered_pc),
                    'filter_time': pd.Timedelta(nanoseconds = filter_time),
                    'mvt_build_time': pd.Timedelta(nanoseconds = build_time)
                    }

            success = prepare_mpnet_dataset(
                points=filtered_pc,
                robot_name=robot,
                scene_name=str(problem),
                variation_number=i
                # Note that i is zero-based
                # E.g. obc_0_1.npy corresponds to scene0002.yaml and path0002.yaml
            )
            
            if not success:
                print("Dataset preparation failed!")

            results.append(pointcloud_results)

    df = pd.DataFrame.from_dict(results)
    if pc_repr == "capt":
        df["capt_build_time"] = df["capt_build_time"].dt.microseconds / 1e3
    else:
        df["mvt_build_time"] = df["mvt_build_time"].dt.microseconds / 1e3
    
    df["filter_time"] = df["filter_time"].dt.microseconds / 1e3

    if pc_repr == "capt":
        pointcloud_stats = df[[
            "original_pointcloud_size",
            "filtered_pointcloud_size",
            "filter_time"
            ]].describe(percentiles = [0.25, 0.5, 0.75, 0.95])
        pointcloud_stats.drop(index = ["count"], inplace = True)
    else:
        pointcloud_stats = df[[
            "original_pointcloud_size",
            "filtered_pointcloud_size",
            "filter_time"
            ]].describe(percentiles = [0.25, 0.5, 0.75, 0.95])
        pointcloud_stats.drop(index = ["count"], inplace = True)

    if pc_repr == "capt":
        print(
            tabulate(
                pointcloud_stats,
                headers = [
                    '  Original PC Size',
                    '  Filtered PC Size',
                    '        Filter Time (ms)'
                    ],
                tablefmt = 'github'
                )
            )
    else:
        print(
            tabulate(
                pointcloud_stats,
                headers = [
                    '  Original PC Size',
                    '  Filtered PC Size',
                    '        Filter Time (ms)'
                    ],
                tablefmt = 'github'
                )
        )   
    
    print(f"Valid / Total # Problems: {valid_problems} / {total_problems}")

if __name__ == "__main__":
    Fire(main)
