import pickle
import time
from tabulate import tabulate
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from typing import Union, List

from fire import Fire
import vamp
from vamp import pointcloud as vpc
from vamp import mpnet_planner as vmp


def main(
    encoder: str = "../mpnet/fetch_bookshelf_thin_encoder.pkl",     # Path to encoder model (TODO: auto construct by robot and problem name)
    planner: str = "../mpnet/fetch_bookshelf_thin_planner.pkl",     # Path to planner model
    hardware: str = "cpu",                                          # Hardware backend, choices=['cpu', 'orangepi_npu', 'raspi_npu']
    robot: str = "fetch",                                           # Robot to plan for
    dataset: str = "problems.pkl",                                  # Pickled dataset to use
    problem: Union[str, List[str]] = ["bookshelf_thin"],            # Problem name or list of problems to evaluate
    problem_index: Union[int, List[int]] = None,                    # Problem index or list of indices to evaluate
    trials: int = 1,                                                # Number of trials to evaluate each instance
    pc_repr: str = None,                                            # Pointcloud representation
    samples_per_object: int = 10000,                                # If pointcloud, samples per object to use
    filter_type: str = "scdf",                                      # Filter type for pointcloud filtering
    filter_radius: float = 0.02,                                    # Filter radius for pointcloud filtering, required if filter_type="scdf"
    voxel_filter_size: float = 0.0308,                              # Voxel filter size for pointcloud filtering, required if filter_type="centervox"
    filter_cull: bool = True,                                       # Cull pointcloud around robot by maximum distance, unused if filter_type="centervox"
    **kwargs,
    ):

    if not Path(encoder).exists():
        print(f"Error: Encoder model not found: {encoder}")
        return 1
    
    if not Path(planner).exists():
        print(f"Error: Planner model not found: {planner}")
        return 1

    if robot not in vamp.ROBOT_JOINTS:
        raise RuntimeError(f"Robot {robot} does not exist in VAMP!")

    if not pc_repr:
        raise ValueError("pc_repr (pointcloud representation) is required. Available repr: capt, mvt")
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

    if problem_index is not None:
        if isinstance(problem_index, int):
            problem_index = [problem_index]

    total_problems = 0
    valid_problems = 0
    failed_problems = 0

    tick = time.perf_counter()
    results = []
    for name, pset in problems['problems'].items():
        if name not in problem:
            continue

        failures = []
        invalids = []
        print(f"Evaluating {robot} on {name}: ")
        for i, data in tqdm(enumerate(pset)):
            if problem_index is not None and i not in problem_index:
                continue

            total_problems += 1

            if not data['valid']:
                invalids.append(i)
                print(f"{i}th problem is invalid")
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
            
            for _ in range(trials):
                result = vmp.plan_with_mpnet(robot, data['start'], data['goals'][0], env, filtered_pc,
                                             encoder, planner, hardware)

if __name__ == "__main__":
    Fire(main)
