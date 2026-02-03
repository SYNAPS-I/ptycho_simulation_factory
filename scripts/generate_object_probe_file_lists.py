#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List

import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import BatchSimulationTask  # noqa: E402


class FileListTask(BatchSimulationTask):
    def build(self):
        self.build_object_dataset()
        self.build_probe_dataset()


def resolve_index_paths(root_dir: str, entries: List[str]) -> List[str]:
    resolved = []
    for entry in entries:
        if os.path.isabs(entry):
            abs_path = os.path.abspath(entry)
        else:
            abs_path = os.path.abspath(os.path.join(root_dir, entry))
        resolved.append(os.path.relpath(abs_path, root_dir))
    return resolved


def get_object_paths(task: FileListTask) -> List[str]:
    dataset = task.object_dataset
    return resolve_index_paths(dataset.root_dir, list(dataset.index))


def get_probe_paths(task: FileListTask) -> List[str]:
    dataset = task.probe_dataset
    return resolve_index_paths(dataset.root_dir, list(dataset.index))


def sample_probe_indices(task: FileListTask, n_objects: int) -> List[int]:
    dataset = task.probe_dataset
    probe_paths = get_probe_paths(task)
    if dataset.probe_file_list is not None:
        return [task.get_deterministic_probe_index() for _ in range(n_objects)]
    if task.sample_probe_randomly:
        p = None
        if dataset.name_probability_map is not None:
            p = np.array(
                [
                    float(dataset.name_probability_map.get(os.path.basename(item), 0.0))
                    for item in probe_paths
                ],
                dtype=float,
            )
            if p.sum() <= 0:
                p = None
            else:
                p = p / p.sum()
        indices = np.random.choice(a=len(dataset), size=n_objects, p=p)
        return [int(i) for i in indices]
    return [task.get_deterministic_probe_index() for _ in range(n_objects)]


def build_defocus_list(task: FileListTask, n_objects: int) -> List[float]:
    defocus_range = task.config.get("probe_dataset", {}).get("random_defocus_range_m")
    if defocus_range is None:
        return [0.0] * n_objects
    if len(defocus_range) != 2:
        raise ValueError("random_defocus_range_m must have length 2")
    low, high = float(defocus_range[0]), float(defocus_range[1])
    return np.random.uniform(low, high, size=n_objects).astype(float).tolist()


def write_list(path: str, items: List[str]) -> None:
    with open(path, "w") as f:
        for item in items:
            f.write(str(item) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate object/probe/defocus lists from a batch simulation config."
    )
    parser.add_argument("config", type=str)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to write object_file_list.txt, probe_file_list.txt, probe_defocus_list.txt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Optional random seed for reproducible probe/defocus sampling.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(int(args.seed))

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    task = FileListTask(config)
    object_paths = get_object_paths(task)
    probe_paths = get_probe_paths(task)

    probe_indices = sample_probe_indices(task, len(object_paths))
    selected_probe_paths = [probe_paths[i] for i in probe_indices]
    defocus_list = build_defocus_list(task, len(object_paths))

    os.makedirs(args.output_dir, exist_ok=True)
    write_list(os.path.join(args.output_dir, "object_file_list.txt"), object_paths)
    write_list(os.path.join(args.output_dir, "probe_file_list.txt"), selected_probe_paths)
    write_list(os.path.join(args.output_dir, "probe_defocus_list.txt"), defocus_list)


if __name__ == "__main__":
    main()
