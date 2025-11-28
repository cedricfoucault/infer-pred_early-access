"""
Dataset path utilities.

This module defines the datasets used in the project and provides helper
functions to resolve dataset-specific paths, such as the subjects file
and the corresponding task configuration file. This can be used by analysis
scripts to easily load the appropriate data and configurations for a given
experimental dataset.
"""

import os

DATASET_PATHS = {
    "change-point-env": {
        "subject_list": os.path.join("data", "change-point-env", "subjects.csv"),
        "task_config": os.path.join("config", "task", "change-point-env.yaml"),
    },
    "random-walk-env": {
        "subject_list": os.path.join("data", "random-walk-env", "subjects.csv"),
        "task_config": os.path.join("config", "task", "random-walk-env.yaml"),
    },
}

def get_dataset_paths(dataset_name):
    """
    Given an experimental dataset name, return the paths to the subjects.csv
    and task config files.
    """
    try:
        ds = DATASET_PATHS[dataset_name]
    except KeyError:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Available datasets: {list(DATASET_PATHS.keys())}")

    return ds["subject_list"], ds["task_config"]
