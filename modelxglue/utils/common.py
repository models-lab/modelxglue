import itertools
import json
import os
import shutil
import subprocess
from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def get_root():
    root = os.environ.get('EXPERIMENT_ROOT')
    if root is None:
        raise NameError('No EXPERIMENT_ROOT variable.')
    return root


def copy_files(root, file_list, tgt_folder, remove_target=False):
    if remove_target and os.path.exists(tgt_folder):
        shutil.rmtree(tgt_folder)

    for relative_file in file_list:
        src = os.path.join(root, relative_file)
        dst = os.path.join(tgt_folder, relative_file)

        print("Copying ", src, " to ", dst)
        file_folder = os.path.dirname(dst)
        if not os.path.exists(file_folder):
            os.makedirs(file_folder)

        shutil.copyfile(src, dst)


def read_dataset(path_hg, model_type='ecore', include_duplicates=True, task="classification"):
    ds = load_dataset(path_hg, split="train").filter(lambda x: x['model_type'] == model_type)
    if include_duplicates:
        ds = ds.to_pandas()
    else:
        ds = ds.filter(lambda x: not x['is_duplicated'])
        ds = ds.to_pandas()
    if task == "classification" or task == "clustering":
        return ds
    if "feature_recommendation" in task:
        return ds
    raise ValueError(f"Task {task} not supported.")


def get_model_hyperparameters(data: DictConfig):
    data = OmegaConf.to_container(data, resolve=True)
    keys, values = zip(*data.items())
    permutations_hyper = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations_hyper


def get_model_parameters(cfg: DictConfig):
    if 'parameters' in cfg:
        data = cfg.parameters
        # get all key-value pairs of data.parameters which are listed in data.parameters.combine
        # and create a list of dictionaries with all possible combinations of these key-value pairs
        param_key_values = {k: v for k, v in data.items() if k in cfg.parameters.combine}
        keys, values = zip(*param_key_values.items())
        permutations_hyper = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return permutations_hyper
    else:
        return [{}]
