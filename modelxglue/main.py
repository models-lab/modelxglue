import json
import logging
import os
import random
import warnings

import yaml

import hydra
import numpy as np
import hashlib

from omegaconf import DictConfig, OmegaConf

from modelxglue.evaluation_strategies import train_test_val, k_fold_alone, clustering, recommendation
from modelxglue.features.features import FEATURES
from modelxglue.features.transform import NoneFeatureTransform, DumpXmiTransform, TransformConfiguration, \
    CompositeTransform, \
    VectorizeText, KernelTransform
from modelxglue.models.remote import PythonEnvModelFactory, DockerEnvModelFactory, DockerTransform
from modelxglue.utils.common import read_dataset, get_model_hyperparameters, get_model_parameters

warnings.filterwarnings("ignore")

logger = logging.getLogger()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def save_results(output_path, results, configuration):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_dict = {
        "results": results,
        "configuration": configuration
    }
    with open(output_path, 'w') as f:
        json.dump(new_dict, f)


def load_model_configuration(model_configuration_file: str, args: dict) -> DictConfig:
    model_configuration_file = os.path.expandvars(model_configuration_file)
    with open(model_configuration_file, 'r') as f:
        for key, value in args.items():
            OmegaConf.register_new_resolver(f"args.{key}", lambda default_value: value, replace=True)
        yaml_conf = yaml.safe_load(f)
        conf = OmegaConf.create(yaml_conf)
        conf.file = model_configuration_file
        return conf


# Creates the cache directory for this model and modifies the configuration
# to point to it (cache attribute).
def set_cache_dir(conf, cachedir):
    hash = hashlib.md5()
    hash.update(conf.name.encode('utf-8'))
    id = conf.task + "-" + conf.model + "-" + hash.hexdigest()
    dir = os.path.join(cachedir, id)
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    conf.cache = dir


def get_model_factory(conf: DictConfig):
    if 'python' in conf.environment:
        return PythonEnvModelFactory(conf)
    elif 'docker' in conf.environment:
        return DockerEnvModelFactory(conf)
    else:
        raise ValueError(f'No model factory for {conf.environment}')


def get_transform_by_name(t, cfg):
    type = t['type']
    if type == 'xmi-dump':
        return DumpXmiTransform(cfg.cache)
    elif type == 'docker':
        return DockerTransform(cfg, t)
    elif type == 'vectorize-text':
        separator = ' ' if 'separator' not in t else t.separator
        return VectorizeText(t.columns, t.strategy, separator)
    elif type == 'kernel':
        return KernelTransform(t.column)
    elif type == 'composite':
        steps = [get_transform_by_name(s, cfg) for s in t.steps]
        return CompositeTransform(steps)
    else:
        raise ValueError(f'Transform {type} not supported.')


def get_transform(cfg) -> TransformConfiguration:
    result = TransformConfiguration()

    transform = cfg.get('transform', None)
    if transform is None or len(transform) == 0:
        return result

    for t in transform:
        when = t['only'] if 'only' in t else 'all'
        transform_object = get_transform_by_name(t, cfg)
        result.add(when, transform_object)

    return result


def execute_task(cfg, cfg_transform, model_hyperparameters, ml_model, pd_dataset):
    if cfg.task.task_name == 'classification':
        if cfg.task.evaluation_strategy == 'train_test_val':
            results = train_test_val(pd_dataset, seed=cfg.seed, features=cfg.model.encoding_features,
                                     hyperparameters=model_hyperparameters, model=ml_model,
                                     metric_name=cfg.task.metric,
                                     cfg_transform=cfg_transform,
                                     train_test_val_splits=(cfg.task.train_split,
                                                            cfg.task.test_split,
                                                            cfg.task.val_split))
        elif cfg.task.evaluation_strategy == 'k_fold':
            results = k_fold_alone(pd_dataset, seed=cfg.seed, features=cfg.model.encoding_features,
                                   hyperparameters=model_hyperparameters, model=ml_model,
                                   folds=cfg.task.folds,
                                   cfg_transform=cfg_transform,
                                   metric_name=cfg.task.metric)
        else:
            raise ValueError(f'{cfg.task.evaluation_strategy} strategy not supported.')
    elif cfg.task.task_name == 'clustering':
        results = clustering(pd_dataset, seed=cfg.seed, features=cfg.model.encoding_features,
                             hyperparameters=model_hyperparameters, model=ml_model,
                             resampling=cfg.task.resampling, metric_name=cfg.task.metric,
                             size_dataset=cfg.task.size_dataset,
                             cfg_transform=cfg_transform)
    elif cfg.task.task_name == 'feature_recommendation':
        results = recommendation(pd_dataset, seed=cfg.seed, features=cfg.model.encoding_features,
                                 metric_name=cfg.task.metric,
                                 train_test_val_splits=(cfg.task.train_split,
                                                        cfg.task.test_split,
                                                        cfg.task.val_split),
                                 model=ml_model, topk=cfg.task.topk,
                                 cfg_transform=cfg_transform,
                                 hyperparameters=model_hyperparameters)
    else:
        raise ValueError(f'{cfg.task.task_name} not supported.')
    return results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # This is not straightforward, because now everyone can define its own model
    # some_compatibility_checks(cfg.model.ml_model, cfg.model.encoding_features)
    seed_all(cfg.seed)

    pd_dataset = read_dataset(path_hg=cfg.dataset.dataset_hg,
                              model_type=cfg.dataset.model_type,
                              include_duplicates=cfg.dataset.include_duplicates,
                              task=cfg.task.task_name)
    logger.info(f'Loaded {cfg.dataset.dataset_hg}, samples: {len(pd_dataset)}')

    model_hyperparameters = get_model_hyperparameters(cfg.model.hyperparameters) \
        if "hyperparameters" in cfg.model else []
    logger.info(f'Number of models of hyperparameters: {len(model_hyperparameters)}')

    args_model = get_model_parameters(cfg.model)
    ml_model = cfg.model.ml_model
    cfg_transform = TransformConfiguration()

    for arg in args_model:
        if "reference" in cfg.model:
            conf = load_model_configuration(cfg.model.reference, arg)
            set_cache_dir(conf, cfg.cachedir)
            ml_model = get_model_factory(conf)
            cfg_transform = get_transform(conf)

        results = execute_task(cfg, cfg_transform, model_hyperparameters, ml_model, pd_dataset)

        args_id = '-'.join([v for k, v in sorted(arg.items())])
        OmegaConf.register_new_resolver('model.current_args', lambda default_value: args_id, replace=True)
        logger.info(f'Saving results at {cfg.output}')
        save_results(cfg.output, results, OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
