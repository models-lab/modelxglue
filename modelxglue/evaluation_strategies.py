import json
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm

from .dataset.split import SplitConfiguration
from .features.features import get_features, dataset_as_format
from .features.transform import TransformConfiguration
from .models.models import get_model_object, ModelFactory
from .utils.metrics import eval_metrics


def do_train(model2train, X, y):
    if isinstance(model2train, ClassifierMixin):
        model2train.fit(X, y)
    else:
        model2train.train(X, y)


def do_test(model2train, X):
    if isinstance(model2train, ClassifierMixin):
        return model2train.predict(X)
    else:
        return model2train.test(X)


def train_test_val(dataset, seed, features, hyperparameters, model, metric_name, cfg_transform: TransformConfiguration,
                   train_test_val_splits=(0.7, 0.2, 0.1)):
    ids_train_val, ids_test, y_train_val, y_test = train_test_split(list(dataset['ids']), list(dataset['labels']),
                                                                    test_size=train_test_val_splits[1],
                                                                    random_state=seed,
                                                                    stratify=list(dataset['labels']))
    ids_train, ids_val, y_train, y_val = train_test_split(ids_train_val, y_train_val,
                                                          test_size=train_test_val_splits[2] / train_test_val_splits[0],
                                                          random_state=seed,
                                                          stratify=y_train_val)
    scores = []
    if len(hyperparameters) > 1:
        # This is to keep supporting built-in models, we will remove the else branch when built-in models go away
        if isinstance(model, ModelFactory):
            X_train, X_val = dataset_as_format(dataset, model.conf.dataset.format, ids_train, ids_val)
            training_ft, val_ft = cfg_transform.get_transform_for('training-set', 'test-set')
            X_train = training_ft.transform(X_train, what='training-set')
            X_val = val_ft.transform(X_val, what='validation-set')
        else:
            X_train, X_val = get_features(features, dataset, dataset, ids_train, ids_val)

        for p in tqdm(hyperparameters):
            model2train = get_model_object(model, p, seed)
            do_train(model2train, X_train, y_train)

            y_pred = do_test(model2train, X_val)
            score = eval_metrics(metric_name, y_true=y_val, y_pred=y_pred)[0]
            scores.append(score)
    elif len(hyperparameters) == 1:
        scores.append(0)

    if scores:
        idx = np.argmax(scores)
        best_hyper = hyperparameters[idx]
    else:
        best_hyper = {}

    # evaluation test set
    if isinstance(model, ModelFactory):
        X_train_val, X_test = dataset_as_format(dataset, model.conf.dataset.format, ids_train_val, ids_test)
        training_ft, test_ft = cfg_transform.get_transform_for('training-set', 'test-set')
        X_train_val = training_ft.transform(X_train_val, what='training-validation-set')
        X_test = test_ft.transform(X_test, what='test-set')
    else:
        X_train_val, X_test = get_features(features, dataset, dataset, ids_train_val, ids_test)

    model2train = get_model_object(model, best_hyper, seed)
    do_train(model2train, X_train_val, y_train_val)
    y_pred = do_test(model2train, X_test)
    score_test = eval_metrics(metric_name, y_true=y_test, y_pred=y_pred)

    result = {
        "score_test": score_test,
        "best_hyper": best_hyper,
        "scores_validation": scores
    }
    return result


def k_fold(dataset, seed, features, hyperparameters, model, folds, cfg_transform: TransformConfiguration, metric_name):
    docs_corpus, y_corpus, corpus_ids = list(dataset['txt']), list(dataset['labels']), list(dataset['ids'])

    skf = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
    scores = defaultdict(list)
    for train_index, test_index in tqdm(skf.split(docs_corpus, y_corpus),
                                        desc='Iteration over folds', total=folds):
        y_train, y_val = np.array(y_corpus)[train_index], np.array(y_corpus)[test_index]
        ids_train, ids_val = np.array(corpus_ids)[train_index], np.array(corpus_ids)[test_index]
        # This is to keep supporting built-in models, we will remove the else branch when built-in models go away
        if isinstance(model, ModelFactory):
            X_train, X_val = dataset_as_format(dataset, model.conf.dataset.format, ids_train, ids_val)
            training_ft, val_ft = cfg_transform.get_transform_for('training-set', 'test-set')
            X_train = training_ft.transform(X_train, what='training-set')
            X_val = val_ft.transform(X_val, what='validation-set')
        else:
            X_train, X_val = get_features(features, dataset, dataset, ids_train, ids_val)

        # hyperparameter selection
        if hyperparameters:
            for p in tqdm(hyperparameters, desc='Iteration over hyperparameters'):
                model2train = get_model_object(model, p, seed)
                do_train(model2train, X_train, y_train)

                # the score is the accuracy
                y_pred = do_test(model2train, X_val)
                s = eval_metrics(metric_name, y_true=y_val, y_pred=y_pred)
                scores[json.dumps(p)].append(s)
        else:
            p = {}
            model2train = get_model_object(model, p, seed)
            do_train(model2train, X_train, y_train)
            y_pred = do_test(model2train, X_val)
            s = eval_metrics(metric_name, y_true=y_val, y_pred=y_pred)
            scores[json.dumps(p)].append(s)
    return scores


def k_fold_alone(dataset, seed, features, hyperparameters, model, folds, cfg_transform: TransformConfiguration,
                 metric_name):
    scores = k_fold(dataset, seed, features, hyperparameters, model, folds, cfg_transform, metric_name)
    scores_mean = {x: np.mean(np.array(y), axis=0).astype(np.float64).tolist() for x, y in scores.items()}
    scores = {x: np.array(y).astype(np.float64).tolist() for x, y in scores.items()}

    best_hyper = None
    best_mean = -1
    for h, mean in scores_mean.items():
        if mean[0] > best_mean:
            best_mean = mean[0]
            best_hyper = h

    result_best_hyper = {
        "score_folds": scores[best_hyper],
        "score_folds_mean": scores_mean[best_hyper],
        "hyperparameter": json.loads(best_hyper)
    }
    result = {
        "all_scores": scores,
        "mean_all_scores": scores_mean,
        "results_best_hyperparameter": result_best_hyper
    }
    return result


def clustering(dataset, seed, features, hyperparameters, model, resampling, metric_name, size_dataset,
               cfg_transform: TransformConfiguration):
    scores = defaultdict(list)
    ids_train = dataset['ids']
    ids_test = []
    y_train = dataset['labels']
    num_labels = len(set(y_train))

    X_train, _ = dataset_as_format(dataset, model.conf.dataset.format, ids_train, ids_test)
    training_ft, *_ = cfg_transform.get_transform_for('training-set')
    X_train = training_ft.transform(X_train, what='training-set')

    if hyperparameters:
        for p in tqdm(hyperparameters, desc='Hyperparameters'):
            if 'n_clusters' in p and p['n_clusters'] == 'compute':
                p['n_clusters'] = num_labels

            model2train = get_model_object(model, p, seed)
            labels = model2train.build(X_train)
            s = eval_metrics(metric_name, X=X_train, y_true=y_train, y_pred=labels)
            scores[json.dumps(p)].append(s)
    else:
        p = {}
        model2train = get_model_object(model, p, seed)
        labels = model2train.build(X_train)
        s = eval_metrics(metric_name, X=X_train, y_true=y_train, y_pred=labels)
        scores[json.dumps(p)].append(s)

    scores_mean = {x: np.mean(np.array(y), axis=0).astype(np.float64).tolist() for x, y in scores.items()}
    scores = {x: np.array(y).astype(np.float64).tolist() for x, y in scores.items()}
    result = {
        "all_scores": scores,
        "mean_all_scores": scores_mean
    }
    return result


DEBUG = False


def debug(v, name):
    if not DEBUG:
        return
    from numpy import ndarray
    if isinstance(v, ndarray):
        print("Skipping dumping numpy array")
        return

    dump_data(v, "/tmp", name)


def dump_data(v, path, name):
    # check if v is a DataFrame
    if isinstance(v, pd.DataFrame):
        v.to_csv(os.path.join(path, name + ".csv"), index=False)
    else:
        with open(os.path.join(path, name + ".json"), "w") as f:
            json.dump(v, f)


def recommendation(seed, features, metric_name,
                   split: SplitConfiguration, model, topk,
                   cfg_transform: TransformConfiguration,
                   hyperparameters, config):

    scores = []
    if len(hyperparameters) > 1:
        logging.info("Validation...")

        X_train, X_val = get_features(features, split.train_dataset, split.val_dataset)

        training_ft, val_ft = cfg_transform.get_transform_for('training-set', 'test-set')
        X_train = training_ft.transform(X_train, what='training-set')
        df_val = val_ft.transform(X_val, what='validation-set')

        for p in tqdm(hyperparameters):
            model2train = get_model_object(model, p, seed)
            model2train.train(X_train)

            y_pred = model2train.test(df_val)
            y_pred = [x[:topk] for x in y_pred]
            y_val = [t.split(',') for t in df_val['target']]

            debug(df_val, "df_val")
            debug(y_pred, "y_val_pred")
            debug(y_val, "y_val")

            score = eval_metrics(metric_name, y_true=y_val, y_pred=y_pred)[0]
            scores.append(score)
    elif len(hyperparameters) == 1:
        scores.append(0)

    if scores:
        idx = np.argmax(scores)
        best_hyper = hyperparameters[idx]
    else:
        best_hyper = {}

    logging.info("Test...")
    # evaluation test set
    X_train_val, X_test = get_features(features, split.train_val_dataset, split.test_dataset)

    training_ft, test_ft = cfg_transform.get_transform_for('training-set', 'test-set')
    X_train_val = training_ft.transform(X_train_val, what='training-validation-set')
    debug(X_train_val, "X_train_val")
    debug(X_test[['ids']], "X_test")

    model2train = get_model_object(model, best_hyper, seed)
    model2train.train(X_train_val)

    df_test = test_ft.transform(X_test, what='test-set')
    debug(df_test, "df_test")

    y_pred = model2train.test(df_test)
    y_pred = [x[:topk] for x in y_pred]
    y_test = [t.split(',') for t in df_test['target']]

    debug(y_pred, "y_test_pred")
    debug(y_test, "y_test")

    dump_data(y_pred, model.conf.cache, "y_test_pred")
    dump_data(y_test, model.conf.cache, "y_test")

    score_test = eval_metrics(metric_name, y_true=y_test, y_pred=y_pred)

    result = {
        "score_test": score_test,
        "best_hyper": best_hyper,
        "scores_validation": scores
    }
    return result
