from __future__ import annotations

from typing import List

import numpy as np
from omegaconf import ListConfig
from sklearn.metrics import silhouette_score, v_measure_score, balanced_accuracy_score, accuracy_score


def f1_score_ir(y_true, y_pred):
    # Both y_true and y_pred needs to in lowercase
    y_true = [y.lower() for y in y_true]
    y_pred = [y.lower() for y in y_pred]

    # Fill y_pred with None values for missing elements with respect to y_true
    y_pred = [y_pred[i] if i < len(y_pred) else None for i in range(len(y_true))]

    intersection = [i for i in y_true if i in y_pred]
    prec = float(len(intersection)) / float(len(y_pred))
    recall = float(len(intersection)) / float(len(y_true))
    if prec + recall == 0:
        return 0.
    return 2 * prec * recall / (prec + recall)


def reciprocal_rank(y_true, y_pred):
    assert len(y_true) == 1
    y_true = y_true[0]
    if y_true not in y_pred:
        return 0
    else:
        return 1. / (y_pred.index(y_true) + 1)


def success_rate(y_true, y_pred):
    assert len(y_true) == 1
    y_true = y_true[0]
    if y_true in y_pred:
        return 1
    else:
        return 0


def eval_metric(metric, X=None, y_true=None, y_pred=None):
    if metric == 'silhouette_score':
        return silhouette_score(X, y_pred)
    if metric == 'v_measure_score':
        return v_measure_score(labels_true=np.array(y_true), labels_pred=np.array(y_pred))
    if metric == 'balanced_accuracy_score':
        return balanced_accuracy_score(y_true, y_pred)
    if metric == 'accuracy_score':
        return accuracy_score(y_true, y_pred)
    if metric == 'f1_score_ir':
        f1_scores = []
        for y1, y2 in zip(y_true, y_pred):
            f1_scores.append(f1_score_ir(y1, y2))
        return np.mean(f1_scores)
    if metric == 'mrr':
        rrs = []
        for y1, y2 in zip(y_true, y_pred):
            rrs.append(reciprocal_rank(y1, y2))
        return np.mean(rrs)
    if metric == 'sr':
        srs = []
        for y1, y2 in zip(y_true, y_pred):
            srs.append(success_rate(y1, y2))
        return np.mean(srs)


def eval_metrics(metrics: str | ListConfig, X=None, y_true=None, y_pred=None) -> List[float]:
    if isinstance(metrics, str):
        return [eval_metric(metrics, X, y_true, y_pred)]
    elif isinstance(metrics, ListConfig):
        return [eval_metric(m, X, y_true, y_pred) for m in metrics]
