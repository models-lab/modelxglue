import json
import os
import pickle
from argparse import ArgumentParser

import joblib
import numpy as np
from numpy import ndarray


def read_input(filename):
    if filename.endswith(".pkl"):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            return data
    else:
        with open(filename, "r") as f:
            data = json.load(f)
            return data


def dump_data(data, path):
    if isinstance(data, ndarray):
        data = data.tolist()

    with open(path, "w") as f:
        json.dump(data, f)


class ModelImplementation:

    # By default it uses the joblib library
    def dump_model(self, ml_model: object, path):
        joblib.dump(ml_model, path)

    def load_model(self, path):
        print("Loading model from {}".format(path))
        return joblib.load(path)

    def get_serialized_model_file(self, model_inout):
        return os.path.join(model_inout.root, self.get_serialized_model_name())

    def get_serialized_model_name(self):
        return 'model.joblib'

    def get_vector_input(self, X):
        matrix = X.select_dtypes(include=[np.number])
        return matrix

    def ignore_train(self):
        return False


class ModelInOut:
    def __init__(self, root, input, stage, hyper_file=None):
        self.root = root
        self.input = input  # The X
        self.target = None  # The y
        self.stage = stage
        self.original = None

        self.hyper = {}
        self.seed = 123
        if hyper_file:
            hyper_object = read_input(hyper_file)
            self.hyper = hyper_object['hyper']
            if 'seed' in self.hyper:
                self.seed = hyper_object['seed']

    def execute_model(self, model_impl: ModelImplementation):
        serialized_model_file = model_impl.get_serialized_model_file(self)
        X = read_input(self.input)

        if model_impl.ignore_train() and self.stage == "train":
            return

        if self.stage == "train":
            y = None
            if self.target:
                y = read_input(self.target)
            model = model_impl.train(X, y, self)
            model_impl.dump_model(model, serialized_model_file)
        elif self.stage == "test":
            if model_impl.ignore_train():
                model = None
            else:
                model = model_impl.load_model(serialized_model_file)
                assert model is not None, "Model is None"

            y_pred = model_impl.test(model, X, self)
            dump_data(y_pred, os.path.join(self.root, 'y_pred.json'))
        elif self.stage == "build":
            y_pred = model_impl.build(X, self)
            dump_data(y_pred, os.path.join(self.root, 'y_pred.json'))
        else:
            raise Exception("Unknown stage: {}".format(self.stage))


def parse_args(fixed_stage=None):
    parser = ArgumentParser(description='Execute a model')
    parser.add_argument("--root", dest="root", help="the root folder for the resources", required=True)
    parser.add_argument("--input", dest="input", help="the input data", required=True)
    parser.add_argument("--target", dest="target", help="the target variable", required=False)
    if fixed_stage is None:
        parser.add_argument("--stage", dest="stage", help="train or test", required=True)
    parser.add_argument("--hyper", dest="hyper", help="path to the hyperparameters file", required=False)
    parser.add_argument("--original", dest="original", help="path to the folder with the original model", required=False)

    args = parser.parse_args()
    print(fixed_stage)
    stage = fixed_stage if fixed_stage is not None else args.stage
    model_inout = ModelInOut(args.root, args.input, stage, args.hyper)
    model_inout.target = args.target
    model_inout.original = args.original
    return model_inout


def execute_model(callback, fixed_stage=None):
    model_inout = parse_args(fixed_stage)
    callback(model_inout)
