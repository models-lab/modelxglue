#
#
import json
import os
import pickle
import shutil
import sys
import tempfile
import subprocess

import pandas as pd

from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin

from ..features.transform import FeatureTransform
from ..models.models import ModelFactory
from ..utils.docker_env import run_docker

import docker


class FileBasedRemoteModel:
    def __init__(self, conf):
        self.conf = conf
        self.folder = os.path.dirname(conf.file)
        self.cache = tempfile.mkdtemp()

    def dump_data_as_json(self, data, path):
        if isinstance(data, ndarray):
            data = data.tolist()
        elif isinstance(data, pd.Series):
            data = data.to_list()
        elif isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="records")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def dump_data_as_pickle(self, data, path):
        # Save data with pickle
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        return path

    def dump_training_data(self, dirpath, X, y):
        X_file = os.path.join(dirpath, "X.json")
        y_file = os.path.join(dirpath, "y.json")
        self.dump_data_as_json(X, X_file)
        self.dump_data_as_json(y, y_file)
        hyper = self.dump_hyper_parameters(dirpath)
        return X_file, hyper, y_file

    def dump_hyper_parameters(self, dirpath):
        hyper = os.path.join(dirpath, "hyper.json")
        self.dump_data_as_json({"hyper": self.hyper_parameters, "seed": self.seed}, hyper)
        return hyper

    def read_input(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
            return data

    def command_replacements(self, command, root, task, stage, input, target=None, hyper=None):
        command = command.replace("{root}", root)
        command = command.replace("{task}", task)
        command = command.replace("{stage}", stage)
        command = command.replace("{mode}", stage)
        command = command.replace("{input}", input)
        command = command.replace("{original}", self.folder)
        if target:
            command = command.replace("{target}", target)
        if hyper:
            command = command.replace("{hyper}", hyper)
        return command

class PythonEnvModelFactory(ModelFactory):
    def __init__(self, conf):
        super().__init__(conf)

    def get_model_object(self, hyper_parameters, seed):
        # TODO: Handle the type of model
        print(self.conf)
        return PythonEnvClassifier(self.conf, hyper_parameters, seed)


class PythonEnvClassifier(FileBasedRemoteModel):
    def __init__(self, conf, hyper_parameters, seed):
        self.conf = conf
        self.folder = os.path.dirname(conf.file)
        self.hyper_parameters = hyper_parameters
        self.seed = seed

    def execute_command(self, command):
        call_list = command.split(" ")
        call_list.insert(0, self.python_exe())
        call_list = [x for x in call_list if x.strip() != '']

        res = subprocess.run(call_list, capture_output=False)
        if res.returncode != 0:
            print("Error while running command ", " ".join(call_list))
            raise Exception()

    def train(self, X, y=None):
        task = self.conf.task
        dirpath = self.conf.cache
        X_file = self.dump_data_as_pickle(X, os.path.join(dirpath, "X.pkl"))
        hyper = self.dump_hyper_parameters(dirpath)
        y_file = None
        if y is not None:
            y_file = self.dump_data_as_json(y, os.path.join(dirpath, "y.json"))

        self.copy_files(dirpath)
        self.install_virtual_env()
        self.install_dependencies()

        command = os.path.join(dirpath, self.conf.environment.train)
        command = self.command_replacements(command, dirpath, task, "train", X_file, y_file, hyper)
        self.execute_command(command)

    def test(self, X):
        task = self.conf.task
        dirpath = self.conf.cache
        X_file = self.dump_data_as_pickle(X, os.path.join(dirpath, "X.pkl"))
        hyper = self.dump_hyper_parameters(dirpath)

        command = os.path.join(dirpath, self.conf.environment.test)
        command = self.command_replacements(command, dirpath, task, "test", X_file, None, hyper)
        self.execute_command(command)

        return self.read_input(os.path.join(dirpath, "y_pred.json"))


    def build(self, X, requires_output=True):
        task = self.conf.task
        dirpath = self.conf.cache
        X_file = self.dump_data_as_pickle(X, os.path.join(dirpath, "X.pkl"))
        hyper = self.dump_hyper_parameters(dirpath)

        self.copy_files(dirpath)
        self.install_virtual_env()
        self.install_dependencies()

        command = os.path.join(dirpath, self.conf.environment.build)
        command = self.command_replacements(command, dirpath, task, "build", X_file, None, hyper)
        self.execute_command(command)

        if requires_output:
            return self.read_input(os.path.join(dirpath, "y_pred.json"))
        else:
            return None

    def copy_files(self, dirpath):
        shutil.copyfile(os.path.join(self.folder, 'requirements.txt'), os.path.join(dirpath, 'requirements.txt'))
        if 'more_requirements' in self.conf.environment:
            shutil.copyfile(os.path.join(self.folder, self.conf.environment.more_requirements),
                            os.path.join(dirpath, 'more_requirements.txt'))
        shutil.copyfile(os.path.join(self.folder, 'model.py'), os.path.join(dirpath, 'model.py'))
        if 'more_files' in self.conf.environment:
            for f in self.conf.environment.more_files:
                shutil.copyfile(os.path.join(self.folder, f),
                                os.path.join(dirpath, f))

        os.makedirs(os.path.join(dirpath, 'utils'), exist_ok=True)
        shutil.copyfile(os.path.join(os.path.dirname(__file__), '../utils/model_io.py'), os.path.join(dirpath, 'utils', 'model_io.py'))

    # See example
    # https://gist.github.com/mpurdon/be7f88ee4707f161215187f41c3077f6

    def install_virtual_env(self):
        virtual_dir = self.venv_dir()
        if not os.path.exists(virtual_dir):
            subprocess.call([sys.executable, "-m", "virtualenv", virtual_dir])
        else:
            print("found virtual python: " + virtual_dir)

    def install_dependencies(self):
        subprocess.call([self.pip_exe(), "install", "-r", os.path.join(self.conf.cache, "requirements.txt")])
        if 'more_requirements' in self.conf.environment:
            subprocess.call([self.pip_exe(), "install", "-r", os.path.join(self.conf.cache, "more_requirements.txt")])

    def venv_dir(self):
        return os.path.join(self.conf.cache, "venv")

    def pip_exe(self):
        return os.path.join(self.venv_dir(), "bin", "pip")

    def python_exe(self):
        return os.path.join(self.venv_dir(), "bin", "python")


def show_docker_errors_and_raise(err):
    err_list = err.stderr.split(b'\n')
    err_list = [x.decode('utf-8') for x in err_list]
    print('\n'.join(err_list))
    raise err

class DockerEnvModelFactory(ModelFactory):
    def __init__(self, conf):
        super().__init__(conf)

    def get_model_object(self, hyper_parameters, seed):
        # TODO: Handle the type of model
        print(self.conf)
        return RemoteDockerClassifier(self.conf, hyper_parameters, seed)


class RemoteDockerClassifier(FileBasedRemoteModel):
    def __init__(self, conf, hyper_parameters, seed):
        self.conf = conf
        self.folder = os.path.dirname(conf.file)
        self.hyper_parameters = hyper_parameters
        self.seed = seed

    def train(self, X, y=None):
        task = self.conf.task
        dirpath = self.conf.cache
        X_file = self.dump_data_as_json(X, os.path.join(dirpath, "X.json"))
        X_attrs = self.dump_data_as_json(X.attrs, os.path.join(dirpath, "X_attrs.json"))
        hyper = self.dump_hyper_parameters(dirpath)
        y_file = None
        if y is not None:
            y_file = self.dump_data_as_json(y, os.path.join(dirpath, "y.json"))

        X_file = "/shared/X.json"
        y_file = "/shared/y.json"
        hyper = "/shared/hyper.json"

        command = self.conf.environment.train
        command = self.command_replacements(command, '/shared', task, "train", X_file, y_file, hyper)

        try:
            run_docker(self.folder, shared_folder={dirpath: '/shared'}, command=command)
        except docker.errors.ContainerError as err:
            show_docker_errors_and_raise(err)

    def test(self, X):
        task = self.conf.task
        dirpath = self.conf.cache
        X_file = os.path.join(dirpath, "X.json")
        self.dump_data_as_json(X, X_file)
        self.dump_data_as_json(X.attrs, os.path.join(dirpath, "X_attrs.json"))
        self.dump_hyper_parameters(dirpath)

        X_file = "/shared/X.json"
        hyper = "/shared/hyper.json"

        command = self.conf.environment.test
        command = self.command_replacements(command, '/shared', task, "train", X_file, None, hyper)

        try:
            run_docker(self.folder, shared_folder={dirpath: '/shared'}, command=command)
            list_of_results = self.read_input(os.path.join(dirpath, "y_pred.json"))
            return list_of_results
        except docker.errors.ContainerError as err:
            show_docker_errors_and_raise(err)

        return self.read_input(os.path.join(dirpath, "y_pred.json"))

    def build(self, X, requires_output=True):
        task = self.conf.task
        dirpath = self.conf.cache
        X_file = os.path.join(dirpath, "X.json")
        self.dump_data_as_json(X, X_file)
        self.dump_data_as_json(X.attrs, os.path.join(dirpath, "X_attrs.json"))
        self.dump_hyper_parameters(dirpath)

        X_file = "/shared/X.json"
        hyper = "/shared/hyper.json"

        command = self.conf.environment.build
        command = self.command_replacements(command, '/shared', task, "build", X_file, None, hyper)

        try:
            run_docker(self.folder, shared_folder={dirpath: '/shared'}, command=command)
            list_of_results = self.read_input(os.path.join(dirpath, "y_pred.json"))
            return list_of_results
        except docker.errors.ContainerError as err:
            show_docker_errors_and_raise(err)

        if requires_output:
            return self.read_input(os.path.join(dirpath, "y_pred.json"))
        else:
            return None

class DockerTransform(FeatureTransform, FileBasedRemoteModel):

    def __init__(self, conf, transform_data, transform_folder=None):
        self.conf = conf
        self.transform_data = transform_data
        if transform_folder:
            self.transform_folder = transform_folder
        else:
            # This is when the transform data is in the same folder as the conf file (e.g. model.yaml)
            self.transform_folder = os.path.join(os.path.dirname(conf.file), self.transform_data.folder)

    def transform(self, X, what):
        #transform_folder = os.path.join(self.conf.cache, self.transform_data.type, what)
        #X_file = os.path.join(transform_folder, "X.json")
        dirpath = self.conf.cache
        X_file = os.path.join(dirpath, "X.json")
        self.dump_data_as_json(X, X_file)

        try:
            command = self.transform_data.run
            command = command.replace("{root}", "/shared")
            #run_docker(folder, shared_folder={transform_folder: '/shared'}, command=command)
            run_docker(self.transform_folder, shared_folder={dirpath: '/shared'}, command=command)
        except docker.errors.ContainerError as err:
            show_docker_errors_and_raise(err)

        #as_json = self.read_input(os.path.join(transform_folder, "transformed.json"))
        as_json = self.read_input(os.path.join(dirpath, "transformed.json"))
        return pd.DataFrame(as_json)

