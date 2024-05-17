# ModelXGlue Documentation
Here you will find all information on how to configure and use ModelXGlue and how to expand it with new benchmarks and models. 

## Getting started
ModelXGlue is a benchmarking framework designed to support the comparison of ML-based systems for MDE tasks. In it, researchers can customize a benchmark for a specific ML/MDE task by selecting a dataset, ML models (that can be develeoped by different researchers) and relevant metrics to ensure the performance of the ML models in the selected task.

Moreover, each component can operate in a
separate environment to prevent conflicts with libraries
and enable the use of various technologies. Additionally,
ModelXGLue becomes completely automated once input
configurations are provided. 

## Installation
These instructions are based on the assumption that Python 3.8 and Git are installed and are accessible via the aliases "python" and "git", respectively. Older python version could raise some incompatibilities. 

Firstly, clone the git repositories:
```
https://github.com/models-lab/modelxglue
https://github.com/models-lab/modelxglue-mde-components
```

The first repository is ModelXGlue itself. The second repository is where all components such as models and benchmarks are defined and it also shows how the new components could be organised in a clean way. 

You can optionally create a Conda environment for the installation:
```
conda create --name modelxglue python=3.8
conda activate modelxglue
```

Then install all dependencies required by ModelXGlue:
```
python setup.py install
```

Lastly, define the environment variable "$COMPONENTS" as the path to the directory that will contain all the components (models). For example:
```
export COMPONENTS="/home/X/modelxglue-mde-components/components"
```

## First examples
Next, a couple of initial examples are shown.

- Execute a benchmark in the clustering task, using the modelset_ecore_deduplicated dataset and as a model, use kmeans only. 
```
python -m modelxglue.main --config-dir ../modelxglue-mde-components/benchmarks/clustering model=kmeans dataset=modelset_ecore_deduplicated task=clustering
```

- Execute a benchmark in the classification task using the k-fold strategy, using the modelset_ecore_deduplicated dataset and evaluating it with two models: ffnn and mar. 
```
python -m modelxglue.main --config-dir ../modelxglue-mde-components/benchmarks/classification/ model=[ffnn,mar] dataset=modelset_ecore_deduplicated task=classification_k_fold --multirun
```

The results will be written on the results folder of the modelxglue directory. 

## Definition of a new reusable ML model (under "components" directory)
Using the strcture defined in the components repository, create a new directory with the name of the model inside the associated task. For example, if one wants to define a KNN model for classification, create the directory "knn" inside "/home/X/modelxglue-mde-components/components/classification". 

If the model is already implemented a in Python library (such as sklearn), one good way to define the model will contain at least the following files:

- model.py: Implements how the training of the model will be performed. 
  This should be defined under a Model class that inherits from a base class defined in ModelXGlue called "ModelImplementation". For instance, this, could be the file if we want to use a KNN Classifier:
```
import os
import sys

from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.dirname(__file__))
from utils.model_io import ModelInOut, execute_model, ModelImplementation

  class Model(ModelImplementation):
    def train(self, X, y, inout: ModelInOut):
        X = self.get_vector_input(X)
        model = KNeighborsClassifier(**inout.hyper)
        model.fit(X, y)
        return model

    def test(self, loaded_model, X, inout: ModelInOut):
        X = self.get_vector_input(X)
        y_pred = loaded_model.predict(X)
        return y_pred


    def main(inout: ModelInOut):
    inout.execute_model(Model())


    if __name__ == "__main__":
        execute_model(main)
        exit(0)
  ```
  
- model.yaml: It contains a description of the configuration of 
  the model used. It defines the parameters and hyperparameters needed, the format of the dataset and the transformations it needs to be fed to the model and finally, details about the environment where the model will be executed. In the case of the KNN, it could be:
```
name: Model classification with Scikit-Learn models (KNN)
task: classification
model: KNN

parameters:
  encoding: [TFIDF, GloVe]

dataset:
  format: txt

transform:
  - type: vectorize-text
    strategy: ${args.encoding:tfidf}
    columns: [txt]
    separator: newline

hyper:
  n_neighbors: "a list of integers"

environment:
  python: venv
  train: model.py --stage train --root {root} --input {input} --target {target} --hyper {hyper}
  test:  model.py --stage test  --root {root} --input {input}
```

- requirements.txt: It defines all the dependencies required by the model. 
In the case of KNN, this could be:
```
joblib==1.1.1
pandas==1.5.3
scikit-learn==1.0.2
```

In other cases, this structure will vary as needed. For example, when defining a model that isn't defined in a library, we might need an extra file that describes it. For instance, to describe a Graph Neural Network (GNN), we might create the file "GNN.py" that defines all the layers of the neural network, how the training is performed and how the inference is done. If for any reasons the model is going to be executed inside a Docker, then a Dockerfile might be needed. If the model is defined in Java (e.g, Lucene), you will also need the Java project and a Dockerfile. 

Each model is independent from the others so they can be executed using different technologies.

## Construction of a benchmark (under "benchmark" directory)
A benchmark is defined using the models defined in the previous section, selecting a task and the dataset that will be used. In order to construct a benchmark, the following items have to be defined.

- Models: For each model being benchmarked, create a configuration file
    that refers to the model defined as a component in the previous step. It should define all the needed hyperparameters as well. For example: 
```
reference: : $COMPONENTS/aurora
ml_model: FFNN
hyperparameters:
- size: [50, 100, 150]
```

- Datasets: Define the datasets that will be used as input of the models. An example fo the ecore dedup dataset is
the following:
```
dataset_name: "modelset_ecore_dedup"
model_type: ecore
dataset_hg: "antolin/modelset"
include_duplicates: false
```

- Task: Describe the task to be benchmarked defining the type of evaluation and other relevant evaluation metrics. 
In the case of defining a task for classification using 10 folds and also calculating the balanced accuracy score
and the accuracy could be the following:
```
task_name: "classification_k_fold"
evaluation_strategy: "k_fold"
folds: 10
metric: ["balanced_accuracy_score",
"accuracy_score"]
```

## Structure of the files
```
modelxglue (repository 1)
|
\-- results
    |
    \-- result_benchmark1_model1.txt
    \-- result_benchmark1_model2.txt
    \-- result_benchmark2_model1.txt
    \-- ...
\-- setup.py
\-- other implementation files

modelxglue-mde-components (repository 2)
|
\-- benchmarks
    |
    \-- benchmark1
        |
        \-- config.yaml (define seed, output path, cachedir, defaults...)
        \-- dataset
            \-- dataset1.yaml (define name and type of the dataset, location in HuggingFace or other needed parameters)
            \-- dataset2.yaml
            \-- ...
        \-- model
            \-- model1.yaml (define the values of the hyperparameters, encoding features and where is the component)
            \-- model2.yaml
            \-- ...
        \-- task
            \-- task1.yaml (define the task name, evaluation strategy, parameters such as folds, and metrics to evaluate)
            \-- task2.yaml
            \-- ...
    \- benchmark2
        \-- ...
    \- benchmark3
    \- ...
```