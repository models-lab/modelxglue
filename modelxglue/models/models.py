from omegaconf import DictConfig
from typing import Union

class ModelFactory:
    def __init__(self, conf: DictConfig):
        self.conf = conf

    def get_model_object(self, hyper_parameters, seed):
        pass


def get_model_object(model_type: Union[str, ModelFactory], p, seed):
    if isinstance(model_type, ModelFactory):
        return model_type.get_model_object(p, seed)

    raise Exception("Invalid model type") 
