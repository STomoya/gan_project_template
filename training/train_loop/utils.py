
from typing import Iterable

import storch
from omegaconf import DictConfig
from storch.dataset import ImageFolder, make_transform_from_config


def build_dataset(config: DictConfig):
    transform = make_transform_from_config(config.transforms)
    dataset = ImageFolder(config.data_root, transform)
    dataloader = dataset.setup_loader(**config.loader).toloader()
    return dataloader

def build_models(config: DictConfig):
    G = storch.construct_class_by_name(**config.generator)
    D = storch.construct_class_by_name(**config.discriminator)
    return G, D

def build_optimizers(G_params: Iterable, D_params: Iterable, config: DictConfig):
    optim_G = storch.construct_class_by_name(G_params, **config.generator)
    optim_D = storch.construct_class_by_name(D_params, **config.discriminator)
    return optim_G, optim_D
