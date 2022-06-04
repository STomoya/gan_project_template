
import storch
from storch import hydra_utils


def train(config):
    pass


def launch():
    config = hydra_utils.get_hydra_config('config', 'config.yaml')
    if config.config.experiment.exec_status_output:
        train_fn = storch.save_exec_status(config.config.experiment.exec_status_output)(train)
    else:
        train_fn = train
    train_fn(config)
