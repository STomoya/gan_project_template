
import storch
from storch import hydra_utils
from storch.path import Folder, Path


def setup_folder(config, child_folders: dict):
    folder = Folder(Path(config.ckpt_folder) / config.name / storch.get_now_string())
    folder.add_children(**child_folders)
    folder.mkdir()
    return folder


def train(config):

    folder = setup_folder(config.config.experiment, dict(model='model', image='image'))
    hydra_utils.save_hydra_config(config, folder.root / 'config.yaml')


def launch():
    config = hydra_utils.get_hydra_config('config', 'config.yaml')
    if config.config.experiment.exec_status_output:
        train_fn = storch.save_exec_status(config.config.experiment.exec_status_output)(train)
    else:
        train_fn = train
    train_fn(config)
