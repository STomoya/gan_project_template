
from storch import save_exec_status
from storch.hydra_utils import get_hydra_config

from training import train_loop

if __name__=='__main__':
    config = get_hydra_config('config', 'config.yaml')
    train_fn = getattr(train_loop, config.config.train_loop).train
    save_exec_status(config.config.run.exec_status)(train_fn)(config)
