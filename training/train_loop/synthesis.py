
import copy

import storch
import torch
from storch import hydra_utils, loss
from storch.path import Folder, Path
from storch.status import Status
from storch.torchops import freeze, optimizer_step, update_ema
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image

from training.train_loop import utils


def setup_folder(config, child_folders: dict):
    folder = Folder(Path(config.ckpt_folder) / config.name / storch.get_now_string())
    folder.add_children(**child_folders)
    folder.mkdir()
    hydra_utils.save_hydra_config(config, folder.root / 'config.yaml')
    return folder


def train(config):
    folder = setup_folder(config.config.experiment, dict(model='model', image='image'))
    cfg = config.config
    tcfg = cfg.train

    # env
    device = torch.device(cfg.env.device)
    amp = cfg.env.amp

    # dataset
    dataset = utils.build_dataset(cfg.data)

    # models
    latent_dim = cfg.model.generator.latent_dim
    G, D = utils.build_models(cfg.model)
    if tcfg.ema:
        G_ema = copy.deepcopy(G)
        G_ema.to(device)
        freeze(G_ema)
    G.to(device)
    D.to(device)

    # optimizers
    optim_G, optim_D = utils.build_optimizers(G.parameters(), D.parameters(), tcfg.optimizer)

    # criterion
    adv_fn = loss.NonSaturatingLoss()
    gp_fn = loss.r1_regularizer()

    # status
    status = Status(len(dataset)*tcfg.epochs, False, folder.root / cfg.experiment.log_file,
        cfg.experiment.log_interval, cfg.experiment.name)
    status.log_stuff(config, G, optim_G, D, optim_D, dataset)

    scaler = GradScaler() if amp else None
    const_z = torch.randn((tcfg.test_sample, latent_dim), device=device)

    epochs = 0
    while not status.is_end():
        for real in dataset:
            real = real.to(device)
            z = torch.randn(real.size(0), latent_dim, device=device)

            with autocast(amp):
                # G forward
                fake = G(z)

                # D forward (SG)
                real_logits = D(real)
                fake_logits = D(fake.detach())

                # loss
                adv_loss = adv_fn.d_loss(real_logits, fake_logits)
                gp_loss = 0
                D_loss = adv_loss + gp_loss

            optimizer_step(D_loss, optim_D, scaler, zero_grad=True, set_to_none=True)

            with autocast(amp):
                # D forward
                fake_logits = D(fake)

                # loss
                G_loss = adv_fn.g_loss(fake_logits)

            optimizer_step(G_loss, optim_G, scaler, zero_grad=True, set_to_none=True, update_scaler=True)
            if tcfg.ema:
                update_ema(G, G_ema, tcfg.ema_decay, copy_buffers=True)

            if tcfg.running > 0 and status.batches_done % tcfg.running == 0:
                save_image(fake, folder.root / f'running.jpg', normalize=True, value_range=(-1, 1))
            if status.batches_done % tcfg.save == 0:
                kbatches = status.get_kbatches()
                with torch.no_grad():
                    if tcfg.ema:
                        images = G_ema(const_z)
                        state_dict = G_ema.state_dict()
                    else:
                        G.eval()
                        images = G(images)
                        state_dict = G.state_dict()
                        G.train()
                save_image(images, folder.image / f'{kbatches}.jpg', normalize=True, value_range=(-1, 1))
                torch.save(state_dict, folder.model / f'{kbatches}.jpg')

            status.update(**{
                'Loss/G': G_loss.item(),
                'Loss/D': D_loss.item()
            })

            if status.is_end():
                break

        epochs += 1
        if tcfg.epoch_save is not None and epochs % tcfg.epoch_save == 0:
            kbatches = status.get_kbatches()
            state_dict = G_ema.state_dict() if tcfg.ema else G.state_dict()
            torch.save(state_dict, folder.model / f'epochs{epochs}_{kbatches}.jpg')

    status.plot(folder.root / 'progress')
