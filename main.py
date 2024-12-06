"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import hydra
from submitit.helpers import RsyncSnapshot
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from pytorch_lightning import Trainer, seed_everything
from recipe.logger import setup_wandb, get_git_hash, find_existing_checkpoint
from recipe.utils import check_model_dataset_consistency
import logging
import os
import getpass
import tempfile


log = logging.getLogger(__name__)
git_hash = get_git_hash()

# import torch
# torch.set_float32_matmul_precision('medium')

@hydra.main(
    version_base="1.2",
    config_path="configs",
    config_name="train_defaults.yaml",
)
def main(config: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(config))
    logger.info(f"saving logs, configs, and model checkpoints to {config.logs_dir}")
    seed_everything(config.seed, workers=True)

    if config.resume:
        if isinstance(config.resume, str):
            run_path = config.resume
            assert os.path.exists(run_path), f"resume path {run_path} does not exist"
        else:
            run_path = os.getcwd()
        resume_ckpt = find_existing_checkpoint(run_path, verbose=True)
    else:
        resume_ckpt = None
    wandb_logger = setup_wandb(config, log, git_hash, resume_ckpt)

    datamodule = instantiate(config.datamodule)
    # only load the model from a checkpoint
    # meant to be used for a model that finished training

    if isinstance(config.resume, str):
        model_class = get_class(config.model._target_)
        model = model_class.load_from_checkpoint(
            resume_ckpt,
            config_optim=config.optim,
            eval_fn=getattr(datamodule, "eval_fn", None),
            tokenizer=datamodule.tokenizer,
        )
        # resume_ckpt = None  # this avoids restarting from the same optimizer state
    else:
        model = instantiate(
            config.model,
            config_optim=config.optim,
            eval_fn=getattr(datamodule, "eval_fn", None),
            tokenizer=datamodule.tokenizer,
        )

    check_model_dataset_consistency(model, datamodule)
    trainer = Trainer(
        **config.trainer,
        logger=wandb_logger,
        callbacks=getattr(model, "callbacks", None),
        strategy="auto" if config.gpus <= 1 else "ddp_find_unused_parameters_true",
        fast_dev_run=16 if config.debug else False,
        profiler="simple" if config.debug else None,
        detect_anomaly=config.debug,
        deterministic=True,
        gradient_clip_val=config.optim.grad_clip,
        enable_progress_bar=not config.use_wandb,
    )
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)
    if wandb_logger:
        wandb_logger.experiment.finish()


if __name__ == "__main__":
    user = getpass.getuser()
    print("git hash: ", git_hash)
    # TODO change base_dir: a snapshot of the code will be made when you run it,
    # such that a slurm requeue (via hydra/submitit) can pick up from the code snapshot.
    # therefore, make sure that this directory is visible from a shared location within your slurm cluster.
    base_dir = f"/checkpoint/{user}/snapshots" 
    os.makedirs(base_dir, exist_ok=True)
    snapshot_dir = tempfile.mkdtemp(prefix=base_dir)
    print("Snapshot dir is: ", snapshot_dir)
    root = os.getcwd()
    print("Root is: ", root)
    with RsyncSnapshot(snapshot_dir=snapshot_dir, root_dir=root):
        main()
