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
from pytorch_lightning.callbacks import ModelCheckpoint
from recipe.logger import setup_wandb, get_git_hash, find_existing_checkpoint
from recipe.utils import check_model_dataset_consistency
import logging
import os
import getpass
import tempfile
import torch
import shutil


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
    
    # Read training mode from Hydra config
    train_base = config.train_base  # Train base model without adapter?

    # Check for direct checkpoint path 
    is_direct_ckpt = False
    resume_ckpt = None
    if hasattr(config, 'ckpt_path') and config.ckpt_path:
        # User has specified a direct path to a checkpoint
        if os.path.isfile(config.ckpt_path) and config.ckpt_path.endswith('.ckpt'):
            resume_ckpt = config.ckpt_path
            is_direct_ckpt = True  # Flag this as a direct checkpoint
            logger.info(f"Using specified checkpoint: {resume_ckpt}")
        else:
            logger.warning(f"Specified checkpoint path {config.ckpt_path} is not valid, falling back to resume logic")
    
    # Fall back to resume logic if no direct checkpoint is specified or the specified path is invalid
    if resume_ckpt is None and config.resume:
        if isinstance(config.resume, str):
            run_path = config.resume
            assert os.path.exists(run_path), f"resume path {run_path} does not exist"
        else:
            run_path = os.getcwd()
        resume_ckpt = find_existing_checkpoint(run_path, verbose=True) if not train_base else None
    
    # Setup Weights and Biases with the direct checkpoint flag
    wandb_logger = setup_wandb(config, log, git_hash, resume_ckpt, is_direct_ckpt)
    #wandb_logger = None

    datamodule = instantiate(config.datamodule)
    
    # Load model without checkpoint initially
    model = instantiate(
        config.model,
        config_optim=config.optim,
        eval_fn=getattr(datamodule, "eval_fn", None),
        tokenizer=datamodule.tokenizer,
        train_base=train_base,
        checkpoint_path=None,  # Don't load checkpoint here
    )

    check_model_dataset_consistency(model, datamodule)

    # Add after model creation but before training starts
    print("\n===== Checking trainable parameters =====")
    trainable_params = 0
    all_param_count = 0
    planner_param_count = 0
    adapter_param_count = 0

    for name, param in model.named_parameters():
        all_param_count += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"Trainable: {name} - {param.shape}")
            
            if "planner" in name:
                planner_param_count += param.numel()
            if "adapter" in name:
                adapter_param_count += param.numel()

    print(f"Total parameters: {all_param_count:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Planner parameters: {planner_param_count:,}")
    print(f"Adapter parameters: {adapter_param_count:,}")

    # Check if the planner and adapter are submodules
    print("\nSubmodules of model:")
    for name, module in model.named_children():
        print(f"Submodule: {name}")
    print("=====================================\n")

    # Define ModelCheckpoint Callback to Save Checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.logs_dir,
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        save_last=True,
    )

    # print("Config: ", config.trainer)
    config.trainer.accelerator = config.accelerator
    trainer = Trainer(
        **config.trainer,
        logger=wandb_logger,
        #callbacks=getattr(model, "callbacks", None),
        #accelerator=None,
        callbacks=[checkpoint_callback],  # Ensure checkpoints are saved
        strategy="auto" if config.gpus <= 1 else "ddp_find_unused_parameters_true",
        fast_dev_run=16 if config.debug else False,
        profiler="simple" if config.debug else None,
        detect_anomaly=config.debug,
        deterministic=True,
        gradient_clip_val=config.optim.grad_clip,
        enable_progress_bar=not config.use_wandb,
    )
    
    # If we have a checkpoint, load it manually with strict=False before training
    if resume_ckpt:
        print(f"Loading checkpoint with strict=False: {resume_ckpt}")
        checkpoint = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        # Load only matching keys, skip missing ones
        result = model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"Ignored {len(result.missing_keys)} missing keys")
        # Set resume_ckpt to None to prevent PyTorch Lightning from loading it again
        resume_ckpt = None
    
    # Start training
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)

    # Print where model checkpoint is saved
    print(f"Model checkpoint saved in {config.logs_dir}")

    if wandb_logger:
        wandb_logger.experiment.finish()


if __name__ == "__main__":
    # region Debugging
    # print("Checking supported GPUs")
    # print(torch.cuda.is_available())
    # print(torch.__version__)
    # print(torch.version.cuda)
    # print(torch.backends.cudnn.version())
    #endregion
    # user = getpass.getuser()
    # print("git hash: ", git_hash)
    # TODO change base_dir: a snapshot of the code will be made when you run it,
    # such that a slurm requeue (via hydra/submitit) can pick up from the code snapshot.
    # therefore, make sure that this directory is visible from a shared location within your slurm cluster.
    base_dir = f"/home/fmai/checkpoint/" 
    os.makedirs(base_dir, exist_ok=True)
    snapshot_dir = tempfile.mkdtemp(prefix=base_dir)
    print("Snapshot dir is: ", snapshot_dir)
    root = os.getcwd()
    print("Root is: ", root)
    
    # Copy the current directory to the snapshot directory
    shutil.copytree(root, snapshot_dir, dirs_exist_ok=True)
    
    # Change the working directory to the snapshot directory
    os.chdir(snapshot_dir)
    
    main()

# region Linux Implementation
# if __name__ == "__main__":
#     user = getpass.getuser()
#     print("git hash: ", git_hash)
#     # TODO change base_dir: a snapshot of the code will be made when you run it,
#     # such that a slurm requeue (via hydra/submitit) can pick up from the code snapshot.
#     # therefore, make sure that this directory is visible from a shared location within your slurm cluster.
#     base_dir = f"/checkpoint/{user}/snapshots" 
#     os.makedirs(base_dir, exist_ok=True)
#     snapshot_dir = tempfile.mkdtemp(prefix=base_dir)
#     print("Snapshot dir is: ", snapshot_dir)
#     root = os.getcwd()
#     print("Root is: ", root)
#     with RsyncSnapshot(snapshot_dir=snapshot_dir, root_dir=root):
#         main()
# endregion