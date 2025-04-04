"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import hydra
# from submitit.helpers import RsyncSnapshot # Keep if using the Linux Rsync version later
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
from recipe.models.planning import TransformerSampleEncoder  # Add this import


log = logging.getLogger(__name__)
git_hash = get_git_hash()

# import torch
# torch.set_float32_matmul_precision('medium')

# Store the original root directory before Hydra potentially changes it
original_root_dir = os.getcwd()

@hydra.main(
    version_base="1.2",
    config_path="configs",
    config_name="train_defaults.yaml",
)
def main(config: DictConfig) -> None:
    # --- Snapshot logic moved here ---
    # Use original_root_dir captured before Hydra's potential chdir
    root = original_root_dir
    print("Original Root is: ", root)

    # Get base_dir from Hydra config
    base_dir = config.get("base_dir", "/tmp/mlmu_snapshots") # Use get with a default
    os.makedirs(base_dir, exist_ok=True)
    # Use a temporary directory within the configured base_dir
    snapshot_dir = tempfile.mkdtemp(prefix=os.path.join(base_dir, "snapshot_"))
    print("Snapshot dir is: ", snapshot_dir)

    try:
        # Copy the current directory (which might have been changed by Hydra's chdir=True)
        # to the snapshot directory. We copy from the *original* root.
        if os.path.exists(root):
             print(f"Copying from {root} to {snapshot_dir}")
             # Using shutil.copytree, ensure target doesn't exist or use dirs_exist_ok=True (Python 3.8+)
             if os.path.exists(snapshot_dir):
                  shutil.rmtree(snapshot_dir) # Remove existing snapshot dir if necessary
             # Copy contents from the original root directory
             shutil.copytree(root, snapshot_dir, dirs_exist_ok=True, symlinks=True)
             # Change the working directory to the snapshot directory
             os.chdir(snapshot_dir)
             print(f"Changed CWD to snapshot dir: {os.getcwd()}")
        else:
             print(f"Error: Source directory {root} not found.")
             return # Exit if source dir doesn't exist

        # --- Original main function logic starts here ---
        logger.info(OmegaConf.to_yaml(config))
        # Logs dir path needs careful handling:
        # If config.logs_dir is relative, it's now relative to snapshot_dir.
        # If it's absolute, it's fine.
        # If you want logs relative to the *original* execution dir, resolve it:
        logs_dir = config.logs_dir
        if not os.path.isabs(logs_dir):
            logs_dir = os.path.join(root, logs_dir)
            print(f"Resolved relative logs_dir to: {logs_dir}")
            # Optionally update config, though ModelCheckpoint might take the absolute path directly
            # OmegaConf.update(config, "logs_dir", logs_dir, merge=True)

        logger.info(f"Saving logs, configs, and model checkpoints to {logs_dir}")
        seed_everything(config.seed, workers=True)

        # Read training mode flags from Hydra config
        train_base = config.train_base
        train_planner = config.train_planner # Read the new flag


        # Check for direct checkpoint path
        is_direct_ckpt = False
        resume_ckpt = None
        if hasattr(config, 'ckpt_path') and config.ckpt_path:
            # Resolve potential relative path from original root
            ckpt_path_abs = config.ckpt_path
            if not os.path.isabs(ckpt_path_abs):
                 ckpt_path_abs = os.path.join(root, ckpt_path_abs)

            if os.path.isfile(ckpt_path_abs) and ckpt_path_abs.endswith('.ckpt'):
                resume_ckpt = ckpt_path_abs
                is_direct_ckpt = True
                logger.info(f"Using specified checkpoint: {resume_ckpt}")
            else:
                logger.warning(f"Specified checkpoint path {config.ckpt_path} (resolved to {ckpt_path_abs}) is not valid, falling back to resume logic")

        # Fall back to resume logic
        if resume_ckpt is None and config.resume:
             run_path_to_search = None
             if isinstance(config.resume, str):
                  # Assume config.resume is relative to original root or absolute
                  run_path_to_search = config.resume
                  if not os.path.isabs(run_path_to_search):
                       run_path_to_search = os.path.join(root, run_path_to_search)
                  assert os.path.exists(run_path_to_search), f"resume path {run_path_to_search} does not exist"
             else:
                  # If resume=True, search in the resolved logs_dir
                  run_path_to_search = logs_dir # Search in the potentially resolved logs dir

             if run_path_to_search:
                 resume_ckpt = find_existing_checkpoint(run_path_to_search, verbose=True) if not train_base else None


        # Setup Weights and Biases
        # Read the top-level use_wandb flag
        use_wandb_flag = config.get("use_wandb", True) # Default to True if not specified
        if use_wandb_flag:
            # Pass the main config, logger, git hash, etc.
            wandb_logger = setup_wandb(config, log, git_hash, resume_ckpt, is_direct_ckpt)
        else:
            wandb_logger = None
            print("Weights & Biases logging disabled (use_wandb=False).")

        datamodule = instantiate(config.datamodule)

        # Load model, passing the training flags
        model = instantiate(
            config.model,
            config_optim=config.optim,
            eval_fn=getattr(datamodule, "eval_fn", None),
            tokenizer=datamodule.tokenizer,
            train_base=train_base,
            train_planner=train_planner,
            checkpoint_path=None,
            # Get planning parameters from planning config
            num_samples=config.planning.num_samples,
            continuation_length=config.planning.continuation_length,
            split_idx_mode=config.planning.split_idx_mode,
            # Add the entire planning configuration to the hparams
            planning=config.planning,
            cut_outputs=config.planning.cut_outputs
        )

        check_model_dataset_consistency(model, datamodule)

        # Print parameter info
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

        # Define ModelCheckpoint Callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=logs_dir, # Use the resolved logs_dir
            filename="{epoch}-{val_loss:.2f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
            save_last=True,
        )

        # Configure Trainer
        # config.trainer.accelerator = config.accelerator # This might be redundant if already in config
        trainer = Trainer(
            **config.trainer,
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            strategy="auto" if config.gpus <= 1 else "ddp_find_unused_parameters_true",
            fast_dev_run=16 if config.debug else False,
            profiler="simple" if config.debug else None,
            detect_anomaly=config.debug,
            deterministic=True,
            gradient_clip_val=config.optim.grad_clip,
            enable_progress_bar=not config.use_wandb,
        )

        # Manual Checkpoint Loading
        if resume_ckpt and not trainer.fast_dev_run: # Don't load checkpoint in fast_dev_run
            print(f"Loading checkpoint with strict=False: {resume_ckpt}")
            checkpoint = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
            result = model.load_state_dict(checkpoint["state_dict"], strict=False)
            print(f"Ignored {len(result.missing_keys)} missing keys")
            
            # Add the embedding initialization code here
            if hasattr(config.planning, 'use_gpt_embeddings') and config.planning.use_gpt_embeddings and train_planner:
                print("\n===== Initializing TransformerSampleEncoder embeddings from GPT2 =====")
                if not (hasattr(model, 'planner') and hasattr(model.planner, 'adapter') and hasattr(model.model, 'transformer')):
                    raise RuntimeError("Required model components not found for embedding initialization. "
                                      "Make sure model has 'planner', 'planner.adapter', and 'model.transformer' attributes.")
                    
                # Access the GPT2 embedding table
                gpt_embeddings = model.model.transformer.wte.weight.detach().clone()
                
                # Find the TransformerSampleEncoder in your planner/adapter
                if not (hasattr(model.planner.adapter, 'encoder') and 
                        isinstance(model.planner.adapter.encoder, TransformerSampleEncoder)):
                    raise RuntimeError("TransformerSampleEncoder not found in model.planner.adapter.encoder. "
                                      "Check your model architecture or disable use_gpt_embeddings.")
                
                encoder = model.planner.adapter.encoder
                
                # Check if the embedding dimensions match
                if encoder.token_embedding.weight.shape != gpt_embeddings.shape:
                    raise ValueError(f"Embedding shapes don't match. Cannot initialize. "
                                   f"GPT2: {gpt_embeddings.shape}, Encoder: {encoder.token_embedding.weight.shape}")
                
                # Only proceed if all checks pass
                encoder.token_embedding.weight.data.copy_(gpt_embeddings)
                print(f"Successfully copied GPT2 embeddings to TransformerSampleEncoder")
                print(f"Embedding shape: {gpt_embeddings.shape}")
                print("=================================================================\n")
            
            # Prevent PL from trying to load it again
            resume_ckpt_for_trainer = None
        else:
            resume_ckpt_for_trainer = None # Ensure it's None if not resuming

        # Start training
        trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt_for_trainer)

        # Print where model checkpoint is saved
        print(f"Model checkpoint saved in {logs_dir}")

        if wandb_logger:
            wandb_logger.experiment.finish()

    finally:
        # --- Cleanup: Change back to original directory ---
        os.chdir(root) # Change back to the original directory
        print(f"Changed CWD back to: {os.getcwd()}")
        # Optionally remove the snapshot directory
        # print(f"Removing snapshot directory: {snapshot_dir}")
        # shutil.rmtree(snapshot_dir)


if __name__ == "__main__":
    # This now calls the @hydra.main decorated function
    main()

# region Linux Implementation (Keep commented out or integrate if needed)
# if __name__ == "__main__":
#     user = getpass.getuser()
#     print("git hash: ", git_hash)
#     base_dir = f"/checkpoint/{user}/snapshots"
#     os.makedirs(base_dir, exist_ok=True)
#     snapshot_dir = tempfile.mkdtemp(prefix=base_dir)
#     print("Snapshot dir is: ", snapshot_dir)
#     root = os.getcwd()
#     print("Root is: ", root)
#     with RsyncSnapshot(snapshot_dir=snapshot_dir, root_dir=root):
#          # Inside here, Hydra needs to be called, or config passed manually
#          # This structure conflicts slightly with the @hydra.main decorator approach
#          # You'd typically use one or the other (Hydra decorator OR manual snapshot + manual config loading)
#          pass # main() # This would need adjustment if using RsyncSnapshot
# endregion