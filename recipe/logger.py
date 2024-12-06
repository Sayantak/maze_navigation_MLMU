"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import List, Optional, Tuple, Dict
import logging
from omegaconf import OmegaConf, DictConfig
import yaml
import os
import datetime
import git
from pathlib import Path
from loguru import logger


def get_git_hash() -> Optional[str]:
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return sha
    except:
        print("not able to find git hash")


def log_job_info(log: logging.Logger):
    import submitit

    """Logs info about the job directory and SLURM job id"""
    job_logs_dir = os.getcwd()
    log.info(f"Logging to {job_logs_dir}")
    job_id = "local"

    try:
        job_env = submitit.JobEnvironment()
        job_id = job_env.job_id
    except RuntimeError:
        pass

    log.info(f"job id {job_id}")


def check_internet():
    import http.client as httplib

    conn = httplib.HTTPSConnection("8.8.8.8", timeout=5)
    try:
        conn.request("HEAD", "/")
        return True
    except Exception:
        return False
    finally:
        conn.close()


def log_internet_status() -> str:
    import socket

    have_internet = check_internet()
    if have_internet:
        return "successfully connected to Google"
    time = datetime.datetime.now()
    machine_name = socket.gethostname()
    return f"Could not connect to Google at {time} from {machine_name}"


def setup_wandb(
    config: DictConfig, log: logging.Logger, git_hash: str = "",
    resume_ckpt: Optional[str] = None,
) -> Optional["WandbLogger"]:
    from pytorch_lightning.loggers import WandbLogger
    import wandb

    log_job_info(log)
    # create a name for the run
    config.name = name_from(config)
    if not config.use_wandb:
        return None
    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))
    job_logs_dir = os.getcwd()
    # increase timeout per wandb folks' suggestion
    os.environ["WANDB_INIT_TIMEOUT"] = "60"
    config_dict["job_logs_dir"] = job_logs_dir
    config_dict["git_hash"] = git_hash
    code_dir = os.path.dirname(os.path.dirname(__file__))
    wandb_id = find_run_wandb_id(resume_ckpt)
    get_logger = lambda: WandbLogger(
        config=config_dict,
        settings=wandb.Settings(code_dir=code_dir, start_method="fork"),
        name=config.name,
        id=wandb_id,
        resume="must" if resume_ckpt else None,
        **config.wandb,
    )
    try:
        wandb_logger = get_logger()
    except Exception as e:
        print(f"exception: {e}")
        print(log_internet_status())
        print("starting wandb in offline mode. To sync logs run")
        print(f"wandb sync {job_logs_dir}")
        os.environ["WANDB_MODE"] = "offline"
        wandb_logger = get_logger()
    return wandb_logger


def name_from(config: DictConfig) -> str:
    if name := config.get("name", None):
        return name
    name = []
    name.append(config.model._target_.split(".")[-1].removesuffix("PL"))
    name.append(config.model.get("train_mode", ""))
    name.append(f"{config.datamodule._target_.split('.')[-1].removesuffix('Dataset')}")
    return "_".join(name)


def find_existing_checkpoint(dirpath: str, verbose: bool = False, resume_index=-1) -> Optional[str]:
    """Searches dirpath for an existing model checkpoint.
    If found, returns its path.
    """
    ckpts = list(Path(dirpath).rglob("*.ckpt"))
    # some paths are symlinks and need to be resolved
    resolved = [p.resolve() for p in ckpts]
    # there's a bug when resolving it repeats the path. unclear why
    # keep only the last part of the resolved path
    ckpts = [
        "/".join(str(p).split("/")[:-1] + [str(r).split("/")[-1]])
        for p, r in zip(ckpts, resolved)
    ]
    ckpts = sorted(ckpts, key=os.path.getctime)
    if ckpts:
        ckpt = str(ckpts[resume_index])
        if verbose:
            logger.info(f"Found ckpts: {ckpts}")
            logger.info(f"Resuming from: {ckpt}")
        return ckpt
    return None


def find_run_wandb_id(path: str) -> str:
    if not path:
        return None
    # id is contained at */{ID}/checkpoints/*
    run_id = path.split("/checkpoints/")[0].split("/")[-1]  # a bit hacky
    return run_id
