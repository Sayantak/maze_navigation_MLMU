"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import pandas as pd
import wandb
import os
from omegaconf import OmegaConf
from hydra.utils import get_class
from recipe.logger import find_existing_checkpoint
from loguru import logger


class Runs:
    def __init__(
        self,
        entity: str = "past",
        project: str = "absorbing-state",
        filters=None,
    ):
        self.entity = entity
        self.project = project

        api = wandb.Api()
        self.runs = api.runs(entity + "/" + project, filters=filters)
        if len(self.runs) == 0:
            logger.warning("No runs found!")
            self.df = None
        self.nested_config_keys = ["model", "evaluation_module", "datamodule"]
        self.config_keys = ["name", "job_logs_dir", "model_name"]

        self.df = self.create_df()

    @staticmethod
    def load_model(path=None, run=None, verbose=True, resume_index=-1):
        """
        Load the logging path from a wandb run object.

        Args:
            path (str, optional): The path to the logging directory. If not provided, the `run` argument must be specified.
            run (wandb.run, optional): A wandb run object. If provided, the `job_logs_dir` from the run config will be used as the logging path.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
            resume_index (int, optional): The index of the checkpoint to resume from. Defaults to -1 (latest checkpoint).

        Returns:
            tuple: The loaded model and the loaded config.
        """
        if path is None and run is None:
            raise ValueError("Either 'path' or 'run' must be specified.")
        if run is not None:
            # Get the job logs directory from the run config
            job_logs_dir = run.config.get("job_logs_dir")

            if job_logs_dir is None:
                raise ValueError(
                    "The 'job_logs_dir' key is not present in the run config."
                )

            # Check if the job logs directory exists
            if not os.path.isdir(job_logs_dir):
                raise ValueError(
                    f"The job logs directory '{job_logs_dir}' does not exist."
                )
            path = job_logs_dir
        # assumes all ckpts here have the same config
        config = OmegaConf.load(f"{path}/.hydra/config.yaml")
        model_cls = get_class(config.model._target_)
        ckpt = find_existing_checkpoint(
            path, verbose=verbose, resume_index=resume_index
        )
        if ckpt is None:
            # try the version for local directory structure (if not run with wandb)
            ckpt = find_existing_checkpoint(
                path + '/../lightning_logs/', verbose=verbose, resume_index=resume_index
            )
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in {path}")
        model = model_cls.load_from_checkpoint(ckpt)
        return model, config

    @staticmethod
    def load_dataset(path=None, run=None, update_conf=None):
        from hydra.utils import instantiate

        if path is None and run is None:
            raise ValueError("Either 'path' or 'run' must be specified.")
        if run is not None:
            # Get the job logs directory from the run config
            job_logs_dir = run.config.get("job_logs_dir")
            if job_logs_dir is None:
                raise ValueError(
                    "The 'job_logs_dir' key is not present in the run config."
                )

            # Check if the job logs directory exists
            if not os.path.isdir(job_logs_dir):
                raise ValueError(
                    f"The job logs directory '{job_logs_dir}' does not exist."
                )
            path = job_logs_dir
        config = OmegaConf.load(f"{path}/.hydra/config.yaml")
        if update_conf is not None:
            config = OmegaConf.merge(config, update_conf)
        dataset = instantiate(config.datamodule)
        dataset.prepare_data()
        dataset.setup()
        return dataset

        
    

    def create_df(self):
        """Creates a dataframe of all runs"""
        data = []
        for run in self.runs:
            run_data = dict()
            run_data.update(self.extract_summary(run.summary._json_dict))
            run_data.update(self.extract_config(run.config))
            run_data.update({"tags": run.tags})
            if "sanity_check" in run.tags:
                run_data.update({"is_sanity_check": True})
            else:
                run_data.update({"is_sanity_check": False})
            data.append(run_data)
        runs_df = pd.DataFrame.from_records(data)
        if len(data) == 0:
            logger.warning("Trying to create df... No runs found!")
            return runs_df
        # filter sanity checks
        runs_df = runs_df[~runs_df["is_sanity_check"]]
        return runs_df

    def extract_config(self, config: dict) -> dict:
        data = dict()
        for config_key in self.config_keys:
            if config_key in config:
                data[config_key] = config[config_key]

        # data module keys
        for module in self.nested_config_keys:
            if module not in config:
                continue
            for k in config[module]:
                if k == "_target_":
                    data[module] = config[module]["_target_"]
                else:
                    data[k] = config[module][k]
        if "model" in config:
            data["model"] = config["model"]["_target_"].split(".")[-1]
        else:
            logger.warning("No model found in config")
            data["model"] = None
        return data

    def extract_summary(self, summary: dict) -> dict:
        """Summary containing accuracy and other metrics"""
        data = dict()
        for (
            k,
            v,
        ) in summary.items():
            if k.startswith("_"):
                continue
            # skip class specific top1
            if k.split("_")[-1].isdigit():
                continue
            data[k] = v
        return data
