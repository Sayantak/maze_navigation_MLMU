"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import random
import torch
import os
import logging
import pickle
import tqdm

from torch.utils.data import DataLoader
from recipe.datasets.abstractdataset import AbstractDataset
from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from ..tokenizer import BasicTokenizer
from maze_dataset import MazeDatasetConfig, MazeDataset as MazeDatasetBase
from maze_dataset.generation import LatticeMazeGenerators
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode
from ..utils import get_data_root, should_create_file, commit_hash
from ..trace import DictTokenizer, TokenizedDataset, AStarTrace
from pathlib import Path
import math
import datasets as ds
from dataclasses import dataclass
from typing import Tuple, Optional, Iterator
from torch.utils.data import DataLoader, IterableDataset


class AStarTraceIterableDataset(IterableDataset):
    def __init__(
        self,
        name: str,
        num_sequences: Optional[int] = None,
        reasoning_range: Optional[Tuple[int, int]] = None,
        use_test: bool = False,
        load_batch_size: int = 10000,
        plan_only: bool = False,
    ):
        """Constructs an object to load and iterate over training sequences
        stored in MongoDB.

        Args:
            name (str): Dataset name.
            num_sequences (Optional[int], optional): Number of sequences to
                iterate over. If None, iterate over all sequences stored in
                MongoDB. Defaults to None.
            reasoning_range (Optional[Tuple[int, int]], optional): Only
                generate sequences whose `reasoning` sequnece falls into the
                specified range. If None, do not filter sequences. Defaults
                to None.
            shuffle (bool, optional): Randomize iteration order. Defaults
                to False.
            use_test (bool, optional): If True, load from test set, otherwise
                load from training set. Defaults to False.
            load_batch_size (int, optional): Batch size used for bulk-loading
                over network from MongoDB. Defaults to 10000.
            plan_only (bool, optional): If True, only include solution plan
                and exclude execution trace. If False, include execution trace
                and plan. Defaults to False.
        """
        self.use_test = use_test
        self.load_batch_size = load_batch_size
        self.dataset = TokenizedDataset(name)
        self.tokenizer = DictTokenizer(self.dataset.vocabulary)
        self.plan_only = plan_only

        if not use_test and reasoning_range is None:
            self.ids = self.dataset.train_ids
            logging.debug(f"Found {len(self.ids)} train sequence ids.")
        elif not use_test and reasoning_range is not None:
            self.ids = self.dataset.train_ids_within_range(*reasoning_range)
            logging.debug(
                f"Found {len(self.ids)} train sequence ids "
                + f"in range {reasoning_range}."
            )
        elif use_test and reasoning_range is None:
            self.ids = self.dataset.test_ids
            logging.debug(f"Found {len(self.ids)} test sequence ids.")
        elif use_test and reasoning_range is not None:
            self.ids = self.dataset.test_ids_within_range(*reasoning_range)
            logging.debug(
                f"Found {len(self.ids)} test sequence ids "
                + f"in range {reasoning_range}."
            )

        self.ids.sort()
        if num_sequences is not None:
            self.ids = self.ids[:num_sequences]

    def __iter__(self) -> Iterator[AStarTrace]:
        if not self.use_test:
            batch_loader = self.dataset.train_it(self.ids, self.load_batch_size)
        else:
            batch_loader = self.dataset.test_it(self.ids, self.load_batch_size)
        for batch in batch_loader:
            tensor_list = self.tokenizer.tokenize_batch(batch, self.plan_only)
            for trace, b in zip(tensor_list, batch):
                yield trace, b


class AstarMazeDataset(LightningDataModule, AbstractDataset):
    def __init__(
        self,
        grid_n=3,
        n_mazes=4,
        return_prediction_mask=True,
        planner=None,  # Add planner
        seed=42,
        **abstract_kwargs,
    ):
        LightningDataModule.__init__(self)
        AbstractDataset.__init__(self, **abstract_kwargs)
        self.return_prediction_mask = return_prediction_mask
        self.grid_n = grid_n
        self.n_mazes = n_mazes
        self.seed = seed
        self.planner = planner # Store planner reference

        dataroot = get_data_root()
        self.path = os.path.join(dataroot, "astarmaze_dataset")
        self._train_filename = os.path.join(
            self.path, f"train_g{grid_n}-n{n_mazes}_seed{seed}.ds"
        )
        self._val_filename = os.path.join(
            self.path, f"val_g{grid_n}-n{n_mazes}_seed{seed}.ds"
        )
        self._tokenizer_filename = os.path.join(self.path, f"tokenizer_g{grid_n}-n{n_mazes}_seed{seed}.pkl")

        self.prepare_data()
        self.tokenizer = pickle.load(open(self._tokenizer_filename, "rb"))

    def transfer_from_mongodb(self, name: str, num_sequences: Optional[int] = None):
        if num_sequences is not None:
            num_sequences_val = math.ceil(num_sequences * 0.02)
        else:
            num_sequences_val = None
        train_ds = AStarTraceIterableDataset(name, num_sequences=num_sequences, plan_only=True, use_test=False)
        val_ds = AStarTraceIterableDataset(name, num_sequences=num_sequences_val, plan_only=True, use_test=True)

        for ads, filename in [(train_ds, self._train_filename), (val_ds, self._val_filename)]:
            text = []
            ids = []
            for trace, tokens in tqdm.tqdm(ads):
                path_start_id = ads.tokenizer.encode(["<PATH_START>"])
                text.append(" ".join(tokens.prompt) + " <PATH_START> " + " ".join(tokens.plan))
                ids.append(trace.prompt.tolist() + path_start_id + trace.trace_plan.tolist())

            dataset = ds.Dataset.from_dict({"text": text, "ids": ids})
            dataset.save_to_disk(filename)

        # save also the tokenizer
        with open(self._tokenizer_filename, "wb") as f:
            pickle.dump(train_ds.tokenizer, f)
        commit_hash(self._train_filename, __file__)
        commit_hash(self._val_filename, __file__)

    def prepare_data(self, save: bool = True, regenerate: bool = False):
        train_exists = not should_create_file(self._train_filename, __file__)
        val_exists = not should_create_file(self._val_filename, __file__)
        if train_exists and val_exists and not regenerate:
            return
        else:
            # if we are in ddp mode, fail!
            if os.environ.get("PL_TRAINER_GPUS", "0") > "1":
                raise ValueError("This dataset can not be made in DDP mode, run it locally first")
            print("moving from mongodb")
            assert self.grid_n in (10,20,30), "only 10x10, 20x20, 30x30 grids are supported for A* mazes"
            self.transfer_from_mongodb(f"maze.{self.grid_n}-by-{self.grid_n}-deterministic.simple", self.n_mazes)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self._train_data = ds.load_from_disk(self._train_filename)
            self._val_data = ds.load_from_disk(self._val_filename)
            self.train = self._train_data["ids"]
            # QAs for training are also put in validation to make them easily
            # accessible later
            val = self._val_data["ids"]
            self.val = {
                "val": val,
                "train": self.train[:: len(self.train) // len(val)],
            }

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        # let's make a dataloader for each data split we have
        get_loader = lambda x: DataLoader(
            x,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )
        if isinstance(self.val, dict):
            return [get_loader(v) for v in self.val.values()]
        return get_loader(self.val)

    def collate_fn(self, batch):
        # batch is a list of lists
        batch = [torch.tensor(i) for i in batch]  # convert to tensor
        batch = pad_sequence(
            batch, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attn = torch.ones_like(batch)
        attn = attn.masked_fill(batch == self.tokenizer.pad_token_id, 0)
        batch_dict = {"input_ids": batch, "attention_mask": attn}
        _, prefix_attn = self._eval_get_prefix(batch_dict)
        if self.return_prediction_mask:
            batch_dict["prediction_mask"] = attn - prefix_attn
        # Generate plans using the planner and include them
        if hasattr(self, "planner") and self.planner is not None:
            batch_dict["plans"] = self.planner(batch_dict["input_ids"], batch_dict["attention_mask"])
        return batch_dict

    def eval_fn(
        self, model, batch, dataloader_idx=0, return_samples=False, **model_kwargs
    ):
        """runs some evaluation based on which data_loader. returns a dict containing
        at least the 'metrics' key whose value is a dict of metrics.
        can also return keys for generated text samples"""
        # get accuracies based on the model's output
        # 1. find prefix to condition the model
        # 2. model.generate(based on prefix)
        # 3. check per token accuracy for the suffix
        val_set_name = list(self.val.keys())[dataloader_idx]
        # val_set_name = "val"
        prefix, prefix_attn = self._eval_get_prefix(batch)
        ans = self._eval_get_model_answers(prefix, prefix_attn, model, **model_kwargs)
        attn = batch["attention_mask"]
        suffix_attn = attn.clone()
        # convert to 1 for each token in the suffix
        suffix_attn = suffix_attn.where(prefix_attn == 0, 0)
        acc, per_sample_acc = self._eval_get_acc(ans, batch["input_ids"], suffix_attn)
        full_path_correct = sum([i == 1 for i in per_sample_acc]) / len(per_sample_acc)
        res = {
            "metrics": {
                f"{val_set_name}/acc": acc,
                f"{val_set_name}/full_acc": full_path_correct,
            }
        }
        if return_samples:
            res.update(
                SAVE_generated=self.tokenizer.batch_decode(ans.tolist()),
                SAVE_truth=self.tokenizer.batch_decode(batch["input_ids"].tolist()),
                SAVE_per_sample_acc=per_sample_acc,
                _generated_ids=ans,
                _suffix_attn=suffix_attn,
                _prefix_ids=prefix,
                _prefix_attn=prefix_attn,
                _truth_ids=batch["input_ids"],
                _truth_attn=attn,
            )
        return res

    def _eval_get_prefix(self, x):
        """This function is filtering inputs up to the `<PATH_START>` token and then padding the rest of the input. Example:"""
        # setup
        pad_id = self.tokenizer.pad_token_id
        separator = self.tokenizer.encode(["<PATH_START>"])[0]
        prefix = x["input_ids"].clone()
        attn = x["attention_mask"].clone()
        # filter inputs up to seprator then pad
        up_to_sep_mask = prefix == separator
        # include the separator in the up_to_sep_mask
        up_to_sep_mask = up_to_sep_mask.cumsum(dim=1) > 0
        up_to_sep_mask[prefix == separator] = False  # keep sep in prefix
        # replace suffix with padding
        prefix = prefix.masked_fill(up_to_sep_mask, pad_id)
        # mask attn for suffix
        attn = attn.masked_fill(up_to_sep_mask, 0)
        return prefix, attn

    def _eval_get_model_answers(self, prefix, attn, model, **model_kwargs):
        """This function is generating model answers for a given set of input prefixes
        takes a tensor which contains the prefix and padding which will be replaced by the model answers
        attn ensures the model doesn't attend to the padding tokens (this is important for bi-directional models)
        example:
        prefix = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]])
        attn = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]])
        the model will make predictions for each token at every iteration but will
        only replace the 0s when the iteration index reaches them
        """
        eot_token = -100  # don't need this for now
        ans = prefix.clone()
        attn = attn.clone()
        column = range(prefix.shape[0])
        for idx in range(ans.shape[1]):
            # if the ans is masked replaced with model output
            old_values = ans[column, idx]
            column_attn = attn[column, idx]
            # skip preds if only old values would be chosen anyway
            if not column_attn.all():
                preds = model(
                  input_ids=ans, attention_mask=attn, **model_kwargs
              ).logits.argmax(dim=-1)
                # keep old values if they are not attn masked
                ans[column, idx] = old_values.where(column_attn != 0, preds[column, idx])
            # update attn mask with 1s for each non eot token
            attn[column, idx] = attn[column, idx].masked_fill(
                ans[column, idx] != eot_token, 1
            )

        return ans

    def _eval_get_acc(self, ans, target, attn):
        """This function is calculating the accuracy of the model's answers.
        We only compute accuracy where the attention mask is 1.
        should make attn such that it doesn't contain prefix tokens."""
        # calculate accuracy
        correct = ans == target
        correct = correct.masked_fill(attn == 0, 0)
        per_sample_acc = correct.sum(-1) / attn.sum(-1)
        acc = correct.sum() / attn.sum()
        return acc.item(), per_sample_acc.tolist()


if __name__ == "__main__":
    datamodule = AstarMazeDataset(grid_n=10, n_mazes=100)
    print("Setting up data", datamodule)
    datamodule.prepare_data()
    datamodule.setup()
    print("train example:", datamodule.train[0])
    train_loader = datamodule.train_dataloader()
    val_loaders = datamodule.val_dataloader()
    print("train_loader len", len(train_loader))
    print(
        "val_loader len",
        len(val_loaders),
    )
    batch = next(iter(train_loader))
    print("train_batch:", batch)
    print(
        "train_batch decoded:",
        *datamodule.tokenizer.batch_decode(batch["input_ids"].tolist()),
        sep="\n",
    )
    print("-" * 50)
    print("moving to gpu", datamodule.load_into("cuda"))
    batch = next(iter(train_loader))
    print("example train batch:\n", batch)
    print("decoded:\n", datamodule.tokenizer.batch_decode(batch["input_ids"].tolist()))
    print("-" * 50)
    print("--- Now testing eval ---")
    print("evaluating constant model:")
    # stupid model that always predicts token 2
    from dataclasses import dataclass

    @dataclass
    class out:
        input_ids: torch.Tensor

        @property
        def logits(self):
            shape = [*self.input_ids.shape, 3]
            return torch.ones(*shape) + torch.arange(3)[None, None, :]

    model = lambda input_ids, **kwargs: out(input_ids)
    for val_loader in val_loaders:
      batch = next(iter(val_loader))
      res = datamodule.eval_fn(model, batch, return_samples=True)
      for k, v in res.items():
          if isinstance(v, torch.Tensor) and "attn" not in k:
              v = f"{v} translation: {datamodule.tokenizer.batch_decode(v.tolist())}"
          print(f"{k}: {v}")
    print("done")
