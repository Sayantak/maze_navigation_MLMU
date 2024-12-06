"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from abc import ABC
from dataclasses import dataclass
from datasets import Dataset
import torch
from transformers import AutoTokenizer


@dataclass
class AbstractDataset(ABC):
    """Abstract class for datasets. Needs to define train and val attributes. Assigns default values to batch_size and num_workers."""

    preload_device: str = "cpu"
    batch_size: int = 32
    num_workers: int = 4

    def load_into(self, device):
        self.preload_device = device
        if isinstance(self.train, torch.Tensor):
            self.train = self.train.to(device, non_blocking=True)
            if isinstance(self.val, torch.Tensor):
                self.val = self.val.to(device, non_blocking=True)
            else:
                self.val = {
                    k: v.to(device, non_blocking=True)
                    for k, v in self.val.items()
                    if isinstance(v, torch.Tensor)
                }
        elif isinstance(self.train, Dataset):
            self.train = self.train.map(
                lambda x: {
                    k: v.to(device, non_blocking=True)
                    for k, v in x.items()
                    if isinstance(v, torch.Tensor)
                }
            )
            self.val = self.val.map(
                lambda x: {
                    k: v.to(device, non_blocking=True)
                    for k, v in x.items()
                    if isinstance(v, torch.Tensor)
                }
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(batch_size={self.batch_size}, num_workers={self.num_workers})"

    def set_tokenizer(self, tokenizer):
        # no need for fast tokenizer because we are not tokenizing on the fly
        if isinstance(tokenizer, str) or tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer or "google-bert/bert-base-uncased", use_fast=False
            )
            if tokenizer.pad_token_id is None:
                # GPT2 has no pad token
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.eos_token_id is None:
                # BERT has no eos token
                tokenizer.eos_token = tokenizer.sep_token
        self.tokenizer = tokenizer
