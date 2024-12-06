"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import random
import torch
from torch.utils.data import DataLoader
from recipe.datasets.abstractdataset import AbstractDataset
from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from ...tokenizer import BasicTokenizer
import re
from ...utils import get_data_root
from .gen_data import generate_and_save
import os


def clean_lines(lines):
    lines = [
        [token for token in re.split(r"([|,=/])", line.strip()) if token not in (",",)]
        for line in lines
        if len(line.strip()) > 0
    ]
    return [" ".join(line) for line in lines]


class StarGraphDataset(LightningDataModule, AbstractDataset):
    def __init__(
        self,
        tokenizer=None,
        return_prediction_mask=False,
        n_train=200_000,
        n_val=20_000,
        degree=2,
        pathlen=5,
        num_nodes=50,
        **abstract_kwargs,
    ):
        LightningDataModule.__init__(self)
        AbstractDataset.__init__(self, **abstract_kwargs)
        self.return_prediction_mask = return_prediction_mask
        dataroot = get_data_root()

        train_path = os.path.join(
            dataroot, f"graphs/deg_{degree}_path_{pathlen}_nodes_{num_nodes}_train_{n_train}.txt"
        )
        test_path = os.path.join(
            dataroot, f"graphs/deg_{degree}_path_{pathlen}_nodes_{num_nodes}_test_{n_val}.txt"
        )

        os.makedirs(os.path.dirname(train_path), exist_ok=True)

        if not os.path.exists(train_path) or not os.path.exists(test_path):
          print(f"generating new data under {train_path}")
          generate_and_save(
              n_train=n_train,
              n_test=n_val,
              degSource=degree,
              pathLen=pathlen,
              numNodes=num_nodes,
              reverse=False,
          )

        with open(train_path) as f:
            train = clean_lines(f.readlines())
        with open(test_path) as f:
            val = clean_lines(f.readlines())

        words = [word for string in train + val for word in string.split()]

        if tokenizer is None:
            self.tokenizer = BasicTokenizer(words + ["[bos]", "[eos]"])
        else:
            self.set_tokenizer(tokenizer)

        # train/val split
        def wrap(sequence_list):
            return [f"[bos] {s} [eos]" for s in sequence_list]

        self._train_data = wrap(train)
        self._val_data = wrap(val)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = self.tokenizer(self._train_data)["input_ids"]
            self.val = {
                "val": self.tokenizer(self._val_data)["input_ids"],
                "train": self.train[:100],
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
        prefix, prefix_attn = self._eval_get_prefix(batch)
        ans = self._eval_get_model_answers(prefix, prefix_attn, model, **model_kwargs)
        attn = batch["attention_mask"]
        suffix_attn = attn.clone()
        # convert to 1 for each token in the suffix
        suffix_attn = suffix_attn.where(prefix_attn == 0, 0)
        acc, per_sample_acc = self._eval_get_acc(ans, batch["input_ids"], suffix_attn)
        acc_exact_match = (torch.tensor(per_sample_acc) == 1).float().mean().item()
        res = {
            "metrics": {
                f"{val_set_name}/acc": acc,
                f"{val_set_name}/em": acc_exact_match,
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
        """This function is filtering inputs up to the `=` token and then padding the rest of the input. Example:"""
        # setup
        pad_id = self.tokenizer.pad_token_id
        separator = self.tokenizer.encode("=")[0]
        prefix = x["input_ids"].clone()
        attn = x["attention_mask"].clone()
        # filter inputs up to question mark then pad
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
    datamodule = StarGraphDataset(
        batch_size=10,
    )
    print("Setting up data", datamodule)
    datamodule.setup()
    print("train example:", datamodule.train[0])
    train_loader = datamodule.train_dataloader()
    val_loaders = datamodule.val_dataloader()
    print("train_loader len", len(train_loader))
    print(
        "val_loaders len (careful here this might be a list of loaders)",
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
    if isinstance(val_loaders, list):
        for i, val_loader in enumerate(val_loaders):
            print(f"val_loader num{i} len", len(val_loader))
            batch = next(iter(val_loader))
            print(f"val_loader num{i} batch:", batch)
            print(
                "decoded:",
                datamodule.tokenizer.batch_decode(batch["input_ids"].tolist()),
            )
    print("-" * 50)
    print("moving to gpu", datamodule.load_into("cuda"))
    batch = next(iter(train_loader))
    print("example train batch:\n", batch)
    print("decoded:\n", datamodule.tokenizer.batch_decode(batch["input_ids"].tolist()))
    keys = list(datamodule.val.keys())
    for i, val_loader in enumerate(val_loaders):
        batch = next(iter(val_loader))
        batch_decoded = datamodule.tokenizer.batch_decode(batch["input_ids"].tolist())
        print(f"example val_loader num{i} with key {keys[i]} batch:\n", batch_decoded)
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
    for i, val_loader in enumerate(val_loaders):
        batch = next(iter(val_loader))
        print(f"- on num{i} with key {keys[i]}")
        res = datamodule.eval_fn(model, batch, dataloader_idx=i, return_samples=True)
        for k, v in res.items():
            if isinstance(v, torch.Tensor) and "attn" not in k:
                v = f"{v} translation: {datamodule.tokenizer.batch_decode(v.tolist())}"
            print(f"{k}: {v}")
    print("done")
