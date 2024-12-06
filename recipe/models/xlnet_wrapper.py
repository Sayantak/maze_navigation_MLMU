"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from transformers import XLNetLMHeadModel


class MaskPrep(torch.nn.Module):
    def __init__(self, batch_size, seq_length, device):
        super().__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.device = device
        # 0 is visible and 1 is masked
        self.register_buffer("triu", torch.ones(seq_length, seq_length).triu())
        self.register_buffer("eye", torch.eye(seq_length))

    def prepare_perm(self, mode=True, batch_size=None, seq_length=None):
        bsz = self.batch_size if batch_size is None else batch_size
        seq = self.seq_length if seq_length is None else seq_length
        if mode == "permutations":
            perm = torch.randperm(seq)
        elif mode == "hybrid":
            perm = torch.randperm(seq) if torch.rand(1) > 0.5 else torch.arange(seq)
        else:
            perm = torch.arange(seq)
        # g is the positional attention mask
        g_mask = self.triu[:seq, :seq][perm][:, perm]
        g_mask = g_mask.unsqueeze(0).expand(bsz, -1, -1)
        target_mapping = self.eye[:seq, :seq][perm][:, perm]
        target_mapping = target_mapping.unsqueeze(0).expand(bsz, -1, -1)
        return g_mask, target_mapping

    def __repr__(self):
        return f"MaskPrep(seq_length={self.seq_length})"


class XLNetCustom(XLNetLMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        model_device = next(self.parameters()).device
        self.mask_prep = MaskPrep(1, config.n_positions, model_device)
        self.register_buffer("mems", torch.zeros(1, 1, self.config.d_model))
        self.get_mems = (
            lambda x: (self.mems.expand(-1, x.size(0), -1),) * self.config.n_layer
        )

    def to(self, *args, **kwargs):
        ret_value = super().to(*args, **kwargs)
        self.mask_prep.device = next(self.parameters()).device
        return ret_value

    def forward(
        self,
        input_ids,
        labels=None,
        mode="ar",
        perm_mask=None,
        target_mapping=None,
        mems=None,
        **kwargs,
    ):
        assert mode in [
            "ar",
            "permutations",
            "hybrid",
        ], f"mode {mode} unrecognized, must be either 'ar' or 'permutations'"
        bsz, seq = input_ids.shape
        perm_mask_, target_mapping_ = self.mask_prep.prepare_perm(mode, bsz, seq)
        perm_mask = perm_mask if perm_mask is not None else perm_mask_
        target_mapping = (
            target_mapping if target_mapping is not None else target_mapping_
        )
        mems = self.get_mems(input_ids) if mems is None else mems
        out = super().forward(
            input_ids,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            labels=labels or input_ids,
            **kwargs,
        )
        return out

    def generate(
        self, input_ids, do_sample=True, temperature=1.0, top_k=None, max_new_tokens=100
    ):
        max_new_tokens = min(
            max_new_tokens, self.config.n_positions - input_ids.size(-1)
        )
        for _ in range(max_new_tokens):
            input_ids = torch.cat(
                [input_ids, torch.zeros(input_ids.size(0), 1).to(input_ids)], dim=-1
            )
            # only need one token when generating
            target_mapping = torch.zeros_like(input_ids).unsqueeze(1).float()
            target_mapping[:, :, -1] = 1.0
            out = self(input_ids, mode="ar", target_mapping=target_mapping)
            logits = out.logits[:, -1]  # bsz, vocab
            if do_sample:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(logits, dim=-1)
            input_ids[:, -1] = next_token
        return input_ids
