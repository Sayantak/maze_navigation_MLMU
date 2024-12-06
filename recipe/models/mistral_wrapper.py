"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn.functional as F
from transformers import MistralForCausalLM

class MistralCustom(MistralForCausalLM):
    def set_bos_token_id(self, bos_token_id):
        self.bos_token_id = bos_token_id

    def forward(
        self,
        input_ids,
        labels=None,
        mode="ar",
        **kwargs,
    ):
        assert labels is None
        assert mode is None or mode in [
            "ar",
        ], f"mode {mode} unrecognized, must be either 'ar'"

        attn_mask = kwargs.pop("attention_mask", None)
        # in case that exists
        kwargs.pop("_relation", None)
        kwargs.pop("_reference", None)
        if attn_mask is None:
            attn_mask = torch.ones_like(input_ids)
        
        
        input_ids = F.pad(input_ids, (1, -1), value=self.bos_token_id)
        attn_mask = F.pad(attn_mask, (1, -1), value=1)

        out = super().forward(
            input_ids,
            labels=input_ids,
            attention_mask=attn_mask,
            **kwargs,
        )

        return out
