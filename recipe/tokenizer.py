"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Sequence, Union
from multiprocessing import Pool
from sortedcontainers import SortedSet
import torch

class BasicTokenizer:
    def __init__(self, vocab: Sequence[str]):
        vocab = set(vocab)
        vocab.update(["[PAD]", "[MASK]", "[BOS]", "[EOS]", "[UNK]"])
        vocab = SortedSet(vocab)
        self.vocab_size = len(vocab)
        self.s_to_i = {s: i for i, s in enumerate(vocab)}
        self.i_to_s = {i: s for i, s in enumerate(vocab)}
        self.pad_token_id = self.s_to_i["[PAD]"]
        self.mask_token_id = self.s_to_i["[MASK]"]
        self.bos_token_id = self.s_to_i["[BOS]"]
        self.eos_token_id = self.s_to_i["[EOS]"]

    def encode(self, text: str):
        return [self.s_to_i[s] for s in text.split()]

    def decode(self, sequence: Sequence):
        return " ".join([self.i_to_s.get(i, '[UNK]') for i in sequence])

    def batch_encode(self, texts: Sequence[str], n_workers=1):
        if n_workers == 1:
            return [self.encode(text) for text in texts]
        with Pool(min(n_workers, len(texts))) as p:
            res = p.map(self.encode, texts)
        return res

    def batch_decode(self, sequences: Sequence[Sequence], n_workers=1):
        if isinstance(sequences[0], torch.Tensor):
            sequences = [seq.tolist() for seq in sequences]
        if n_workers == 1:
            return [self.decode(seq) for seq in sequences]
        with Pool(min(n_workers, len(sequences))) as p:
            res = p.map(self.decode, sequences)
        return res

    def __call__(self, inputs: Union[Sequence[str], str], n_workers=1):
        if isinstance(inputs, str):
            ids = self.encode(inputs)
            return {"input_ids": ids}
        if isinstance(inputs, Sequence):
            if isinstance(inputs[0], str):
                ids = self.batch_encode(inputs, n_workers)
                return {"input_ids": ids}
        raise ValueError("inputs must be a string or a sequence of strings")
        


        
