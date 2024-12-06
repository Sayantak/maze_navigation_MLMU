"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# %%
import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from recipe.utils import Output
from loguru import logger


@dataclass
class PASTConfig:
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    bias: bool = True
    train_mode: str = (
        "ar"  # 'ar', 'permutations', 'all_but_last', 'front-back', 'absorbing', 'multi-forward', float
    )
    tie_lmhead: bool = True
    attn_sink: bool = False
    # THESE ARE MOSTLY DEV PARAMTERS
    perm_chunk_size: int = 1  # chunk size for permutation training mode
    stack_enc_dec: bool = True  # connect only last layer of encoder to decoder layers
    scale_loss: bool = False  # scale loss by rate of masking


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config, rotary, self_attention):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = Attention(config, rotary, self_attention)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask=None, mems=None, mems_mask=None):
        x = x + self.attn(
            self.ln_1(x), attn_mask=attn_mask, mems=mems, mems_mask=mems_mask
        )
        x = x + self.mlp(self.ln_2(x))
        return x


class RotaryPositionalEmbeddings(nn.Module):
    """
    Rotary encoding transforms pairs of features by rotating in the 2D plane.
    That is, it organizes the $d$ features as $\frac{d}{2}$ pairs.
    Each pair can be considered a coordinate in a 2D plane, and the encoding will rotate it
    by an angle depending on the position of the token.
    """

    def __init__(self, d: int, base: int = 10_000):
        """
        * `d` is the number of features $d$ to be rotated
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()

        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        """
        Cache $\cos$ and $\sin$ values for the position indexes inferred
        from the input tensor `x` which has shape `[batch_size, n_heads, seq_len, d]`
        """
        # Return if cache is already built
        if self.cos_cached is not None and x.shape[-2] <= self.cos_cached.shape[0]:
            return

        # Get sequence length
        seq_len = x.shape[-2]

        # $\Theta = {\theta_i = 10000^{-\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.d, 2, device=x.device).float() / self.d)
        )

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=x.device).float()

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Cache them
        self.cos_cached = idx_theta2.cos()
        self.sin_cached = idx_theta2.sin()

    def _neg_half(self, x: torch.Tensor):
        # $\frac{d}{2}$
        d_2 = self.d // 2

        # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.Tensor):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[batch_size, n_heads, seq_len, d]`
        """
        # Cache $\cos$ and $\sin$ values
        self._build_cache(x)

        # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
        x_rope, x_pass = x[..., : self.d], x[..., self.d :]

        # Calculate
        # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * self.cos_cached[: x.shape[-2]]) + (
            neg_half_x * self.sin_cached[: x.shape[-2]]
        )

        return torch.cat((x_rope, x_pass), dim=-1)


class Attention(nn.Module):

    def __init__(self, config, rotary=None, self_attention=True):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.attn_sink = config.attn_sink
        self.rotary = rotary
        self.self_attention = self_attention
        # key, query, value projections for all heads, but in a batch
        if self_attention:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        else:
            self.c_attn = nn.Linear(config.n_embd, 1 * config.n_embd, bias=config.bias)
            self.c_attn_mem = nn.Linear(
                config.n_embd, 2 * config.n_embd, bias=config.bias
            )

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, attn_mask=None, mems=None, mems_mask=None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        if self.self_attention:
            q, k, v = self.c_attn(x).split(C, dim=2)
        else:
            # (B, T', C)
            assert mems.size(0) == B, f"expected mems to have same batch size {B} as x"
            mems_mask = mems_mask.expand(B, 1, T, -1)

            q = self.c_attn(x)
            k, v = self.c_attn_mem(mems).split(C, dim=2)
            attn_mask = mems_mask

        # (B, nh, T, hs)
        k = k.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        # (B, nh, T', hs)
        q = q.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        q, k = self.rotary(q), self.rotary(k)

        # manual attention
        if self.attn_sink:
            w = torch.einsum("bhtd,bhsd->bhts", q, k)
            w = w * (int(C // self.n_head) ** -0.5)
            w = w.masked_fill(~attn_mask, float("-inf"))
            # add empty attention sink token
            w = torch.cat(
                [w, torch.zeros(B, self.n_head, T, 1, device=w.device)], dim=-1
            )
            w = torch.nn.functional.softmax(w, dim=-1)
            w = w[:, :, :, :-1]  # remove empty attention sink token
            y = torch.einsum("bhts,bhsd->bhtd", w, v)
        else:
            # (B, nh, T, T)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout
            )

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Transformer(torch.nn.Module):
    def __init__(self, config, rotary=None, self_attention=True):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [Block(config, rotary, self_attention) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.stack_enc_dec = config.stack_enc_dec

    def forward(self, x, mask=None, mems_list=None, mems_mask=None):
        new_mems = []
        mems_list = mems_list or [None] * len(self.layers)
        for layer, mems in zip(self.layers, mems_list):
            x = layer(x, mask, mems, mems_mask)
            if not self.stack_enc_dec:
                new_mems.append(x)
        x = self.ln_f(x)
        if self.stack_enc_dec:
            new_mems = [x] * len(self.layers)
        return x, new_mems


class PAST(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rotary = RotaryPositionalEmbeddings(config.n_embd // config.n_head, 10_000)
        self.enc = Transformer(config, rotary=self.rotary, self_attention=True)
        self.dec = Transformer(config, rotary=self.rotary, self_attention=False)
        self.wte = torch.nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = torch.nn.Embedding(1, config.n_embd)
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size)
        if config.tie_lmhead:
            self.wte.weight = self.lm_head.weight

    def forward(
        self,
        input_ids,
        attention_mask=None,
        enc_mask=None,
        mode=None,
        prediction_mask=None,
        **kwargs,
    ):
        """
        Forward pass of PAST.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor, optional): Usual attention mask of shape (batch_size, 1, 1, sequence_length).
            enc_mask (torch.Tensor, optional): Encoder mask of shape (batch_size, 1, sequence_length, sequence_length).
                This is generally useful if you want to implement a specific schedule for the absorbing diffusion loss.
            mode (str, optional): Mode of operation, can be "ar" (autoregressive), "permutations", "all_but_last", "absorbing", or "front-back"
            prediction_mask (torch.Tensor, optional): Prediction mask of shape (batch_size, 1, 1, sequence_length).
            This is the mask for elements to use in the loss calculation. This is useful for tasks (like maze navigation) that use
                a prefix that is not useful to make predictions on because it is entirely random.

        Returns:
            Output: A named tuple containing the following fields:
                - logits (torch.Tensor): Logits of shape (batch_size, sequence_length, vocab_size).
                - loss (torch.Tensor): Loss value.
        """
        mode = mode if mode is not None else self.config.train_mode
        B, T, device = input_ids.size(0), input_ids.size(1), input_ids.device

        # This is mostly used for padding mask purposes
        if attention_mask is not None:
            attention_mask = attention_mask.float() > 0
        else:
            attention_mask = torch.ones(B, 1, 1, T, dtype=torch.bool, device=device)
        attention_mask = attention_mask.view(B, 1, 1, T)
        if prediction_mask is None:
            prediction_mask = attention_mask
        prediction_mask = (prediction_mask.float() > 0).view(B, 1, 1, T)

        if enc_mask is not None:
            model_pred_mask = ~enc_mask[:, :, :1, :].bool()
        else:
            enc_mask, model_pred_mask = self.get_mask(
                prediction_mask, mode, device=device
            )

        enc_mask = enc_mask & attention_mask
        prediction_mask = prediction_mask & model_pred_mask & attention_mask
        # # Pass through the model
        pos_emb = self.wpe(torch.zeros(T, device=device, dtype=torch.long))
        pos_emb = pos_emb.view(1, T, -1).expand(B, T, -1)
        tok_emb = self.wte(input_ids)
        _, mems_list = self.enc(tok_emb, enc_mask)
        x, _ = self.dec(pos_emb, None, mems_list, enc_mask)
        logits = self.lm_head(x)
        loss = self.get_loss(logits, input_ids, prediction_mask)
        return Output(logits=logits, loss=loss)

    def get_loss(
        self,
        logits: torch.FloatTensor,
        input_ids: torch.IntTensor,
        prediction_mask: torch.BoolTensor,
    ):
        prediction_mask = prediction_mask.view(input_ids.shape).float()
        logits = logits.view(-1, logits.size(-1))
        tgts = input_ids.view(-1)
        loss = torch.nn.functional.cross_entropy(logits, tgts, reduction="none")
        loss = loss.view(input_ids.shape) * prediction_mask
        if loss.view(-1).isnan().any():
            logger.error("NaNs in loss!")
            logger.error(
                torch.arange(tgts.shape[0], device=loss.device)[loss.view(-1).isnan()]
            )
        if self.config.scale_loss:
            # depends on sequence context, averaged over the batch
            masked = prediction_mask.sum(dim=-1, keepdim=True) + 1e-5
            return (loss / masked).sum(1).mean(0)
        else:
            # averaged over num tokens predicted this round
            return loss.sum() / prediction_mask.sum()

    def get_mask(self, need_to_pred, mode, device):
        need_to_pred = need_to_pred.bool()
        B, T = need_to_pred.size(0), need_to_pred.size(-1)
        if mode == "ar":
            enc_mask = torch.ones(T, T, device=device).tril(-1).bool()
            enc_mask = enc_mask.view(1, 1, T, T).expand(B, 1, T, T)
            enc_mask[:,:,:,0] = True # HACK!! avoid nans due to empty ctx, does not matter unless we are predicting from nothing
            # predict everything because we have a causal mask
            prediction_mask = torch.ones(1, 1, 1, T, dtype=bool, device=device).expand(
                B, 1, 1, T
            )
        elif mode == "permutations":
            # permutations are generated in chunks e.g.,
            # [3,4,5,0,1,2] is a chunked permutation of size 3 on range(6)
            # we can also choose to keep the first few tokens unpermuted and only permute
            # the tail end of the sequence. This is determined by "need_to_pred"
            # e.g. [0, 1, 2, 3, 5,4] is a permutation that keeps [:4] in order.
            perms = [
                torch.cat(
                    [
                        torch.arange(start),
                        start
                        + self._get_permuted_chunks(T-start, self.config.perm_chunk_size),
                    ]
                )
                for start in (need_to_pred.float().view(B, T)).argmax(-1).tolist()
            ]
            enc_mask = torch.stack(
                [
                    torch.ones(T, T, device=device).tril(-1).bool()[perm][:, perm]
                    for perm in perms
                ]
            )
            enc_mask = enc_mask.view(B, 1, T, T)
            enc_mask[:,:,:,0] = True # HACK!! avoid nans due to empty ctx, does not matter unless we are predicting from nothing
            # predict everything because we have a causal mask
            prediction_mask = torch.ones(1, 1, 1, T, dtype=bool, device=device).expand(
                B, 1, 1, T
            )
        elif mode == "all_but_last":
            # acts as all but last for ar generation after training a diffuse model
            enc_mask = torch.ones(B, 1, T, T, dtype=torch.bool, device=device)
            enc_mask[:, :, :, -1] = False
            prediction_mask = ~enc_mask[:, :, :1, :]
        elif mode == "front-back":
            # randomly select an ordered or reversed sequence
            perms = [
                torch.arange(T) if torch.rand(1) > 0.5 else torch.arange(T - 1, -1, -1)
                for _ in range(B)
            ]
            enc_mask = torch.stack(
                [
                    torch.ones(T, T, device=device).tril(-1).bool()[perm][:, perm]
                    for perm in perms
                ]
            )
            enc_mask = enc_mask.view(B, 1, T, T)
            # predict everything because we have a causal mask
            prediction_mask = torch.ones(1, 1, 1, T, dtype=bool, device=device).expand(
                B, 1, 1, T
            )
        elif mode == "multi-forward":
            # like mlmu, but uniformly draw a cutoff point and make the mask true before and false after
            # how many do i need to pred?
            num_to_pred = need_to_pred.sum(-1).flatten()
            # choose a number between 1 and num_to_pred
            rands = torch.rand(num_to_pred.shape, device=num_to_pred.device)
            cutoff = torch.floor(rands * num_to_pred).long() + 1
            cutoff = cutoff.view(B, 1, 1, 1)
            cum_counts = torch.cumsum(need_to_pred, dim=-1)
            enc_mask = (cum_counts > cutoff) | ~need_to_pred
            prediction_mask = ~enc_mask[:, :, :1, :]

        elif mode == "absorbing":
            enc_mask = self._get_diffusion_masks(B, T, device=device) | ~need_to_pred
            prediction_mask = ~enc_mask[:, :, :1, :]  # predict only invisible things
            # B, 1, 1, T
        elif isinstance(mode, float):
            enc_mask = (
                self._get_diffusion_masks(B, T, rate=mode, device=device)
                | ~need_to_pred
            )
            prediction_mask = ~enc_mask[:, :, :1, :]  # B, 1, 1, T
        else:
            modes = ["ar", "permutations", "all_but_last", "front-back", "absorbing", "multi-forward"]
            raise ValueError(f"mode must be one of {modes} or float, got {mode}")
        return enc_mask, prediction_mask

    def _get_diffusion_masks(self, bsz, seq, rate=None, device=None):
        # Generate a 2D tensor of random values between 0 and 1
        uniform_tensor = torch.rand(bsz, seq, device=device)

        # Generate a 2D tensor of probabilities following a uniform distribution
        prob_tensor = torch.rand(bsz, 1, device=device) if rate is None else rate

        # Generate the boolean array based on the probabilities
        mask_array = prob_tensor > uniform_tensor
        mask_array[:, 0] = True  # HACK avoid nans due to empty, gotta do something about that
        return mask_array.unsqueeze(1).unsqueeze(1).expand(bsz, 1, seq, seq)

    def _get_permuted_chunks(self, n, chunk_size):
        a = torch.randperm((n + chunk_size - 1) // chunk_size) * chunk_size
        a = (a.view(-1, 1).repeat(1, chunk_size) + torch.arange(chunk_size)).view(-1)
        a = a[a < n]
        return (a + torch.randint(chunk_size, (1,))) % n
