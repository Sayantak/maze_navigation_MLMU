"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import XLNetConfig, BitsAndBytesConfig
from .xlnet_wrapper import XLNetCustom
from .gpt_wrapper import GPT2Custom, GPT2Config
from .past import PASTConfig, PAST
import torch
import math


def nearest_multiple(x):
    if x < 128:
        return 128
    base = 128
    res = base * math.ceil(x / base)
    print(f"updating vocab size {x} to nearest multiple of {base} with {res}")
    return res


class PLModel(LightningModule):
    def __init__(self, eval_fn=None, config_optim=None, **model_kwargs) -> None:
        super().__init__()
        self.config_optim = config_optim
        self.eval_fn = eval_fn
        self.train_mode = model_kwargs.get("train_mode", "ar")
        self.eval_mode = "all_but_last" if self.train_mode in ["absorbing", "multi-forward"] else "ar"
        self.loss_mask = model_kwargs.get("loss_mask", "all")
        self.callbacks = [
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                verbose=True,
                save_on_train_epoch_end=True,
            ),
            ModelCheckpoint(
                monitor="step",
                mode="max",
                save_top_k=5,
                every_n_train_steps=5000,
                verbose=True,
                save_on_train_epoch_end=True,
            ),
            LearningRateMonitor(),
        ]
        self.save_hyperparameters(logger=False, ignore=["eval_fn"])
        # important to load model later
        # avoid saving to logger becase we do that elsewhere

    def forward(self, batch, **model_kwargs):
        out = self.model(
            **batch,
            mode=self.train_mode,
            **model_kwargs,
        )
        return out

    def training_step(self, batch, batch_idx):
        bsz = batch["input_ids"].size(0)
        out = self(batch)
        self.log("train/loss", out.loss, prog_bar=True, batch_size=bsz, sync_dist=True)
        return out.loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        bsz = batch["input_ids"].size(0)
        out = self(batch)
        metric_name = f"val/loss/{dataloader_idx}" if dataloader_idx > 0 else "val/loss"
        self.log(
            metric_name,
            out.loss,
            prog_bar=True,
            batch_size=bsz,
            add_dataloader_idx=False,
            sync_dist=True
        )
        if self.eval_fn is not None:
            eval_dict = self.eval_fn(
                self.model,
                batch,
                dataloader_idx=dataloader_idx,
                return_samples=batch_idx == 0,
                mode=self.eval_mode,
            )
            for k, v in eval_dict["metrics"].items():
                k = f"{k}/{dataloader_idx}"
                prog = "acc" in k
                self.log(k, v, prog_bar=prog, batch_size=bsz, add_dataloader_idx=False, sync_dist=True)
            if hasattr(self.logger, "log_text") and batch_idx == 0:
                filtered = [k for k in eval_dict.keys() if k.startswith("SAVE_")]
                columns = [k.strip("SAVE_") for k in filtered]
                data = [eval_dict[k] for k in filtered]
                data = list(zip(*data))
                self.logger.log_text(
                    f"samples/{dataloader_idx}",
                    data=data,
                    columns=columns,
                )
        return out.loss

    def configure_optimizers(self):
        optim_name = self.config_optim.optimizer
        weight_decay = self.config_optim.weight_decay
        learning_rate = self.config_optim.learning_rate
        betas = self.config_optim.betas
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        if optim_name == "adamw":
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        elif optim_name == "sgd":
            optimizer = torch.optim.SGD(optim_groups, lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer {optim_name}")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.config_optim.warmup_pct,
            anneal_strategy="cos",
            final_div_factor=25,
        )
        # total steps can be set with trainer.max_steps directly
        # but may break when using max_batches instead
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class XLNetPL(PLModel):
    def __init__(
        self, eval_fn=None, config_optim=None, tokenizer=None, **model_kwargs
    ) -> None:
        super().__init__(eval_fn=eval_fn, config_optim=config_optim, **model_kwargs)
        if tokenizer is not None:
            model_kwargs["vocab_size"] = tokenizer.vocab_size
        model_config = XLNetConfig(**model_kwargs)
        model_config.vocab_size = nearest_multiple(model_config.vocab_size)
        self.model = XLNetCustom(model_config)


class PASTPL(PLModel):
    def __init__(
        self, eval_fn=None, config_optim=None, tokenizer=None, **model_kwargs
    ) -> None:
        super().__init__(eval_fn=eval_fn, config_optim=config_optim, **model_kwargs)
        if tokenizer is not None:
            model_kwargs["vocab_size"] = tokenizer.vocab_size
        model_config = PASTConfig(**model_kwargs)
        model_config.vocab_size = nearest_multiple(model_config.vocab_size)
        self.model = PAST(model_config)


class GPT2PL(PLModel):
    def __init__(
        self,
        eval_fn=None,
        config_optim=None,
        tokenizer=None,
        from_pretrained=False,
        **model_kwargs,
    ) -> None:
        super().__init__(eval_fn=eval_fn, config_optim=config_optim, **model_kwargs)
        if from_pretrained:
            print("loading from pretrained, ignoring rest of config.")
            self.model = GPT2Custom.from_pretrained("openai-community/gpt2")
        else:
            if tokenizer is not None:
                model_kwargs["vocab_size"] = tokenizer.vocab_size
            model_kwargs.update(
                vocab_size=nearest_multiple(tokenizer.vocab_size),
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            model_config = GPT2Config(
                **model_kwargs,
            )
            self.model = GPT2Custom(model_config)
            self.model.apply(self.model._init_weights)


class MistralPL(PLModel):
    def __init__(
        self,
        eval_fn=None,
        config_optim=None,
        tokenizer=None,
        lora_r_alpha=256,
        **model_kwargs,
    ) -> None:
        super().__init__(eval_fn=eval_fn, config_optim=config_optim, **model_kwargs)
        from_pretrained = model_kwargs.pop("from_pretrained")
        self.tokenizer = tokenizer
        assert from_pretrained, "Mistral7B only supports loading from pretrained"
        print("loading from pretrained, ignoring rest of config.")
        self.model = None
        self.lora_r_alpha = lora_r_alpha

    def to(self, *args, **kwargs):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from .mistral_wrapper import MistralCustom

        # FIXME big hack to enable DDP training with peft models
        assert self.model is None, "model already loaded"
        bnb_config = BitsAndBytesConfig(
            load_in_8_bit=True,
            # bnb_4bit_use_double_quant=True,
            # bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = MistralCustom.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            quantization_config=bnb_config,
            resume_download=True,
            device_map={"": args[0].index},
        )
        model.set_bos_token_id(self.tokenizer.bos_token_id)

        lora_config = LoraConfig(
            r=self.lora_r_alpha,
            lora_alpha=self.lora_r_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = prepare_model_for_kbit_training(model)
        self.model = get_peft_model(model, lora_config)

        out = super().to(*args, **kwargs)

        return out
