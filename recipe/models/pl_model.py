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
from .planning import TokenPlanner, TokenSampleAdapter
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
        param_dict = {}
        for pn, p in self.named_parameters():
            if p.requires_grad:
                param_dict[pn] = p
                print(f"Adding trainable parameter: {pn} with shape {p.shape}")
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
        planner=None,  # New argument for the planner
        from_pretrained=False,
        train_base=True,
        checkpoint_path=None,
        num_samples=5,
        continuation_length=10,
        **model_kwargs,
    ) -> None:
        super().__init__(eval_fn=eval_fn, config_optim=config_optim, **model_kwargs)
        self.batch_counter = 0  # Add a counter to track batches
        self.printed_initial_weights = False
        
        if checkpoint_path and not train_base:
            # Load from checkpoint
            print(f"Loading model from checkpoint: {checkpoint_path}")
            self.model = GPT2PL.load_from_checkpoint(
                checkpoint_path,
                config_optim=config_optim,
                eval_fn=eval_fn,
                tokenizer=tokenizer,
            ).model
        else:
            if from_pretrained:
                print("Loading from pretrained, ignoring rest of config.")
                self.model = GPT2Custom.from_pretrained("openai-community/gpt2")
            else:
                print("Model inititalisation according to config.")
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

        # Conditionally initialize planner and adapter
        if not train_base:
            # Initialize TokenPlanner if no planner is provided
            print("Attaching TokenSampleAdapter for fine-tuning")
            if planner is None:
                # Critical fix: properly register adapter as a PyTorch module
                self.adapter = TokenSampleAdapter(
                    hidden_size=self.model.config.hidden_size,
                    vocab_size=self.model.config.vocab_size,
                    pad_token_id=tokenizer.pad_token_id if tokenizer else None
                )
                
                # Double-check that adapter params require gradients
                for name, param in self.adapter.named_parameters():
                    param.requires_grad = True
                    print(f"Setting adapter param {name} requires_grad=True")

                # Freeze GPT-2 parameters
                for param in self.model.parameters():
                    param.requires_grad = False  # Prevent GPT-2 from updating
                # Ensure PyTorch does not update the model
                self.model.training = False  # Explicitly set training mode to False
                self.model.eval()  # Ensure model is in eval mode

                for name, param in self.model.named_parameters():
                    if name.startswith("plan_adapter"):
                        param.requires_grad = True  # Allow adapter to update

                self.planner = TokenPlanner(
                    model=self.model,           # Pass the model
                    adapter=self.adapter,       # Adapter can be set up separately if needed
                    tokenizer=tokenizer,
                    config=self.model.config,   # Ensure config matches model
                )
                self.num_samples = num_samples
                self.continuation_length = continuation_length
                
                # Print summary of parameters
                print("\n===== Parameter Status After Setup =====")
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        print(f"Will train: {name}, shape: {param.shape}")
                print("=======================================\n")
            else:
                self.planner = planner
                self.add_module('planner', self.planner)
        else:
            print("Training base GPT-2 without adapter")
            self.planner = None # No planner for base model

    def forward(self, batch, **model_kwargs):
        # Print adapter weights at the start and periodically during training
        if not self.printed_initial_weights and hasattr(self, 'adapter') and False:
            # Find the first weight matrix in the adapter
            for name, param in self.adapter.named_parameters():
                if 'weight' in name and param.dim() == 2:  # Find a 2D weight matrix
                    # Print a small sample of the weights (first 5x5 elements)
                    weight_sample = param[:5, :5].detach().cpu().numpy()
                    print(f"\n========== INITIAL ADAPTER WEIGHTS ({name}) ==========")
                    print(weight_sample)
                    print("==========================================================\n")
                    # Store this parameter for later comparison
                    self.monitored_param_name = name
                    self.monitored_param = param
                    self.printed_initial_weights = True
                    break

        if False:
            print("\n======Checking adapter parameters and gradients:=======")
            for name, param in self.adapter.named_parameters():
                print(f"{name} | requires_grad={param.requires_grad} | grad_fn={param.grad_fn} | grad={(param.grad is not None)}")
            print("==========================================================\n")

        # Periodically print the weights again to see if they're changing
        if hasattr(self, 'monitored_param') and self.training and False:
            self.batch_counter += 1
            if self.batch_counter % 2 == 0:  # Every 10 batches
                weight_sample = self.monitored_param[:5, :5].detach().cpu().numpy()
                print(f"\n========== ADAPTER WEIGHTS AFTER {self.batch_counter} BATCHES ==========")
                print(weight_sample)
                print("================================================================\n")
        
        # Generate plans using the planner only if fine-tuning
        if self.planner is not None:
            # print("num_samples: ", self.num_samples)
            # print("continuation_length: ", self.continuation_length)
            # print("Forward pass is reached")
            # Sample split index randomly between 2 and max sequence length
            max_idx = batch["input_ids"].shape[1] - 1
            split_idx = torch.randint(2, max_idx, (1,)).item()

            # Split prompts at the split index
            prompts = batch["input_ids"][:, :split_idx]
            prompt_masks = batch["attention_mask"][:, :split_idx]

            plans = self.planner.forward(
                prompts=prompts,
                prompt_masks=prompt_masks,
                split_index=split_idx,  # Randomly sampled split index
                num_samples=self.num_samples,  # Number of plans to generate (K)
                continuation_length=self.continuation_length  # Length of each generated plan
            )

            # Assertion: Check if plans have the expected shape [batch_size, vec_dim]
            batch_size = batch["input_ids"].shape[0]
            vec_dim = plans.shape[-1]
            assert plans.shape == (batch_size, vec_dim), \
                f"Expected plans shape ({batch_size}, {vec_dim}), but got {plans.shape}"

            # Create the formatted plans tensor
            seq_len = batch["input_ids"].shape[1]
            formatted_plans = torch.zeros(batch_size, seq_len, vec_dim, device=plans.device, dtype=plans.dtype)

            # Expand the plan vector to match the sequence length dimension from split_idx onwards
            expanded_plans = plans.unsqueeze(1).expand(-1, seq_len - split_idx, -1)

            # Assign the expanded plans to the corresponding slice in formatted_plans
            formatted_plans[:, split_idx:, :] = expanded_plans

            # Inject the formatted plans into the batch
            batch["plans"] = formatted_plans
        
        out = self.model(
            **batch,
            mode=self.train_mode,
            **model_kwargs,
        )
        return out

    def on_after_backward(self):
        if hasattr(self, "adapter") and False:
            print("\n======Adapter gradients after backward=======")
            for name, param in self.adapter.named_parameters():
                print(f"{name} | grad: {param.grad is not None} | grad norm: {param.grad.norm().item() if param.grad is not None else 'N/A'}")
        if hasattr(self, "model") and False:
            print("\n======Plan Adapter gradients after backward=======")
            for name, param in self.model.named_parameters():
                print(f"{name} | grad: {param.grad is not None} | grad norm: {param.grad.norm().item() if param.grad is not None else 'N/A'}")

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
