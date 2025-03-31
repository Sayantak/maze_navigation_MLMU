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
        from_pretrained=False,
        train_base=True,
        train_planner=False,
        checkpoint_path=None,
        num_samples=5,
        continuation_length=10,
        **model_kwargs,
    ) -> None:
        super().__init__(eval_fn=eval_fn, config_optim=config_optim, **model_kwargs)
        self.batch_counter = 0
        self.printed_initial_weights = False
        self.num_samples = num_samples
        self.continuation_length = continuation_length
        self.train_planner = train_planner
        self.train_base = train_base

        # --- Input Validation ---
        if not train_base and not train_planner:
            # This case implies no training, maybe just inference?
            # For now, let's proceed assuming a base model is needed, but frozen.
            print("Warning: train_base=False and train_planner=False. Model will be loaded/initialized but not trained.")
            # Alternatively, raise ValueError("Both train_base and train_planner are False. Nothing to train.")

        # --- Base Model Initialization ---
        # Simplified checkpoint handling: We load manually in main.py *after* instantiation.
        # The `checkpoint_path` argument here is less useful now.
        if from_pretrained:
            print("Loading base GPT model from pretrained 'openai-community/gpt2'.")
            self.model = GPT2Custom.from_pretrained("openai-community/gpt2")
        else:
            print("Initializing base GPT model from scratch according to config.")
            if tokenizer is not None:
                model_kwargs["vocab_size"] = tokenizer.vocab_size
            model_kwargs.update(
                vocab_size=nearest_multiple(tokenizer.vocab_size),
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            model_config = GPT2Config(**model_kwargs)
            self.model = GPT2Custom(model_config)
            self.model.apply(self.model._init_weights)

        # --- Planner/Adapter Initialization (Unconditional) ---
        print("Initializing Planner and Adapter modules.")
        self.adapter = TokenSampleAdapter(
            hidden_size=self.model.config.hidden_size,
            vocab_size=self.model.config.vocab_size,
            pad_token_id=tokenizer.pad_token_id if tokenizer else None
        )
        self.planner = TokenPlanner(
            model=self.model,
            adapter=self.adapter,
            tokenizer=tokenizer,
            config=self.model.config,
        )

        # --- Parameter Freezing Logic ---
        # Freeze/Unfreeze Adapter based on train_planner
        if self.train_planner:
            print("Planner/Adapter parameters will be trained (requires_grad=True).")
            for name, param in self.adapter.named_parameters():
                param.requires_grad = True
            # Assuming TokenPlanner uses adapter params; if it had its own, handle here
        else:
            print("Planner/Adapter parameters are FROZEN (requires_grad=False).")
            for name, param in self.adapter.named_parameters():
                param.requires_grad = False
            # Set adapter to eval mode if not training it
            self.adapter.eval()

        # Freeze/Unfreeze Base Model based on train_base
        if self.train_base:
            print("Base GPT model parameters will be trained (requires_grad=True).")
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train() # Set base model to training mode
        else:
            print("Base GPT model parameters are FROZEN (requires_grad=False), except for plan_adapter parameters.")
            for name, param in self.model.named_parameters():
                if 'plan_adapter' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            self.model.eval() # Set base model to evaluation mode


        # --- Final Parameter Status Check ---
        print("\n===== Final Parameter Training Status =====")
        total_params = 0
        trainable_params = 0
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"  [TRAINABLE] {name} ({param.shape})")
            else:
                print(f"  [FROZEN]    {name} ({param.shape})")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("===========================================\n")


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
        
        # Generate plans only if train_planner is True
        if self.train_planner: # Changed condition from self.planner is not None
            # print("Generating plans...") # Optional debug print
            max_idx = batch["input_ids"].shape[1] - 1
            if max_idx < 2:
                 print(f"Warning: Sequence length ({max_idx+1}) too short for planning, skipping plan generation for this batch.")
                 # Ensure batch["plans"] doesn't exist or is handled downstream
                 # If GPT2Custom requires "plans", create a dummy zero tensor here:
                 # batch_size = batch["input_ids"].shape[0]
                 # seq_len = batch["input_ids"].shape[1]
                 # vec_dim = self.model.config.hidden_size # Or adapter output dim
                 # batch["plans"] = torch.zeros(batch_size, seq_len, vec_dim, device=batch["input_ids"].device, dtype=self.dtype)
                 pass # Assuming GPT2Custom handles missing "plans" key gracefully
            else:
                 split_idx = torch.randint(2, max_idx, (1,)).item()
                 prompts = batch["input_ids"][:, :split_idx]
                 prompt_masks = batch["attention_mask"][:, :split_idx]

                 # Ensure adapter is in the correct mode (train/eval) based on requires_grad status
                 # This might be redundant if requires_grad handles it, but can be explicit:
                 # self.adapter.train(self.adapter.parameters().__next__().requires_grad)

                 plans = self.planner.forward(
                     prompts=prompts,
                     prompt_masks=prompt_masks,
                     split_index=split_idx,
                     num_samples=self.num_samples,
                     continuation_length=self.continuation_length
                 )

                 # Assertion: Check if plans have the expected shape [batch_size, vec_dim]
                 batch_size = batch["input_ids"].shape[0]
                 vec_dim = plans.shape[-1]
                 assert plans.shape == (batch_size, vec_dim), \
                     f"Expected plans shape ({batch_size}, {vec_dim}), but got {plans.shape}"

                 # Create the formatted plans tensor
                 seq_len = batch["input_ids"].shape[1]
                 formatted_plans = torch.zeros(batch_size, seq_len, vec_dim, device=plans.device, dtype=plans.dtype)

                 # Expand the plan vector
                 expanded_plans = plans.unsqueeze(1).expand(-1, seq_len - split_idx, -1)

                 # Assign the expanded plans
                 formatted_plans[:, split_idx:, :] = expanded_plans

                 # Inject the formatted plans into the batch
                 batch["plans"] = formatted_plans
        # else: # Optional: Explicitly remove or zero out plans if not training planner
             # if "plans" in batch: del batch["plans"]

        # --- Call the base model ---
        # Ensure GPT2Custom handles the case where batch["plans"] might not exist
        out = self.model(
            **batch,
            mode=self.train_mode,
            **model_kwargs,
        )
        return out

    def on_after_backward(self):
        # --- Debug Prints (Keep disabled or remove) ---
        # if hasattr(self, "adapter") and False: ...
        # if hasattr(self, "model") and False: ...
        pass # Keep disabled prints out of the way


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
