"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import GPT2LMHeadModel, GPT2Config

class PlanAdapter(nn.Module):
    """Handles integration of forward predictions (plans) into the model's hidden states."""
    def __init__(self, hidden_size):
        super().__init__()
        self.plan_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, plans, hidden_states):
        if plans is not None and plans.nelement() > 0:
            # print("Processing plans")
            processed_plans = self.plan_projection(plans)
            if processed_plans.shape != hidden_states.shape:
                processed_plans = torch.zeros_like(hidden_states)  # No influence if shape mismatch
            return self.scale * processed_plans + hidden_states
        return hidden_states

class GPT2Custom(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.plan_adapter = PlanAdapter(config.n_embd)
    def forward(
        self,
        input_ids=None,
        mode="ar",
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        prediction_mask=None,
        plans=None,  # New argument for forward predictions
        **kwargs,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        prediction_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            1 means the token should be predicted, 0 means it should be ignored (useful when some tokens are random)
        """
        assert labels is None
        assert mode is None or mode in [
            "ar",
            "absorbing",
            "all_but_last",
        ], f"mode {mode} unrecognized, must be either 'ar'"

        # print("In GPT2Custom forward")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if prediction_mask is None:
            prediction_mask = torch.ones_like(input_ids, dtype=torch.bool)
        prediction_mask = prediction_mask.bool()


        if mode == "ar":
            input_ids = F.pad(input_ids, (1, -1), value=self.config.bos_token_id)
            attention_mask = F.pad(attention_mask, (1, -1), value=1)
            prediction_mask = F.pad(prediction_mask, (1, -1), value=0)
            labels = input_ids.clone()
        elif mode == "absorbing":
            labels = input_ids.clone()
            (bsz, seq), device = input_ids.size(), input_ids.device
            num_masked_tokens = torch.randint(1, seq, (bsz,), device=device)
            # 1: something is always masked
            # seq: we never mask everything
            mask = torch.arange(seq, device=device).view(1, -1) >= num_masked_tokens.view(
                -1, 1
            )
            # permutations
            perms = torch.stack([torch.randperm(seq, device=device) for _ in range(bsz)])
            mask = mask.gather(1, perms)
            mask[:,0] = True
            # show all tokens that don't need prediction
            attention_mask = attention_mask.bool() & (mask | ~prediction_mask)
            # only need to predict tokens that are masked
            prediction_mask = prediction_mask & ~mask
            # replace tokens with mask token
            input_ids = input_ids.masked_fill(~attention_mask, self.config.pad_token_id)
            attention_mask = attention_mask.long()
        elif mode == "all_but_last":
            input_ids = input_ids.masked_fill(~attention_mask.bool(), self.config.pad_token_id)

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Integrate forward-generated plans using the PlanAdapter
        hidden_states = self.plan_adapter(plans, hidden_states)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # set labels to -100 where we don't need to predict
            labels = labels.masked_fill(~prediction_mask, -100)
            loss_fct = CrossEntropyLoss()
            
            if mode == "ar":
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
            elif mode == "absorbing":
                # no need to shift the labels
                loss = loss_fct(
                    lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
                )
            # print("Loss: ", loss)

                

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, plans=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # Handle plans during generation
        plans = kwargs.get("plans", None)
        if plans is not None and past_key_values:
            plans = plans[:, input_ids.size(1)-1:input_ids.size(1), :]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "plans": plans,
        }
    