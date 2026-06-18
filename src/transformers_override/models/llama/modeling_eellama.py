# Code modified from https://github.com/huggingface/transformers/blob/v4.57-release/src/transformers/models/llama/modeling_llama.py
# See licenses/transformers_LICENSE.txt (Apache-2.0 License)

# Additional modifications:
# Copyright 2026 hkgm

# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" EELlama: A Llama model that supports early-exit. """

from typing import Callable, Optional, Union
from typing_extensions import override

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from transformers.models.llama.configuration_llama import LlamaConfig 
from .configuration_eellama import EeLlamaConfig
from ...modeling_outputs import (
    BaseModelOutputWithPastAndLogits,
    CausalLMOutputWithPastAndEeLogits,
)
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel, eager_attention_forward, repeat_kv
from transformers.generation.logits_process import LogitsProcessorList
from transformers import AttentionInterface, AttentionMaskInterface


logger = logging.get_logger(__name__)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def tuned_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    # Repeat keys and values to have one copy for each key-value group
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # Shape of query: (batch_size, num_attention_heads, sequence_length, head_dim)
    # Shape of key_states: (batch_size, num_key_value_heads, sequence_length, head_dim)
    # Shape of value_states: (batch_size, num_key_value_heads, sequence_length, head_dim)

    # Compute attributions
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    # Shape of attn_weights: (batch_size, num_heads, sequence_length, sequence_length)
    #                                                ^ for query l  , ^ for value t

    # Before softmax, tune attributions
    if module.config.attention_weight_tuning and "tune_tokens" in module.config.attention_weight_tuning.keys() or "tune_pos" in module.config.attention_weight_tuning.keys():
        layer = module.layer_idx
        if layer < module.config.num_hidden_layers - 1: # exclude the last layer
            # Collect all types of tunings
            all_token_tune_data = [] 
            for tuning_class in ["tune_tokens", "tune_pos"]:
                if tuning_class in module.config.attention_weight_tuning.keys():
                    all_token_tune_data += module.config.attention_weight_tuning[tuning_class]
            # Tune the tokens to tune
            for token_tune_data in all_token_tune_data:
                if "target_layers" in token_tune_data.keys() and (\
                    (token_tune_data["target_layers"]==LayerSubsets.EXIT_LAYERS and not layer in module.config.exit_layers) or 
                    (token_tune_data["target_layers"]==LayerSubsets.NON_EXIT_LAYERS and layer in module.config.exit_layers) \
                ):
                    continue # This layer is not target of this tuning

                # Decaying tuning: tuning = 1 + (full_tuning - 1) * (1 - (layer / (final_layer)))
                # this is mathematically equivalent to: 
                # tuning = (1 - full_tuning) * layer / final_layer + full_tuning
                tuning_value = 1 + (torch.tensor(token_tune_data["tuning"]) - 1) * (1 - (layer / (module.config.num_hidden_layers-1)))
                attn_weights[token_tune_data["positions"][:,0], :, :, token_tune_data["positions"][:,1]] += torch.log(tuning_value)
                # increase weight w for a specific token such that in the softmax we have (e^w)*tuning instead of (e^w). (e^w)*tuning = (e^w)*(e^ln(tuning)) = (e^(w+ln(tuning))

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention, LlamaRMSNorm, LlamaRotaryEmbedding 
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel

@auto_docstring
class EELlamaModel(LlamaModel):
    config: EeLlamaConfig
    def __init__(self, config: EeLlamaConfig):
        super().__init__(config)

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        lm_head: nn.Module,
        logits_processor: LogitsProcessorList,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        eval_stats: Optional[list] = [],
        stats: Optional[dict] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPastAndLogits:
        """
        Forward pass of the model.

        Parameters
        ----------
        ...
        lm_head : list or nn.ModuleList of torch.nn.Module
            Language model head to apply at each exit position. Single-element list
            if self.config.untied_heads is False, an nn.moduleList of as many modules 
            as there are exit layers else.
        eval_stats : list
            Should contain the keys of all statistics to collect
        stats : dict
            A dict to store the results of any metrics entered in eval_stats 
        """

        def copy_cache_states(exit_layer, hidden_states, position_embeddings, past_key_values):
            """
            Perform state copying.

            Copy the hightest computed hidden states to the remaining layers and 
            in each, compute keys and values from this hidden state. That means, 
            starting from the exit_layer continue loop through all remaining layers,
            but compute nothing but the keys and values given the hidden states
            copied from the exit layer.

            Parameters
            ----------
            exit_layer : int
                The layer at which the exit decision was made and from which the 
                hidden states should be copied for the remaining layers.
            hidden_states : torch.Tensor
                The hidden states computed at the exit layer that should be copied 
                to the remaining layers.
            position_embeddings : tuple
                The position embeddings for self-attention.
            past_key_values : Cache
                The KV cache to be updated for the remaining layers.
            """

            # Prepare position embeddings for self-attention
            cos, sin = position_embeddings
            unsqueeze_dim = 1
            cos = cos.unsqueeze(unsqueeze_dim)
            sin = sin.unsqueeze(unsqueeze_dim)

            for l, decoder_layer in enumerate(self.layers[exit_layer+1 : self.config.num_hidden_layers]):
                # § Replace hidden_states = decoder_layer(...):

                normed_hidden_states = decoder_layer.input_layernorm(hidden_states)

                # §§ Replace decoder_layer.self_attn(...):

                input_shape = normed_hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, decoder_layer.self_attn.head_dim)

                key_states = decoder_layer.self_attn.k_proj(normed_hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = decoder_layer.self_attn.v_proj(normed_hidden_states).view(hidden_shape).transpose(1, 2)

                # §§§ Replace query_states, key_states = apply_rotary_pos_emb(...):
                k_embed = (key_states * cos) + (rotate_half(key_states) * sin)
                key_states = k_embed
                # §§§ End of replacing query_states, key_states = apply_rotary_pos_emb(...)

                if past_key_values is not None:
                    # sin and cos are specific to RoPE models; cache_position needed for the static cache
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_values.update(key_states, value_states, decoder_layer.self_attn.layer_idx, cache_kwargs)

                # §§ End of replacing decoder_layer.self_attn(...)
                # § End of replacing hidden_states = decoder_layer(...)


        if not (self.training or self.config.output_full_model) or self.config.enforce_exit_decision:
            assert input_ids.shape[0]==1, "EEeLlama with early exit enabled is only thought for a batch size of 1."     # TODO: It would be possible to process as batch and decide on exiting early for each instance separately. The instance would need to be deleted from the batch in case of early exiting to continue inference with the remaining instances, but all outputs at exiting would need to be saved for finally returning the result of the whole batch.
        assert input_ids != None, "EeLlamaModel requires input_ids."

        assert (not ("exit_layer" in eval_stats or "exit_layer_attributions" in eval_stats)) or (stats is not None), "A dict must be passed as parameter `stats` if a statistical metric to collect is given in `eval_stats`."
        if "exit_layer" in eval_stats and not "exit_layer" in stats: stats['exit_layer'] = []   # initialize exit_layer list
        if "exit_layer_attributions" in eval_stats and not "exit_layer_attributions" in stats: stats["exit_layer_attributions"] = {"attributions": {}, "num_tokens_summed": {}, "prompt_length": input_ids.shape[1]}

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        lm_head_input_shape = hidden_states[:, slice_indices, :].shape
        all_layers_logits = torch.empty((0, lm_head_input_shape[0], lm_head_input_shape[1], self.config.vocab_size), device=inputs_embeds.device)
        last_layer_processed = False
        output_single_token = (isinstance(logits_to_keep, int) and logits_to_keep==1) or (slice_indices.start==-1)
        if not (self.training or self.config.output_full_model) or self.config.enforce_exit_decision:
            # The following definitions are only needed if early-exit is used
            tokens_exited = torch.zeros_like(input_ids, dtype=torch.bool, device=inputs_embeds.device)
            if not self.config.output_full_model:
                last_hidden_state_at_exit = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
                logits_at_exit = torch.zeros((lm_head_input_shape[0], lm_head_input_shape[1], self.config.vocab_size), device=inputs_embeds.device)

            if not output_single_token and "exit_layer" in eval_stats:
                # Since all tokens will be processed in parallel, simply adding exit
                # layers when the exit decision is made will result in exit points
                # being collected out of order. We therefore initialize a list to 
                # write to specific indices.
                exit_points = torch.full_like(input_ids, -1, dtype=torch.int)

        if "exit_layer_attributions" in eval_stats: 
            # Below code needs to know which attention implementation to use for 
            # tracking attention weights
            attention_implementation = eager_attention_forward

        if self.config.attention_weight_tuning: 
            # Use the eager attention function with tuning
            attention_implementation = tuned_eager_attention_forward
            AttentionInterface.register("tuned_eager", tuned_eager_attention_forward)
            # When using a custom attention implementation, you also have to set 
            # the attention mask to be used (it's looked up using the name of the 
            # implementation)
            AttentionMaskInterface.register("tuned_eager", AttentionMaskInterface()["eager"])
            self.set_attn_implementation("tuned_eager")

            if "tune_tokens" in self.config.attention_weight_tuning.keys():
                # Collect the indices of all occurences of the tokens to tune in the input_ids
                for token_tune_data in self.config.attention_weight_tuning["tune_tokens"]:
                    token = token_tune_data["token"]
                    # For some tunings, we don't want to tune the token with the 
                    # given id itself but the token `tune_offset` places thereafter
                    tune_offset = token_tune_data["tune_offset"] if "tune_offset" in token_tune_data.keys() else 0
                    index_shift = torch.tensor([0, tune_offset], device=input_ids.device)   # e.g. for tune_offset = 1, add [0,1] to each tuple of indices in the next line
                    token_tune_data["positions"] = (input_ids[:, :input_ids.shape[1]-tune_offset] == token).nonzero() + index_shift     # input_ids is cropped to indices for which adding tune_offset will still produce indices within bounds.
            if "tune_pos" in self.config.attention_weight_tuning.keys():
                # Collect the indices of all tokens of the part-of-speech to tune
                pos_tag_ids = torch.tensor(self.config.attention_weight_tuning["pos_tag_ids"])
                for token_tune_data in self.config.attention_weight_tuning["tune_pos"]:
                    pos_id = token_tune_data["pos_id"]
                    token_tune_data["positions"] = (pos_tag_ids == pos_id).nonzero()

        if "exit_layer_attributions" in eval_stats:
            # Attentions can only be tracked using the eager attention implementation

            exit_layers = self.config.exit_layers
            def eager_attention_forward_with_stats(*args, **kwargs):
                """Wrapper to catch the outputs of eager attention. Functions as 
                closure for exit_layers and stats to store the attributions """
                attn_output, attn_weights = attention_implementation(*args, **kwargs)
                # shape of attn_weights (batch_size, num_heads, sequence_length, sequence_length)

                # We are interested only in the attributions for the last token 
                # (query) of the sequence, which is at the last index of the third 
                # dimension (dim 2) and we take the mean over all attention heads.
                curr_layer = args[0].layer_idx  # args[0] is the module
                if curr_layer in exit_layers:
                    # Save attributions for this exit_layer

                    curr_token_mean_attributions = attn_weights[:,:,-1,:].mean(dim=1)   # shape (batch_size, sequence_length)

                    if curr_layer not in stats["exit_layer_attributions"]["num_tokens_summed"].keys():
                        stats["exit_layer_attributions"]["attributions"][curr_layer] = curr_token_mean_attributions[:,:stats["exit_layer_attributions"]["prompt_length"]]   # Only track the attributions for tokens of the prompt, since others are only available to all later tokens.
                        stats["exit_layer_attributions"]["num_tokens_summed"][curr_layer] = 1   # first time that attributions for this layer are computed
                    else:
                        stats["exit_layer_attributions"]["attributions"][curr_layer] += curr_token_mean_attributions[:,:stats["exit_layer_attributions"]["prompt_length"]]   # Only track the attributions for tokens of the prompt, since others are only available to all later tokens.
                        stats["exit_layer_attributions"]["num_tokens_summed"][curr_layer] += 1

                # Return outputs of standard attention function
                return attn_output, attn_weights

            # Add eager attention wrapper as attention implementation to the AttentionInterface
            # see https://github.com/huggingface/transformers/blob/v4.57-release/src/transformers/modeling_utils.py#L6123
            # See also https://huggingface.co/docs/transformers/attention_interface
            AttentionInterface.register("eager_with_stats", eager_attention_forward_with_stats)
            # When using a custom attention implementation, you also have to set 
            # the attention mask to be used (it's looked up using the name of the 
            # implementation)
            attnmi = AttentionMaskInterface()
            attn_mask = attnmi["eager"]
            if attention_implementation != eager_attention_forward:
                attn_mask = attnmi[self.config._attn_implementation]
            AttentionMaskInterface.register("eager_with_stats", attn_mask)
            self.set_attn_implementation("eager_with_stats")


        for l, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )


            ##### Early-Exit extension #####
            if l in self.config.exit_layers:
                # Only consider early exit, when processing a single token.
                # batch_size is asserted to be 1 if an exit-decision is to be made.

                # Compute confidence for this exit-layer
                # Shape of hidden_states: (batch_size, sequence_length, self.config.hidden_size)
                hidden_states_exit = torch.clone(hidden_states)
                hidden_states_exit = self.norm(hidden_states_exit)
                
                # Only compute necessary logits, and do not upcast them to float 
                # if we are not computing the loss
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                lm_head_index = self.config.exit_layers.index(l) if self.config.untied_heads else -1
                logits = lm_head[lm_head_index](hidden_states_exit[:, slice_indices, :])
                # Shape of logits: (batch_size, logits_to_keep, vocab_size)

                if self.training or self.config.output_full_model:
                    # Collect the logits at this layer for later calculation of 
                    # the loss
                    all_layers_logits = torch.cat([all_layers_logits, logits[None].to(inputs_embeds.device)], dim=0)

                last_layer_processed = (l==self.config.num_hidden_layers-1) # If this is the last layer, set a flag to tell the end of the function that the logits are already computed.

                # Exit decision
                if (not (self.training or self.config.output_full_model)) or self.config.enforce_exit_decision:   # for training, the whole model has to be computed with all losses, but if desired, the theoretical exit points can still be computed and the logits at those points returned.
                    # Compare from here: https://github.com/huggingface/transformers/blob/v4.57-release/src/transformers/generation/utils.py#L2800
                    next_token_logits = logits.to(copy=True, dtype=torch.float32, device=input_ids.device)
                    next_token_scores = next_token_logits
                    if output_single_token:
                        next_token_scores = logits_processor(input_ids, next_token_logits[:, -1, :]).unsqueeze(1)   # unsqueeze(1) to add back the logits_to_keep dimension.
                    else:
                        # don't use the logits_processor in case of generation with teacher forcing
                        # it should even be empty
                        assert len(logits_processor)==0
                    # Shape of next_token_scores: (batch_size, logits_to_keep, vocab_size)
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    # Shape of probs: (batch_size, logits_to_keep, vocab_size)

                    exit_early = torch.zeros_like(tokens_exited, dtype=torch.bool, device=inputs_embeds.device)
                    # Shape of exit_early: (batch_size, sequence_length) # equals shape of tokens_exited
                    if self.config.ee_entropy_threshold is not None:
                        # Use entropy-based confidence
                        probs_nonzero = probs[0,:,:].clamp(min=1e-12)   # assumes batch_size is 1
                        entropy = - (probs_nonzero * torch.log(probs_nonzero)).sum(dim=-1)  # one per logits_to_keep
                        threshold = self.config.ee_entropy_threshold if isinstance(self.config.ee_entropy_threshold, float) else self.config.ee_entropy_threshold[self.config.exit_layers.index(l)]
                        exit_early[-1,slice_indices] = entropy < threshold    # shape (logits_to_keep,)   # assumes batch_size is 1
                    else:
                        # Use softmax confidence
                        softmax_confidence = probs.max(dim=-1).values[0]   # assumes batch_size is 1
                        threshold = self.config.ee_softmax_threshold if isinstance(self.config.ee_softmax_threshold, float) else self.config.ee_softmax_threshold[self.config.exit_layers.index(l)]
                        exit_early[-1,slice_indices] = softmax_confidence > threshold

                    # If tokens have already exited, don't exit them again (i.e. keep their lower-layer outputs)
                    exit_early = torch.logical_and(exit_early, torch.logical_not(tokens_exited))

                    for kept_token_index, t in enumerate(torch.arange(input_ids.shape[1])[slice_indices]):   # Iterate over all tokens of the sequence whose logits should be kept. Since the logits have only been computed for the logits_to_keep, we also need the index of those in the range [0:logits_to_keep], which is given by the index when enumerating the list of indices to keep. Thus: t is the global index of the kept tokens (counting over all tokens in the sequence), kept_token_index is the local index (counting only over the slice of kept indices)
                        if exit_early[-1, t]:   # assumes batch_size is 1
                            # Exit early (for token t)

                            if output_single_token: # assumes batch_size is 1
                                # Only one logit_to_keep and only one batch instance. So
                                # exactly one token is generated by this run, as for usual
                                # inference using the models generate method.
                                # In this case, stop computation here and return

                                if "exit_layer" in eval_stats:
                                    stats["exit_layer"].append(l)

                                if not self.training: print(".", end="")
                                if use_cache:
                                    copy_cache_states(l, hidden_states, position_embeddings, past_key_values)
                                if not (self.training or self.config.output_full_model):
                                    return BaseModelOutputWithPastAndLogits(
                                        last_hidden_state=hidden_states_exit,
                                        past_key_values=past_key_values,
                                        logits=logits,
                                        all_layers_logits=all_layers_logits,
                                        stats=stats
                                    )

                            else:
                                # There are multiply logits_to_keep, so multiple tokens
                                # should be computed using teacher forcing. In this case,
                                # save the state of the tokens to exit as final state, 
                                # mark them as exited and continue computation for the 
                                # rest.

                                # No state copying needed, since computation will
                                # continue.

                                if "exit_layer" in eval_stats:
                                    exit_points[-1, t] = l  # assumes batch_size is 1

                                if not self.config.output_full_model:
                                    last_hidden_state_at_exit[-1, t, :] = hidden_states_exit[-1, t, :] # assumes batch_size is 1
                                    logits_at_exit[-1, kept_token_index, :] = logits[-1, kept_token_index, :] # assumes batch_size is 1
                                tokens_exited[-1, t] = True # assumes batch_size is 1 
                                # Continue computation.
                    
                    # At this point, the model could return if self.config.output_full_model 
                    # is False and all tokens have exited. This would save computation
                    # by early-exitting in case of teacher-forced generation.
                    # However, it might be necessary to first perform state copying,
                    # if the model is expected to compute more tokens after this 
                    # run. For the moment, we simply continue computation.

                # TODO: To improve efficiency despite using softmax confidence, 
                # the second computation of softmax in the generate function
                # (https://github.com/huggingface/transformers/blob/v4.57-release/src/transformers/generation/utils.py#L2827)
                # should be avoided. Also look at the logits_processors used. They can be computation-heavy.

        hidden_states = self.norm(hidden_states)
        
        if not last_layer_processed:
            # Could already have been computed, if the final layer of the model 
            # is given as exit layer.

            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = lm_head[-1](hidden_states[:, slice_indices, :])

            if self.training or self.config.output_full_model:
                # Collect the logits at this layer for later calculation of the loss
                all_layers_logits = torch.cat([all_layers_logits, logits[None].to(inputs_embeds.device)], dim=0)

        # In case of training or full-model-computation, everything is done unless
        # self.config.enforce_exit_decision is True.
        if (self.training or self.config.output_full_model) and not self.config.enforce_exit_decision:
            # No early exit performed. Just return as without early exit.
            return BaseModelOutputWithPastAndLogits(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                logits=logits,
                all_layers_logits=all_layers_logits,
                stats=stats
            )
        assert inputs_embeds.shape[0] == 1   # batches are only expected for training. From here on, there should only exist a single batch element. # assumes batch_size is 1


        # Exit all remaining tokens
        if output_single_token:  # assumes batch_size is 1
            # Only one logit_to_keep and only one batch instance. So
            # exactly one token is generated by this run, as for usual
            # inference using the models generate method.
            # In this case, stop computation here 

            if "exit_layer" in eval_stats:
                stats["exit_layer"].append(self.config.num_hidden_layers-1)

            return BaseModelOutputWithPastAndLogits(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                logits=logits,
                all_layers_logits=all_layers_logits,
                stats=stats
            )
        else:   # assumes batch_size is 1
            # There are multiply logits_to_keep, so multiple tokens
            # should be computed using teacher forcing. In this case,
            # save the state of the tokens to exit as final state, 
            # mark them as exited and continue computation for the 
            # rest.
            exit_token = torch.logical_not(tokens_exited)   # Exit all tokens that haven't yet.
            for kept_token_index, t in enumerate(torch.arange(input_ids.shape[1])[slice_indices]):   # Iterate over all tokens of the sequence whose logits should be kept. Since the logits have only been computed for the logits_to_keep, we also need the index of those in the range [0:logits_to_keep], which is given by the index when enumerating the list of indices to keep. Thus: t is the global index of the kept tokens (counting over all tokens in the sequence), kept_token_index is the local index (counting only over the slice of kept indices)
                if exit_token[0, t]:   # assumes batch_size is 1
                    # Exit for token t
                    if "exit_layer" in eval_stats:
                        exit_points[-1, t] = l  # assumes batch_size is 1
                    if not self.config.output_full_model:
                        last_hidden_state_at_exit[-1, t, :] = hidden_states[-1, t, :] # assumes batch_size is 1
                        logits_at_exit[-1, kept_token_index, :] = logits[-1, kept_token_index, :] # assumes batch_size is 1
                    tokens_exited[-1, t] = True # assumes batch_size is 1

            assert tokens_exited[-1,slice_indices].all()    # assumes batch_size is 1

            if "exit_layer" in eval_stats:
                stats["exit_layer"] = exit_points.tolist()

            hidden_states_to_return = last_hidden_state_at_exit if not self.config.output_full_model else hidden_states
            return BaseModelOutputWithPastAndLogits(
                last_hidden_state=hidden_states_to_return,
                past_key_values=past_key_values,
                logits=logits_at_exit if not self.config.output_full_model else logits,
                all_layers_logits=all_layers_logits,
                stats=stats
            )
        
        assert False, "Function should already have returned."


@auto_docstring
class EeLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    config: EeLlamaConfig

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        print("My config_class is ", self.config_class)
        self.model = EELlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.untied_heads:
            self.lm_heads = nn.ModuleList([nn.Linear(config.hidden_size, config.vocab_size, bias=False) for l in config.exit_layers if l!=self.config.num_hidden_layers-1])    # create an LM head for all but the final layer
            self.lm_heads.append(self.lm_head)    # add the final layer's head
            
        self.logits_processor: LogitsProcessorList = LogitsProcessorList()  # Initialize with empty LogitsProcessorList. Will be set in _get_logits_processor if the generate method is used to execute this model.

        # Initialize weights and apply final processing
        self.post_init()

    @override
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPastAndEeLogits:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM
        >>> from transformers_override.models.llama.modeling_eellama import EeLlamaForCausalLM 

        >>> model = EeLlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        outputs: BaseModelOutputWithPastAndLogits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            lm_head=[self.lm_head] if not self.model.config.untied_heads else self.lm_heads,
            logits_processor=self.logits_processor,
            **kwargs,
        )

        loss = None

        return CausalLMOutputWithPastAndEeLogits(
            logits=outputs.logits,
            loss=loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            all_layers_logits=outputs.all_layers_logits,
            stats=outputs.stats
        )

    @override # of GenerationMixin
    def _get_logits_processor(self, *args, **kwargs):
        """
        Catch the result of _get_logits_processor.

        In order to process model output logits in the EeLlameModel for 
        evaluating EE confidence, we need to get the logits_processor list used 
        by the decoding method.
        Call in generate method: https://github.com/huggingface/transformers/blob/v4.57-release/src/transformers/generation/utils.py#L2543
        """
        # print("Receiving logits_processor")
        self.logits_processor: LogitsProcessorList = super(EeLlamaForCausalLM, self)._get_logits_processor(*args, **kwargs)
        return self.logits_processor

    @override # of GenerationMixin
    def prepare_inputs_for_generation(
        self,
        *args,
        eval_stats: Optional[list] = [],
        stats: Optional[dict] = None,
        **kwargs
    ):
        """
        Add keyword arguments specific to the EeLlamaForCausalLM to the expected
        model_inputs.

        Parameters
        ----------
        eval_stats : list
            Should contain the keys of all statistics to collect
        stats : dict
            A dict passed as copy-by-value parameter to store evaluated statistics.
        """
        model_inputs = super(EeLlamaForCausalLM, self).prepare_inputs_for_generation(*args, **kwargs)
        model_inputs["eval_stats"] = eval_stats
        model_inputs["stats"] = stats

        return model_inputs


from enum import Enum
class LayerSubsets(int, Enum):
    ALL_LAYERS = 1
    EXIT_LAYERS = 2
    NON_EXIT_LAYERS = 3


__all__ = [
    "EeLlamaForCausalLM",
    "EeLlamaModel"
    "LayerSubsets"
]