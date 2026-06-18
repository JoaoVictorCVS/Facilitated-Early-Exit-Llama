# Code modified from https://github.com/huggingface/transformers/blob/v4.57-release/src/transformers/models/llama/configuration_llama.py
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
"""EELlama model configuration"""

from transformers.modeling_rope_utils import rope_config_validation

from transformers.models.llama.configuration_llama import LlamaConfig 

class EeLlamaConfig(LlamaConfig):
    r"""
    This is the configuration class to store the configuration of a [`EeLlamaModel`]. It is used to instantiate an EELlama
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.
    e.g. [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the EELlama model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`EeLlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to
            understand more about it. This value is necessary to ensure exact reproducibility of the pretraining
            results. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        head_dim (`int`, *optional*):
            The attention head dimension. If None, it will default to hidden_size // num_attention_heads

        exit_layers (`list`, *optional*, defaults to `[]`):
            The layers at which the model computes the lm head and decides whether to exit or not. If the model
            decides to exit in none of the given exit_layers, it will compute outputs at the model's final layer.
            Should be coherent to the configuration used for training the model.
        ee_softmax_threshold (`float` or `list` of `float`, *optional*, defaults to 0.7):
            The model will exit if the softmax confidence at an exit layer is higher than this threshold. 
            If given as list, should contain one threshold for each layer in `exit_layers` in increasing layer order.
        ee_entropy_threshold (`float` or `list` of `float`, *optional*, defaults to None):
            The model will exit if the entropy of predictions at an exit layer is lower than this threshold. 
            If given as list, should contain one threshold for each layer in `exit_layers` in increasing layer order.
        output_full_model (`bool`, *optional*, defaults to False):
            If True, always compute all layers until the last, use the last layer's output and return all layers logits. 
            Useful for performing model evaluation during training.
        untied_heads (`bool`, *optional* , default to `False`):
            Whether each exit layer should have a separate lm head instead of sharing weights.
        attention_weight_tuning: (`dict`, *optional*):
            Dictionary containing the configuration for attention weight tuning. If not specified, no attention weight tuning is applied.
            Example structure:
            ```
            {
                "tune_tokens": 
                    [{"token": 128006, "tune_offset": 1, "tuning": 1.2, "target_layers": LayerSubsets.EXIT_LAYERS}, ...], 
                "tune_pos": 
                    [{"pos_id": 42, "tuning": 1.2, "target_layers": LayerSubsets.EXIT_LAYERS}, ...], 
                "pos_tag_ids": 
                    [...]
            }
            ```
            where `tune_tokens` is a list of dictionaries specifying the tokens to tune by their IDs in the vocabulary,
            `tune_pos` is a list of dictionaries specifying the tokens to tune by their part-of-speech tag IDs given 
            a list of the pos tag ids for each prompt token in `pos_tag_ids`.
        enforce_exit_decision (`bool`, *optional*, defaults to `False`):
            If True, the model will compute the exit decision even during training and when `output_full_model` is True. 
            (For example to store the exit points in the stats)

    ```python
    >>> from transformers import EeLlamaModel, EeLlamaConfig

    >>> # Initializing a EELlama llama-7b style configuration
    >>> configuration = EeLlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = EeLlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "eellama"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `LlamaModel`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    """ When using the from_pretrained method with a huggingface path of a pretrained llama model, all parameters 
        specified by the model will be set accordingly, all others will have their default values defined below. 
        I.e. one can safely use a Llama3 model as pretrained backbone.
    """
    def __init__(
        self,
        *args,  # all the arguments of the parent class

        exit_layers=[],
        untied_heads=False,
        ee_softmax_threshold=0.9,
        ee_entropy_threshold=None,
        output_full_model=False,
        attention_weight_tuning=None,
        enforce_exit_decision=False,

        **kwargs,
    ):
        print("Instantiating an EeLlamaConfig.")
        super().__init__(*args, **kwargs)

        self.exit_layers = exit_layers
        self.ee_softmax_threshold = ee_softmax_threshold
        self.ee_entropy_threshold = ee_entropy_threshold
        self.output_full_model = output_full_model
        self.untied_heads = untied_heads
        self.attention_weight_tuning = attention_weight_tuning
        self.enforce_exit_decision = enforce_exit_decision


__all__ = ["EeLlamaConfig"]