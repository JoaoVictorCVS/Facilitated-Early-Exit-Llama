# https://huggingface.co/transformers/v3.0.2/training.html

"""
Classes and functions for fine-tuning (or training) selected parts of an EELlama
model for facilitated early-exit.

Instantiate a ModelFineTuner with a ModelFineTunerConfig and optional functions 
for further model initialization (e.g., attaching a prompt modifier) or dataset 
preparation. Then call the `fine_tune()` method to start the fine-tuning process. 
The ModelFineTunerConfig can be created from a yaml file using `ModelFineTunerConfig.load(config_path)`.
See `ModelFineTunerConfig.__init__()` for a list of all configuration parameters.
The class ModelFineTuner especially contains a function `compute_loss()` which 
offers extensive options for tuning a model for early-exit, such as a penalization 
of the expected inference cost and a difficulty-aware loss shaping. For using the 
difficuty-aware loss shaping, set configuration parameter `difficulty_aware_loss` 
to True and set environment variable SHAPING_FCT_CONFIG to the number of the desired
loss shaping (e.g. 7 for the best of our experiments).

"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# can also execute with `CUDA_VISIBLE_DEVICES=0 python3 fine-tuning.py`

from collections import OrderedDict
from functools import partial
import time
import yaml
from typing import Any, Dict
import logging

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainerCallback
from transformers.integrations import TensorBoardCallback
from peft import PeftModel
from trl import SFTConfig, SFTTrainer

from transformers_override.models.llama.configuration_eellama import EeLlamaConfig
from transformers_override.models.llama.modeling_eellama import EeLlamaForCausalLM

from utils.configuration_utils import GenericConfig
import utils.logging_utils


logger = logging.getLogger(__name__)

@staticmethod
def causal_lm_ce_loss(logits, labels, vocab_size: int, num_items_in_batch: torch.Tensor | None = None, ignore_index: int = -100, shift_labels: torch.Tensor | None = None, reduction: str = None, **kwargs) -> torch.Tensor:
    # Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py#L45
    # We allow to use no reduction if reduction="none" is passed.
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    shift_labels = shift_labels.to(logits.device)

    if reduction is None:
        # Flatten the tokens
        logits = logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(logits.device)

        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = F.cross_entropy(logits, shift_labels, ignore_index=ignore_index, reduction=reduction)
        if reduction == "sum":
            # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch
    else:
        # Swap sequence and vocab dimensions for cross_entropy (should be (batch_size, vocab_size, sequence_length))
        logits_transposed = logits.permute(0, 2, 1)
        loss = F.cross_entropy(logits_transposed, shift_labels, ignore_index=ignore_index, reduction=reduction)
    return loss


class ModelFineTunerConfig(GenericConfig):
    """Configuration class for the ModelFineTuner."""
    
    def __init__(self):
        """Initialize config with default values."""
        config: Dict[str, Any] = {
            "output_path": "../models/llama-3-eellama-3-8B-instruct-lora-tuned-embedding",  # Path for saving the fine-tuned model parameters.
            "model": {
                "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",  # ID of the model in the transformers library
                "adapter_path": "../models/llama-3-eellama-3-8B-instruct/llama-3-eellama-3-8B-instruct-adapter-cnn-dm/checkpoint-4830", # A LoRA adapter to attach for the EELlama model
                "exit_layers": [7, 15, 23, 31],         # List of the layers at which the model can exit (should correspond to the exit layers used during training of the adapter)
                "state_dict_updates": [],               # Paths to ordered dict pth files for overwriting parts of the model's state dict.
                "copy_chat_template_model_id": None,    # If the model does not have a chat template, pass the transformers ID of another model to copy its chat template from.
                "tied_word_embeddings": None,           # Set to True, if the embedding matrix and the lm head weights should be tied.
                "ee_softmax_threshold": None,           # Either a single float value between 0 and 1 to use for the exit decision at all exit layers, or a list of float values for each exit layer. 
            },
            "keys_to_tune": ["base_model.model.model.embed_tokens.weight", "base_model.model.lm_head.weight"],  # The keys of the model parameters to be fine-tuned.
            "modules_to_tune": [],                      # Will add all keys with these prefixes to keys_to_tune.
            "print_requires_grad": True,                # Set to True, if the keys of all parameters that will be fine-tuned should be printed before fine-tuning.
            "sft_config_args": {},                      # Arguments to pass to the SFT Trainer (see https://huggingface.co/docs/trl/sft_trainer)
            "dataset": {
                "path": "../datasets/cnn-dm_short",     # Path of the dataset to use for fine-tuning. 
                "select_range": None,                   # e.g., [0,100] to select only first 100 samples
                "fraction_train": 0.98,                 # Fraction of the dataset to use for training (the rest will be used for evaluation)
            },
            "loss_config_params": {
                "uniform_weights": False,               # Whether to use uniform weights for the weighting the losses computed at all exit layers or to use weights that increase towards the final layer
                "probability_weights": False,           # Whether to weight the loss of each exit layer according to the model's exit probabilities at this layer (computed from the softmax confidence scores)
                "uniform_entropy_weights": None,        # Whether the weights for the entropy penalization at all exit layers should be uniform.If None, use the same weights as for the ce_loss (`uniform_weights`).
                "prob_weights_temperature": None,       # The temperature S by which to divide during computation of the exit probability with the sigmoid function in the case of using probability_weights or expected-cost penalization (`expected_cost_coefficient`).
                "prob_weights_fct": "sigmoid",          # The function to use for computing exit probabilities from the softmax confidence scores. Either "sigmoid" or "linear".
                "ce_coefficient": 1,                    # The coefficient by which to weight the cross-entropy loss term.
                "entropy_coefficient": 0,               # The coefficient by which to weight the entropy penalization loss term.
                "expected_cost_coefficient": 0,         # The coefficient by which to weight the expected-cost penalization loss term. 
                "expected_cost_includes_exit": True,    # Whether the expected cost computation should include both the probability to not exit at previous layers and the probability to exit at the current layer or only the probability to not exit at previous layers.
                "ignore_completion_only_setting_for_entropy": True,   # Should be set to False. While used in the beginning of our experiments, this is not recommended and only implemented for reproducibility!
                "difficulty_aware_loss": False,         # Whether to apply a difficulty-aware loss shaping to the cross-entropy loss term.
            },
            "evaluate_exit_layer": False,               # Whether the mean exit layer should be evaluated during the evaluation phase of training. This will slow down evaluation.
        }
        super().__init__(config)


class ModelFineTuner:

    class SavePartialModelCallback(TrainerCallback):
        """
        A callback for the SFT Trainer used to save only a certain parameter of the 
        model if a new optimal loss is achieved instead of saving the whole model.
        Particularly, only the embedding layer is fine-tuned, so we only want to save
        the embedding layer.
        """
        def __init__(self, keys_to_save, save_dir, timestamp=int(time.time())):
            self.keys_to_save = keys_to_save
            if save_dir[-1] != "/": save_dir += "/"
            self.save_dir = save_dir
            self.best_metric = None
            self.start_timestamp = timestamp


        def on_evaluate(self, args, state, control, model=None, **kwargs):
            metrics = kwargs.get("metrics", {})
            current = metrics.get(args.metric_for_best_model, None)

            # Only proceed when we have a metric
            if current is None:
                return

            # If best or first metric -> save model parameter
            if self.best_metric is None or \
            (args.greater_is_better and current > self.best_metric) or \
            (not args.greater_is_better and current < self.best_metric):

                self.best_metric = current

                # Extract only the parameter to save
                state_dict_selection = OrderedDict()
                for key_to_save in self.keys_to_save:
                    state_dict_selection[key_to_save] = model.state_dict()[key_to_save].detach().cpu()

                # Save it
                outfile = f"{self.save_dir}tuned_param{self.start_timestamp}.pth"
                torch.save(state_dict_selection, outfile)
                logger.info(f"New optimum! Saved parameters at {outfile}")

                # Prevent the trainer from saving the whole checkpoint
                control.should_save = False
            
    class ComposedLossLoggingCallback(TrainerCallback):
        """
        A callback for the SFT Trainer used to include any sublosses computed 
        in the compute_loss function in the logging.
        """
        def __init__(self, loss_storage):
           self.loss_storage = loss_storage    # to get the loss_storage from the ModelFineTuner
           super().__init__()
        def on_log(self, args, state, control, logs=None, **kwargs):
            # Build mean of all subloss items collected and clear for the 
            # next steps.
            for subloss in self.loss_storage.keys():
                if len(self.loss_storage[subloss])>0:    # Might be zero if the subloss is for example only computed during evaluation and we log during training steps.
                    mean_loss = torch.tensor(self.loss_storage[subloss], dtype=torch.float32).mean()
                    logs[subloss] = mean_loss.item()
                    self.loss_storage[subloss].clear()
    
    class StateResetCallback(TrainerCallback):
        """
        A callback for the SFT Trainer used to reset the model state after each
        training step or substep.
        """
        def on_step_end(self, args, state, control, model=None, **kwargs):
            model.reset_sliding_history_window()

        def on_substep_end(self, args, state, control, model=None, **kwargs):
            model.reset_sliding_history_window()
        
        def on_evaluate(self, args, state, control, model=None, **kwargs):
            model.reset_sliding_history_window()

    class ConfiguredEvaluationSFTTrainer(SFTTrainer):
        """
        An SFTTrainer that allows to modify the model before and after each evaluation.
        Also resets the EELlama model's sliding history window after each 
        evaluation step.
        """
        def __init__(self, mft_config: ModelFineTunerConfig, *args, **kwargs):
            self.is_eval_run = False
            self.mft_config = mft_config
            super().__init__(*args, **kwargs)

        def evaluate(self, *args, **kwargs):
            # Change model parameters before evaluation
            if self.mft_config["evaluate_exit_layer"]:
                self.model.config.enforce_exit_decision = True
            self.is_eval_run = True

            results = super().evaluate(*args, **kwargs)

            # Change back model parameters after evaluation
            self.is_eval_run = False
            if self.mft_config["evaluate_exit_layer"]:
                self.model.config.enforce_exit_decision = False

            return results

        def compute_loss(self, model, inputs, *args, **kwargs):
            if self.is_eval_run:
                if self.mft_config["evaluate_exit_layer"]:
                    # In case of an evaluation step, add eval_stats and stats to the model
                    # inputs of the forward call
                    inputs["eval_stats"] = ["exit_layer"]
                    inputs["stats"] = {}
            return super().compute_loss(model, inputs, *args, **kwargs)

        def prediction_step(self, model, inputs, *args, **kwargs):
            result = super().prediction_step(model, inputs, *args, **kwargs)
            model.reset_sliding_history_window()
            return result


    def __init__(self, config: ModelFineTunerConfig, post_init_model_func=None, prepare_datasets_func=None):
        """Initialize the ModelFineTuner for an EELlama model.

        Parameters
        ----------
        config : ModelFineTunerConfig
            The configuration for the ModelFineTuner.
        post_init_model_func : callable, optional
            A function that takes the model, tokenizer and config as input and 
            returns the modified model and tokenizer. This can be used to apply 
            further modifications to the standard EELlama model.
        prepare_datasets_func : callable, optional
            A function that takes the config as input and returns the train and 
            eval datasets to be used for fine-tuning. If None, the dataset in the
            config will be treated as a cnn/dm dataset.
        """
        self.config = config
        self.model, self.tokenizer = self._init_eellama_model(post_init_model_func)

        # Select layers to tune
        keys_found = 0
        modules_found = set([])
        if self.config["print_requires_grad"]:
            logger.info("Modelparameters to be trained:")
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for key in self.config["keys_to_tune"]:
                if key in name:
                    keys_found += 1
                    param.requires_grad = True
            for module_name in self.config["modules_to_tune"]:
                if module_name in name:
                    modules_found.add(module_name)
                    param.requires_grad = True
                    self.config["keys_to_tune"].append(name)
                    keys_found += 1
            if self.config["print_requires_grad"] and param.requires_grad:
                logger.info(f"- {name}")

        assert keys_found == len(self.config["keys_to_tune"]), \
            "Not all keys to tune were found in the model parameters."
        assert len(modules_found) == len(self.config["modules_to_tune"]), \
            "Not all modules to tune were found in the model parameters."

        self.model.train()

        if prepare_datasets_func is not None:
            self.train_dataset, self.eval_dataset = prepare_datasets_func(self.config)
        else:
            self.train_dataset, self.eval_dataset = self._prepare_cnn_dm_datasets()

        sft_config_args = {
            "gradient_checkpointing": False,
            "gradient_accumulation_steps": 4,  
            "per_device_train_batch_size": 4, 
            "auto_find_batch_size": True,

            "max_length": 512,     # the maximum length of prompt+completion in cnn/dm is 16336 (check reduce(lambda m, item: max(m, item), dataset.map(lambda item: {'length': len(item['highlights']) + len(item['article'])})['length'])) . However, the average length is about 1000. We reduced the set to only short instances.
            "packing": False,

            "num_train_epochs": 3,
            "learning_rate": 1e-5,
            
            "logging_steps": 20,
            # "logging_dir": './runs',  # logging_dir is deprecated
            "output_dir": self.config["output_path"],
            "report_to": "none", # We add a TensorBoardCallback below instead to allow for further use of the tensorboard writer or for modifying logging.

            "bf16": torch.cuda.is_bf16_supported(including_emulation=False),
            "push_to_hub": False,
            "padding_free": False,
            "eval_strategy": "steps",
            "save_strategy": "no",
            "save_total_limit": 1,
            "metric_for_best_model": "eval_loss",
            "load_best_model_at_end": False,
            "eval_on_start": True,

            "seed": 42,
            # "full_determinism": True,
        }
        # Override/add any args from config
        for key, val in self.config["sft_config_args"].items():
            sft_config_args[key] = val
        if self.config["evaluate_exit_layer"]:
            sft_config_args["per_device_eval_batch_size"] = 1       # computing exit points for batches is not implemented
        self.sft_config = SFTConfig(**sft_config_args)

        timestamp = int(time.time())
        logger.info(f"Timestamp used for output naming: {timestamp}")
        self.spm_callback = self.SavePartialModelCallback(
            keys_to_save=self.config["keys_to_tune"],
            save_dir=self.config["output_path"],
            timestamp=timestamp
        )

        tensorboard_logging_dir = f"{self.config['output_path']}/tensorboard_logs/{timestamp}"
        self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_logging_dir)
        # Set directory for reporting to TensorBoard
        os.environ["TENSORBOARD_LOGGING_DIR"] = tensorboard_logging_dir
        # See https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/integrations/integration_utils.py#L628

        self.loss_storage = {
            "ce_loss": [],
            "entropy_loss": [],
            "expected_cost": []
        }
        if self.config["evaluate_exit_layer"]: self.loss_storage["eval_exit_layers"] = []
        self.cll_callback = self.ComposedLossLoggingCallback(self.loss_storage)

        self.state_reset_callback = self.StateResetCallback()

        self.trainer = self.ConfiguredEvaluationSFTTrainer(
            self.config,
            model=self.model,
            processing_class=self.tokenizer,
            args=self.sft_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_loss_func=partial(self.compute_loss, self.model, self.config["model"]["exit_layers"], self.loss_storage, causal_lm_ce_loss, config_params=self.config.get("loss_config_params", None)),
            callbacks=[self.spm_callback, self.cll_callback, self.state_reset_callback, TensorBoardCallback(self.tensorboard_writer)], # Attention: the cll_callback must be before the TensorBoardCallback to log the composed losses so that they are included in the TensorBoard logs.
        )

    def _init_eellama_model(self, post_init_model_func=None):
        """Initializes an EELlama model for fine-tuning based on the configuration.

        Parameters
        ----------
        post_init_model_func : callable, optional
            A function that takes the model, tokenizer and config as input and 
            returns the modified model and tokenizer. This can be used to apply 
            further modifications to the standard EELlama model.
        
        Returns
        -------
        model : EeLlamaForCausalLM
            The initialized EELlama model ready for fine-tuning.
        tokenizer : AutoTokenizer
            The tokenizer corresponding to the EELlama model.
        """
        model_id = self.config["model"]["model_id"]
        adapter_path = self.config["model"]["adapter_path"]
        exit_layers = self.config["model"]["exit_layers"]
        model_kwargs = {}
        if "ee_softmax_threshold" in self.config["model"]:
            model_kwargs["ee_softmax_threshold"] = self.config["model"]["ee_softmax_threshold"]
        model = EeLlamaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",

            exit_layers=exit_layers,
            untied_heads=False,
            output_full_model=True,
            **model_kwargs
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if self.config["model"]["copy_chat_template_model_id"] is not None:
            # Add chat template to non-chat model
            tokenizer.eos_token_id = model.config.eos_token_id
            chat_template = AutoTokenizer.from_pretrained(self.config["model"]["copy_chat_template_model_id"]).chat_template
            chat_template = chat_template.replace("<|eot_id|>", tokenizer.decode(model.config.eos_token_id))  # Ensure that the chat template uses the correct eos token id
            tokenizer.chat_template = chat_template

        if model.model.config.untied_heads:
            # Initialize all lm heads to the same values as the final layers head
            for h in range(len(model.lm_heads)-1):
                model.lm_heads[h].load_state_dict(model.lm_head.state_dict(), assign=True)

        if self.config["model"]["tied_word_embeddings"] != None:
            model.model.config.tie_word_embeddings = self.config["model"]["tied_word_embeddings"]

        model = PeftModel.from_pretrained(model, adapter_path)

        for state_dict_update_path in self.config["model"]["state_dict_updates"]:
            logger.info(f"Loading state_dict {state_dict_update_path}")
            updated_state_dict = torch.load(state_dict_update_path)
            missing_keys, unexpected_keys = model.load_state_dict(updated_state_dict, strict=False)
            assert len(unexpected_keys) == 0, f"The following parameters could not be loaded while loading {state_dict_update_path}: {str(unexpected_keys)}"

        if post_init_model_func is not None:
            model, tokenizer = post_init_model_func(model, tokenizer, self.config)

        return model, tokenizer

    def _prepare_cnn_dm_datasets(self):
        """Prepares the CNN/DailyMail dataset given in the config for fine-tuning
            with the SFTTrainer.
        """
        def format_dataset(example):
            return {"messages": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["completion"]},
            ]}
            
        dataset = load_from_disk(self.config["dataset"]["path"])
        dataset = dataset.rename_column("article", "prompt")
        dataset = dataset.rename_column("highlights", "completion")
        dataset = dataset.map(format_dataset)
        dataset = dataset.remove_columns(['prompt', 'completion'])
        if self.config["dataset"]["select_range"] is not None:
            dataset = dataset.select(range(self.config["dataset"]["select_range"][0], self.config["dataset"]["select_range"][1]))

        dataset = dataset.shuffle(seed=42)
        fraction_train = self.config["dataset"]["fraction_train"]
        logger.info("Split dataset: #train / #eval = %d / %d", int(len(dataset)*fraction_train), len(dataset) - int(len(dataset)*fraction_train))
        train_dataset = dataset.select(range(0,int(len(dataset)*fraction_train)))
        eval_dataset = dataset.select(range(int(len(dataset)*fraction_train),len(dataset)))

        return train_dataset, eval_dataset


    @staticmethod
    def compute_loss(model, exit_layers, loss_storage, causal_lm_ce_loss, outputs, labels, num_items_in_batch=None, config_params=None):
        """
        Computation of the model loss.

        We use a weighted sum of cross-entropy losses of all exit layers (should
        include the final layer). As compared to normal training, pass the full 
        model's predictions as labels to get an early exit loss.

        Parameters
        ----------
        model : EeLlamaForCausalLM
            The EELlama model being fine-tuned from the ModelFineTuner.
        exit_layers : list of int
            The indices of the exit layers for which the loss should be computed
            from the ModelFineTuner.
        loss_storage : dict
            A dictionary to store the computed sublosses for logging from the ModelFineTuner.
        causal_lm_ce_loss : callable
            The cross-entropy loss function for causal language modeling to be 
            used for loss computation without reduction (e.g. for using token-wise 
            probability weights))

        outputs : list of CausalLMOutputWithPast
            Outputs returned by the model at each token.
        labels : torch.Tensor
            The target tokens. Shape (batch_size, sequence_length)
        num_items_in_batch : int
            The number of items in the entire accumulated batch 
            (from transformers/trainer.py:5620: num_items_in_batch = sum((batch["labels"].ne(-100)).sum() for batch in batch_samples))

        config_params : dict, optional
            Additional parameters from the config from the ModelFineTuner.

        Notes
        -----
        The elements of outputs are expected to contain a field all_layers_logits
        of shape shape (len(exit_layers), batch_size, sequence_length, vocabulary_size)
        with all the output logits of all exit layers, tokens and batch elements.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if not outputs.loss==None:
            import warnings
            warnings.warn("Losses shouldn't already be computed to avoid repeated computation")
        num_exit_layers = len(exit_layers)
        assert num_exit_layers==outputs.all_layers_logits.shape[0]

        if "stats" in outputs and "exit_layer" in outputs["stats"]:
            assert len(outputs["stats"]["exit_layer"])==1, "This implementation assumes a batch size of 1 during evaluation."
            loss_storage["eval_exit_layers"] += outputs["stats"]["exit_layer"][0]

        # Combine all batches and tokens to a one-dimensional sequence of length N
        losses = torch.empty((0,), device=device)
        entropy_losses = torch.empty((0,), device=device)
        expected_cost = torch.tensor(0.0, device=device)

        weights_uniform = torch.tensor([1/len(exit_layers)] * len(exit_layers), device=device)    # uniform variant
        weights_increasing = torch.tensor(exit_layers, dtype=float, device=device) / torch.sum(torch.tensor(exit_layers, device=device))  # i / sum_i (i)

        if not config_params:
            ce_coefficient = 1
            entropy_coefficient = 0
            expected_cost_coefficient = 0
        else:
            ce_coefficient = config_params.get("ce_coefficient", 1)
            entropy_coefficient = config_params.get("entropy_coefficient", 0)
            expected_cost_coefficient = config_params.get("expected_cost_coefficient", 0)

        coefficients_sum = ce_coefficient + entropy_coefficient + expected_cost_coefficient
        ce_coefficient = ce_coefficient / coefficients_sum
        entropy_coefficient = entropy_coefficient / coefficients_sum
        expected_cost_coefficient = expected_cost_coefficient / coefficients_sum

        if config_params and config_params.get("probability_weights", False):
            # Weights according to exit probabilities

            # Compute softmax confidence scores for all exit layers, all batch 
            # elements, and all tokens
            with torch.no_grad():
                # Compute the softmax value of the maximum logit for each exit layer, batch element and token
                # For memory efficient computation, we do softmax(logits)[max_index] = e^max_logit/sum(e^logits) = exp(log( e^max_logit/sum(e^logits) )) = exp( max_logit - log(sum(e^logits)) )
                all_layers_logits = outputs.all_layers_logits.to(copy=True, device=outputs.all_layers_logits.device)
                
                # Mask tokens early to reduce computation on padded/discarded tokens
                token_mask = (labels != -100)  # shape (batch_size, sequence_length)
                mask_expanded = token_mask.unsqueeze(0).unsqueeze(-1)  # shape (1, batch_size, sequence_length, 1) for broadcasting
                masked_logits = all_layers_logits.masked_fill(~mask_expanded, float('-inf'))
                
                max_logit = masked_logits.max(dim=-1).values
                logsumexp = torch.logsumexp(masked_logits, dim=-1)
                softmax_confidence = torch.exp(max_logit - logsumexp)   # shape (len(exit_layers), batch_size, sequence_length)
                # Set softmax confidence to 0 for masked tokens (handles any NaN from -inf operations)
                softmax_confidence = softmax_confidence.masked_fill(~token_mask.unsqueeze(0), 0.0)
                # Detach softmax_confidence from the computational graph. It's only used to weigh the loss.
                softmax_confidence = softmax_confidence.detach()

                # Convert to exit probabilities at each layer for each batch element and token
                softmax_threshold = model.config.ee_softmax_threshold if isinstance(model.config.ee_softmax_threshold, float) else torch.tensor(model.config.ee_softmax_threshold, dtype=torch.bfloat16).unsqueeze(1).unsqueeze(2).to(device)
                q = F.sigmoid((softmax_confidence - softmax_threshold)/config_params.get("prob_weights_temperature", 1)).to(dtype=torch.bfloat16)    # probability to quit at each layer if not already quit before
                if exit_layers[-1] == model.config.num_hidden_layers-1:
                    # The model will always exit at the final layer if it didn't exit before.
                    # q[-1,:,:] = 1
                    # But assuming that gives way to high probability weights for the final layer. 
                    # Therefore
                    pass
                else:
                    logger.warning("The last exit layer is not the model's final layer. The loss computation will assume that the model can also exit at the final layer, which is not included in the loss computation.")

                s = torch.ones(q.shape[1:], device=device)    # probability to stay until the layer iterated over # probability to quit at each layer if not already quit before
                for exit_layer in range(num_exit_layers):
                    q_exit_layer = q[exit_layer].clone()
                    q[exit_layer] = q[exit_layer] * s   # probability to exit exactly at exit_layer (and not before)
                    s = s * (1-q_exit_layer)   # update probability to stay until the next layer

            losses = torch.empty((0,all_layers_logits.shape[1]), device=device) # shape (0, batch_size,) to grow to (num_exit_layers, batch_size)
            for exit_layer in range(num_exit_layers):
                assert labels.shape == outputs.all_layers_logits[exit_layer][:,:,0].shape
                loss_batch_token = causal_lm_ce_loss(logits=outputs.all_layers_logits[exit_layer], labels=labels, vocab_size=model.config.vocab_size, reduction="none") # shape: (batch_size, sequence_length)
                batch_token_weights = q[exit_layer] / torch.sum(q, dim=0)   # Normalize the exit probabilities across layers to get weights for the loss of each layer. Shape (batch_size, sequence_length)
                loss_batch = (loss_batch_token*batch_token_weights).sum(dim=-1) # sum over tokens to get the loss for each batch element, weighted by the probability of exiting at this layer
                losses = torch.cat([losses, loss_batch[None]])  # shape (exit_layer+1, batch_size)

                if not config_params or config_params.get("ignore_completion_only_setting_for_entropy", True):
                    valid_logits = outputs.all_layers_logits[exit_layer]
                    logger.warning("ignore_completion_only_setting_for_entropy is set to True. This is not recommended anymore.")
                else:
                    token_mask = (labels != -100)
                    valid_logits = outputs.all_layers_logits[exit_layer][token_mask]
                entropy_loss = Categorical(logits=valid_logits).entropy().mean()    # Compute the entropy for each token over the vocabulary and average over all sequences and batches
                entropy_losses = torch.cat([entropy_losses, entropy_loss[None]])

            
            entropy_weights = weights_uniform if config_params.get("uniform_entropy_weights", None) else weights_increasing

            ce_loss = losses.sum(dim=0).mean()   # Sum over all exit layers and average over batch elements
            entropy_loss = torch.sum(entropy_losses*entropy_weights).to(device)
        
        elif config_params and config_params.get("difficulty_aware_loss", False):
            # Shape the cross entropy loss with a polynomial term, that encourages
            # easy tokens to exit early and difficult tokens to exit late.

            shaping_fct_config = os.environ.get("SHAPING_FCT_CONFIG", None)
            if shaping_fct_config is None:
                # Throw error
                raise ValueError("SHAPING_FCT_CONFIG not set.")

            elif shaping_fct_config=='0':
                # (t,m)=(0.8, 0.4)
                # (eps,n)=(0.01,16)
                a = 15.4056
                b = -14.6074
                c = 3.619

            elif shaping_fct_config=='1':
                # (t,m)=(0.5, 0.1)
                # (eps,n)=(0.01,2)
                a = 2.0575
                b = -1.6412
                c = 0.4505

            elif shaping_fct_config=='2':
                # (t,m)=(0.9, 0.1)
                # (eps,n)=(0.01,66)
                a = 28.1414
                b = -40.6453
                c = 14.7354

            elif shaping_fct_config=='3':
                # (t,m)=(0.7, 0.0)
                # (eps,n)=(0.01,4.6)
                a = 2.0980
                b = -2.9373
                c = 1.0280

            elif shaping_fct_config=='4':
                # (t,m)=(0.5, 0.5)
                # (eps,n)=(0.01,2)
                a = 3.0521
                b = -0.9708
                c = 0.4437

            elif shaping_fct_config=='5':
                # (t,m)=(0.5, 1.4)
                # (eps,n)=(0.01,4.6)
                a = 7.6416
                b = -1.8137
                c = 1.0163

            elif shaping_fct_config=='6':
                # (t,m)=(0.6, 3.0)
                # (eps,n)=(0.01,4.6)
                a = 18.4751
                b = -3.0089
                c = 1.0271

            elif shaping_fct_config=='7':
                # (t,m)=(0.8, 0.6)
                # (eps,n)=(0.01,20)
                a = 21.7166
                b = -19.6843
                c = 4.5376

            elif shaping_fct_config=='8':
                # (t,m)=(0.6, 1.7)
                a = 11.7127
                b = -3.1972
                c = 1.0297

            elif shaping_fct_config=='9':
                # (t,m)=(0.5, 1.0)
                # (eps,n)=(0.01,4.6)
                a = 6.6469
                b = -2.4842
                c = 1.0231

            elif shaping_fct_config=='10':
                # (t,m)=(0.5, 2.0)
                # (eps,n)=(0.01,9)
                a = 13.1129
                b = -4.7875
                c = 2.0009

            elif shaping_fct_config=='11':
                # (t,m)=(0.5, 3.0)
                # (eps,n)=(0.01,14)
                a = 20.1216
                b = -7.6334
                c = 3.1144

            elif shaping_fct_config=='12':
                # (t,m)=(0.4, 2.0)
                # (eps,n)=(0.01,4.6)
                a = 7.4867
                b = -0.0341
                c = 0.9985

            elif shaping_fct_config=='13':
                # (t,m)=(0.6, 2.0)
                # (eps,n)=(0.01,30)
                a = 29.118
                b = -22.1674
                c = 6.7332

            elif shaping_fct_config=='14':
                # (t,m)=(0.5, 1.5)
                # (eps,n)=(0.01,7)
                a = 10.0608
                b = -3.8167
                c = 1.5572

            elif shaping_fct_config=='15':
                # (t,m)=(0.7, 0.2)
                # (eps,n)=(0.01,4.6)
                a = 4.1752
                b = -3.5994
                c = 1.0345

            elif shaping_fct_config=='16':
                # (t,m)=(0.7, 0.6)
                # (eps,n)=(0.01, 14)
                a = 12.6167
                b = -10.9258
                c = 3.1481

            elif shaping_fct_config=='17':
                # (t,m)=(0.7, 1.0)
                # (eps,n)=(0.01, 16)
                a = 17.6832
                b = -13.5271
                c = 3.6079

            # # Shaping function for points (t,m) and (eps, n):
            # # t = 0.8      # -ln(x) has point (0.8, 0.2)
            # # m = 0.2      # leads to curve that is steeper towards 1 for tokens that are at the exit theshold (about 0.93) than ln(x)
            # # eps = 0.1    # -ln(x) has point (0.1, 2.3)
            # # n = 4.6
            # # c = (m*s*(-s*(1+1/ln(t)) + t*(2+1/ln(t)))-n/ln(s)*ln(t)*(t^2)) / (((s-t)^2) * ln(t))
            # # a = ((-c-n/ln(s))*(1+ln(t)) + (s*c)/t) / (s * (s*(1+ln(t))-t*(1+2*ln(t))))
            # # b = (a*t*(1+2*ln(t)) + c/t) / (-1-ln(t))
            # a = 12.4837
            # b = -6.2478
            # c = 1.0601

            def shaping(x, layer):
                p = ((model.config.num_hidden_layers-1) - layer) / (model.config.num_hidden_layers-1 - exit_layers[0])
                return (a*(x**2) + b*x + c)**p

            # Compute CE only on valid shifted labels to avoid work on ignored tokens.
            shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous().to(device)
            valid_mask = shift_labels != -100

            for exit_layer in range(num_exit_layers):
                assert labels.shape == outputs.all_layers_logits[exit_layer][:,:,0].shape
                logits_exit = outputs.all_layers_logits[exit_layer]  # shape: (batch_size, sequence_length, vocab_size)

                # Boolean indexing over (batch_size, sequence_length) flattens token positions:
                # logits_exit[valid_mask] -> (N_valid_tokens, vocab_size), shift_labels[valid_mask] -> (N_valid_tokens,)
                loss_batch_token = F.cross_entropy(logits_exit[valid_mask], shift_labels[valid_mask], reduction="none")  # shape: (N_valid_tokens,)
                out_probs = torch.exp(-loss_batch_token)  # shape: (N_valid_tokens,)
                shaped_loss_batch_token = loss_batch_token * shaping(out_probs, exit_layers[exit_layer])  # shape: (N_valid_tokens,)
                loss = shaped_loss_batch_token.mean()
                losses = torch.cat([losses, loss[None]])

                if entropy_coefficient>0:
                    if not config_params or config_params.get("ignore_completion_only_setting_for_entropy", True):
                        valid_logits = logits_exit
                        logger.warning("ignore_completion_only_setting_for_entropy is set to True. This is not recommended anymore.")
                    else:
                        token_mask = (labels != -100)
                        valid_logits = logits_exit[token_mask]
                    entropy_loss = Categorical(logits=valid_logits).entropy().mean()    # Compute the entropy for each token over the vocabulary and average over all sequences and batches
                    entropy_losses = torch.cat([entropy_losses, entropy_loss[None]])
                else:
                    entropy_losses = torch.cat([entropy_losses, torch.tensor(0.0, device=device)[None]])

            assert losses.shape==(num_exit_layers,)

            uniform_weights = False if not (config_params and "uniform_weights" in config_params) else config_params["uniform_weights"]

            weights = weights_uniform if uniform_weights else weights_increasing
            if config_params and config_params.get("uniform_entropy_weights", None) != None:
                entropy_weights = weights_uniform if config_params["uniform_entropy_weights"] else weights_increasing
            else:
                entropy_weights = weights

            ce_loss = torch.sum(losses*weights).to(device)
            entropy_loss = torch.sum(entropy_losses*entropy_weights).to(device)

                
        else:
            # Per-exit-layer weights

            for exit_layer in range(num_exit_layers):
                assert labels.shape == outputs.all_layers_logits[exit_layer][:,:,0].shape
                loss = model.loss_function(logits=outputs.all_layers_logits[exit_layer], labels=labels, vocab_size=model.config.vocab_size) # The loss function is assigned in the PretrainedModel here: https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_utils.py#L5756
                losses = torch.cat([losses, loss[None]])

                if entropy_coefficient>0:
                    if not config_params or config_params.get("ignore_completion_only_setting_for_entropy", True):
                        valid_logits = outputs.all_layers_logits[exit_layer]
                        logger.warning("ignore_completion_only_setting_for_entropy is set to True. This is not recommended anymore.")
                    else:
                        token_mask = (labels != -100)
                        valid_logits = outputs.all_layers_logits[exit_layer][token_mask]
                    entropy_loss = Categorical(logits=valid_logits).entropy().mean()    # Compute the entropy for each token over the vocabulary and average over all sequences and batches
                    entropy_losses = torch.cat([entropy_losses, entropy_loss[None]])
                else:
                    entropy_losses = torch.cat([entropy_losses, torch.tensor(0.0, device=device)[None]])

            assert losses.shape==(num_exit_layers,)

            uniform_weights = False if not (config_params and "uniform_weights" in config_params) else config_params["uniform_weights"]

            weights = weights_uniform if uniform_weights else weights_increasing
            if config_params and config_params.get("uniform_entropy_weights", None) != None:
                entropy_weights = weights_uniform if config_params["uniform_entropy_weights"] else weights_increasing
            else:
                entropy_weights = weights

            ce_loss = torch.sum(losses*weights).to(device)
            entropy_loss = torch.sum(entropy_losses*entropy_weights).to(device)


        if expected_cost_coefficient > 0:
            # Compute probabilities for all exit points to penalize them accordingly

            # Compute the softmax value of the maximum logit for each exit layer, 
            # batch element and token
            # For memory efficient computation, we do softmax(logits)[max_index] = e^max_logit/sum(e^logits) = exp(log( e^max_logit/sum(e^logits) )) = exp( max_logit - log(sum(e^logits)) )
            # Mask tokens early to reduce computation on padded/discarded tokens
            token_mask = (labels != -100)  # shape (batch_size, sequence_length)
            mask_expanded = token_mask.unsqueeze(0).unsqueeze(-1)  # shape (1, batch_size, sequence_length, 1) for broadcasting
            masked_logits = outputs.all_layers_logits.masked_fill(~mask_expanded, float('-inf'))
            
            max_logit = masked_logits.max(dim=-1).values
            logsumexp = torch.logsumexp(masked_logits, dim=-1)
            softmax_confidence = torch.exp(max_logit - logsumexp)   # shape (len(exit_layers), batch_size, sequence_length)
            # Set softmax confidence to 0 for masked tokens (handles any NaN from -inf operations)
            softmax_confidence = softmax_confidence.masked_fill(~token_mask.unsqueeze(0), 0.0)

            if config_params and config_params.get("prob_weights_fct", None) == "linear":
                q = softmax_confidence.to(dtype=torch.bfloat16)    # probability to quit at each layer if not already quit before
            else:
                # Convert to exit probabilities at each layer for each batch element and token
                softmax_threshold = model.config.ee_softmax_threshold if isinstance(model.config.ee_softmax_threshold, float) else torch.tensor(model.config.ee_softmax_threshold, dtype=torch.bfloat16).unsqueeze(1).unsqueeze(2).to(device)
                q = F.sigmoid((softmax_confidence - softmax_threshold)/config_params.get("prob_weights_temperature", 1)).to(dtype=torch.bfloat16)    # probability to quit at each layer if not already quit before

            if exit_layers[-1] == model.config.num_hidden_layers-1:
                # The model will always exit at the final layer if it didn't exit before.
                q[-1,:,:] = 1
                # # But assuming that gives way to high probability weights for the final layer. 
                # # Therefore
                # pass
            else:
                logger.warning("The last exit layer is not the model's final layer. The loss computation will assume that the model can also exit at the final layer, which is not included in the loss computation.")

            p = torch.empty_like(q)
            for exit_layer in range(num_exit_layers):
                if config_params and config_params.get("expected_cost_includes_exit", True) == False:
                    p[exit_layer] = (1-q[:exit_layer]).prod(dim=0)   # probability to not exit before exit_layer: product of all previous layer l' 1-q[l']
                else:
                    p[exit_layer] = q[exit_layer] * (1-q[:exit_layer]).prod(dim=0)   # probability to exit exactly at exit_layer (and not before): the probability to exit at this layer (q[€xit_layer]) and to not exit at any of the previous layers (product of all previous layer l' 1-q[l'])

            # Multiply the probability to exit each token of each batch at/after each exit layer with the cost of computing the model until that layer.
            if config_params and config_params.get("expected_cost_includes_exit", True) == False:
                exit_layers_tensor = torch.tensor(exit_layers, device=device, dtype=torch.bfloat16)
                previous_exit_layers = torch.cat([
                    torch.tensor([-1], device=device, dtype=torch.bfloat16),
                    exit_layers_tensor[:-1],
                ])
                layer_costs = (exit_layers_tensor - previous_exit_layers) / (model.config.num_hidden_layers + 1)   # Shape (len(exit_layers),)
            else:
                layer_costs = (torch.tensor(exit_layers, device=device, dtype=torch.bfloat16)+1) / (model.config.num_hidden_layers+1)   # Shape (len(exit_layers),)
            # Sum over all exit layers and average over all tokens not masked across all sequences and batches.
            # So theoretically, expected_cost = (p*(labels!=-100)*layer_costs).sum(dim=0).sum() / num_items_in_batch
            # But since the layer_cost is equal for all batches and sequences, we can pull the summation over the batches and sequences ahead.
            expected_cost = ((p*token_mask).sum(dim=1).sum(dim=1) * layer_costs).sum() / num_items_in_batch


        total_loss = ce_coefficient*ce_loss + entropy_coefficient*entropy_loss + expected_cost_coefficient*expected_cost
        loss_storage["ce_loss"].append(ce_loss.detach().cpu().item())
        loss_storage["entropy_loss"].append(entropy_loss.detach().cpu().item())
        loss_storage["expected_cost"].append(expected_cost.detach().cpu().item())

        return total_loss

    
    def fine_tune(self):
        """Fine-tunes the model using the SFTTrainer."""
        logger.info(f"Device: {self.trainer.args.device}")

        logger.info("Start Fine-Tuning.")
        self.trainer.train()

        logger.info("Fine-Tuning completed.")

