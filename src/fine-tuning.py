# Code based on https://huggingface.co/blog/dvgodoy/fine-tuning-llm-hugging-face
# and https://github.com/benyaminjami/Balcony-LLaMA/tree/finetuning 

"""
Script for fine-tuning an EELlama model with PEFT using a standard recipe.

Setting CREATE_DATASET to True will create a dataset as subset of the CNN/DailyMail
dataset.
The EELlama model to be fine-tuned and the fine-tuning procedure can be configured 
by creating a ModelPEFTTunerConfig from a yaml file (see utils/configuration_utils.py).
"""

# Before execution, execute `export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'`
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
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset, load_from_disk
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers.integrations import TensorBoardCallback
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from transformers_override.models.llama.configuration_eellama import EeLlamaConfig
from transformers_override.models.llama.modeling_eellama import EeLlamaForCausalLM

from utils.configuration_utils import GenericConfig
import utils.logging_utils


logger = logging.getLogger(__name__)

CREATE_DATASET = False
if CREATE_DATASET:
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")
    # Reduce to only short instances of the dataset
    print("Filtering dataset")
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = dataset.filter(lambda row: len(tokenizer.encode(row['article']+row['highlights']))<400)
    dataset.save_to_disk("../datasets/cnn-dm_short")
    # dataset = dataset.filter(lambda row: len(tokenizer.encode(row['article']+row['highlights']))<200)
    # dataset.save_to_disk("../datasets/cnn-dm_very_short")


class ModelPEFTTunerConfig(GenericConfig):
    """Configuration class for the ModelFineTuner."""
    
    def __init__(self):
        """Initialize config with default values."""
        config: Dict[str, Any] = {
            "output_path": "../models/test",
            "model": {
                "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
                "copy_chat_template_model_id": None,
                "exit_layers": [7, 15, 23, 31],
            },
            "lora_config_args": {
                "r": 8,                   
                "lora_alpha": 16,
                "bias": "none",           
                "lora_dropout": 0.1,
                "task_type": "CAUSAL_LM",
                "target_modules": ['q_proj', 'v_proj', 'down_proj'],
            },
            "sft_config_args": {},
            "dataset": {
                "train_path": "../datasets/cnn-dm_short",
                "eval_path": "../datasets/cnn-dm_validation_short-shuffled32",
                "select_range": None,  # e.g., [0,100] to select only first 100 samples
                "fraction_train": 1.0,
            },
            # "loss_config_params": {
            #     "uniform_weights": False
            # }
        }
        super().__init__(config)

class ModelPEFTTuner:
    """Class for fine-tuning a LLaMA model with PEFT for Early Exit."""

    class ConfigChangingSFTTrainer(SFTTrainer):
        # Could be used to train only a subset of layers as exit layer per training step.
        pass

    def __init__(self, config: ModelPEFTTunerConfig, post_init_model_func=None, prepare_datasets_func=None):
        """Initialize the ModelPEFTTuner for an EELlama model.

        Parameters
        ----------
        config : ModelPEFTTunerConfig
            The configuration for the ModelPEFTTuner.
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

            "num_train_epochs": 4,
            "learning_rate": 8e-4,
            
            "logging_steps": 20,
            # "logging_dir": './runs',  # logging_dir is deprecated
            "output_dir": self.config["output_path"],
            "report_to": "none", # We add a TensorBoardCallback below instead to allow for further use of the tensorboard writer or for modifying logging.

            "bf16": torch.cuda.is_bf16_supported(including_emulation=False),
            "push_to_hub": False,
            "padding_free": False,
            "eval_strategy": "steps",
            "save_strategy": "best",
            "save_total_limit": 1,
            "metric_for_best_model": "eval_loss",
            "load_best_model_at_end": True,
            "eval_on_start": True,

            "seed": 42,
            # "full_determinism": True,
        }
        # Override/add any args from config
        for key, val in self.config["sft_config_args"].items():
            sft_config_args[key] = val
        self.sft_config = SFTConfig(**sft_config_args)

        timestamp = int(time.time())
        logger.info(f"Timestamp used for output naming: {timestamp}")

        tensorboard_logging_dir = f"{self.config['output_path']}/tensorboard_logs/{timestamp}"
        self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_logging_dir)
        # Set directory for reporting to TensorBoard
        os.environ["TENSORBOARD_LOGGING_DIR"] = tensorboard_logging_dir
        # See https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/integrations/integration_utils.py#L628

        self.loss_storage = {}

        self.lora_config = LoraConfig(**self.config["lora_config_args"])

        # For debugging:
        # model = get_peft_model(self.model, self.lora_config)
        # print(model)
        # import pdb; pdb.set_trace()

        self.trainer = self.ConfigChangingSFTTrainer(
            model=self.model,
            peft_config=self.lora_config,
            processing_class=self.tokenizer,
            args=self.sft_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_loss_func=partial(self.compute_loss, self.model, self.loss_storage, config_params=self.config.get("loss_config_params", None)),
            callbacks=[TensorBoardCallback(self.tensorboard_writer)], 
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

        exit_layers = self.config["model"]["exit_layers"]
        model = EeLlamaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=0,

            exit_layers=exit_layers,
            untied_heads=False,
            output_full_model=True
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.eos_token_id = model.config.eos_token_id

        if self.config["model"]["copy_chat_template_model_id"] != None:
            # Add chat template to non-chat model
            chat_template = AutoTokenizer.from_pretrained(self.config["model"]["copy_chat_template_model_id"]).chat_template
            chat_template = chat_template.replace("<|eot_id|>", tokenizer.decode(model.config.eos_token_id))  # Ensure that the chat template uses the correct eos token id
            tokenizer.chat_template = chat_template

        if model.model.config.untied_heads:
            # Initialize all lm heads to the same values as the final layers head
            for h in range(len(model.lm_heads)-1):
                model.lm_heads[h].load_state_dict(model.lm_head.state_dict(), assign=True)

        if "embed_tokens" in self.config["lora_config_args"]["target_modules"] or "lm_head" in self.config["lora_config_args"]["target_modules"]:
            model.model.config.tie_word_embeddings = False
            logger.warning("Setting model.config.tie_word_embeddings to False since 'embed_tokens' or 'lm_head' is in the target modules for LoRA.")

        if post_init_model_func is not None:
            model, tokenizer = post_init_model_func(model, tokenizer, self.config)

        return model, tokenizer

    def _prepare_cnn_dm_datasets(self):
        """Prepares the CNN/DailyMail dataset given in the config for fine-tuning
            with the SFTTrainer.
        """
        logger.info("Loading dataset from disk")
        dataset = load_from_disk(self.config["dataset"]["train_path"])

        dataset = dataset.shuffle(seed=42)
        if self.config["dataset"]["select_range"] is not None:
            dataset = dataset.select(range(self.config["dataset"]["select_range"][0], self.config["dataset"]["select_range"][1]))

        def format_dataset(example):
            return {"messages": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["completion"]},
            ]}
            
        dataset = dataset.rename_column("article", "prompt")
        dataset = dataset.rename_column("highlights", "completion")
        dataset = dataset.map(format_dataset)
        dataset = dataset.remove_columns(['prompt', 'completion'])
        # The SFTTrainer will apply the chat template. See https://huggingface.co/docs/trl/main/en/sft_trainer#dataset-format-support

        if self.config["dataset"]["eval_path"] is not None:
            train_dataset = dataset
            eval_dataset = load_from_disk(self.config["dataset"]["eval_path"])
            eval_dataset = eval_dataset.rename_column("article", "prompt")
            eval_dataset = eval_dataset.rename_column("highlights", "completion")
            eval_dataset = eval_dataset.map(format_dataset)
            eval_dataset = eval_dataset.remove_columns(['prompt', 'completion'])
            logger.info(f"Length of train dataset: {len(train_dataset)}")
            logger.info(f"Length of eval dataset: {len(eval_dataset)}")
        else:
            fraction_train = self.config["dataset"]["fraction_train"]
            logger.info("Split dataset: #train / #eval = %d / %d", int(len(dataset)*fraction_train), len(dataset) - int(len(dataset)*fraction_train))
            train_dataset = dataset.select(range(0,int(len(dataset)*fraction_train)))
            eval_dataset = dataset.select(range(int(len(dataset)*fraction_train),len(dataset)))

        assert self.tokenizer.eos_token_id != self.tokenizer.pad_token_id
        assert self.model.config.eos_token_id != self.tokenizer.pad_token_id

        return train_dataset, eval_dataset

    @staticmethod
    def compute_loss(model, loss_storage, outputs, labels, num_items_in_batch=None, config_params=None):
        """
        Computation of the model loss.

        We use a weighted sum of cross-entropy losses of all exit layers (incl.
        the final layer). 

        Parameters
        ----------
        model : EeLlamaForCausalLM
            The EELlama model being fine-tuned from the ModelPEFTTuner.
        loss_storage : dict
            A dictionary to store computed sublosses for logging from the ModelPEFTTuner.

        outputs : list of CausalLMOutputWithPast
            Outputs returned by the model at each token.
        labels : torch.Tensor
            The target tokens. Shape (batch_size, sequence_length)
        num_items_in_batch : int
            The number of items in the entire accumulated batch 
            (batch_size * gradient_accumulation_steps)

        config_params : dict, optional
            Additional parameters from the config from the ModelPEFTTuner.

        Notes
        -----
        The elements of outputs are expected to contain a field all_layers_logits
        of shape shape (len(exit_layers), batch_size, sequence_length, vocabulary_size)
        with all the output logits of all exit layers, tokens and batch elements.
        """
        # assert outputs.losses.shape == (num_items_in_batch, len(exit_layers))
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if not outputs.loss==None:
            import warnings
            warnings.warn("Losses shouldn't already be computed to avoid repeated computation")
        exit_layers = model.config.exit_layers
        num_exit_layers = len(exit_layers)
        assert num_exit_layers==outputs.all_layers_logits.shape[0]

        if "stats" in outputs and "exit_layer" in outputs["stats"]:
            assert len(outputs["stats"]["exit_layer"])==1, "This implementation assumes a batch size of 1 during evaluation."
            loss_storage["eval_exit_layers"] += outputs["stats"]["exit_layer"][0]

        # Combine all batches and tokens to a one-dimensional sequence of length N
        losses = torch.empty((0,), device=device)
        for exit_layer in range(num_exit_layers):
            assert labels.shape == outputs.all_layers_logits[exit_layer][:,:,0].shape
            loss = model.loss_function(logits=outputs.all_layers_logits[exit_layer], labels=labels, vocab_size=model.config.vocab_size) # The loss function is assigned in the PretrainedModel here: https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_utils.py#L5756
            losses = torch.cat([losses, loss[None]])
        assert losses.shape==(num_exit_layers,)

        uniform_weights = False if not (config_params and "uniform_weights" in config_params) else config_params["uniform_weights"]
        if uniform_weights:
            weights = torch.tensor([1/len(exit_layers)] * len(exit_layers), device=device)    # uniform variant
        else:
            weights = torch.tensor(exit_layers, dtype=float, device=device) / torch.sum(torch.tensor(exit_layers, device=device))  # i / sum_i (i)
        return torch.sum(losses*weights).to(device)

    def fine_tune(self):
        """Fine-tunes the model using the SFTTrainer."""
        logger.info(f"Device: {self.trainer.args.device}")

        logger.info("Start Fine-Tuning.")
        self.trainer.train()

        logger.info("Fine-Tuning completed.")
        logger.info("Saving final model adapter.")
        self.trainer.save_model(f"{self.config['output_path']}-final")


if __name__ == "__main__":
    mptConfig = ModelPEFTTunerConfig.load("configs/full_model_fine-tuning/eellama-3p2-1B-layerskip-adapter-cnn-dm_short.yaml")
    # mptConfig = ModelPEFTTunerConfig.load("configs/full_model_fine-tuning/eellama-3-8B-instruct-adapter-cnn-dm_short_all_all_exit_layers.yaml")
    logger.info("Using configuration:\n" + mptConfig.to_string())
    fine_tuner = ModelPEFTTuner(mptConfig)
    fine_tuner.fine_tune()

