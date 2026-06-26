"""
Fine-tuning script for EELlamma on SST-2 (binary sentiment classification).

SST-2 is framed as a generation task: the model is given an instruction prompt
containing the review sentence and must generate a single token — "positive" or
"negative". All EELlamma exit layers are trained jointly with a weighted sum of
cross-entropy losses at each exit layer (same recipe as fine-tuning.py).

SkipDecode inference parameters can be specified in the YAML config and are stored
in the saved model config so that SkipDecode can be activated at inference time by
simply loading the saved checkpoint with skipdecode_enabled=True.

Usage
-----
    cd src/
    python fine_tune_sst2.py
    python fine_tune_sst2.py --config configs/full_model_fine-tuning/eellama-skipdecode-sst2.yaml
"""

import argparse
import logging
import os
import time
from functools import partial
from typing import Any, Dict, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from transformers.integrations import TensorBoardCallback
from trl import SFTConfig, SFTTrainer

from transformers_override.models.llama.configuration_eellama import EeLlamaConfig
from transformers_override.models.llama.modeling_eellama import EeLlamaForCausalLM
from utils.configuration_utils import GenericConfig
import utils.logging_utils  # noqa: F401 — sets up logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a sentiment classifier. "
    "Reply with exactly one word: 'positive' or 'negative'."
)

def make_prompt(sentence: str) -> str:
    return (
        f"Classify the sentiment of the following movie review.\n\n"
        f"Review: {sentence}\n"
        f"Sentiment:"
    )

LABEL_WORDS = {0: "negative", 1: "positive"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class SST2FineTunerConfig(GenericConfig):
    """Configuration class for SST-2 fine-tuning."""

    def __init__(self):
        config: Dict[str, Any] = {
            "output_path": "../models/eellama-skipdecode-sst2",
            "model": {
                "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                "copy_chat_template_model_id": None,
                "exit_layers": [3, 7, 11, 15],
                # SkipDecode inference parameters (stored in model config at save time)
                "skipdecode_enabled": False,
                "skipdecode_min_exit_layer": 5,
                "skipdecode_max_exit_layer": 11,
                "skipdecode_num_warmup_layers": 1,
                "skipdecode_max_sequence_length": 128,
                "skipdecode_prompt_size": 40,
            },
            "lora_config_args": {
                "r": 8,
                "lora_alpha": 16,
                "bias": "none",
                "lora_dropout": 0.05,
                "task_type": "CAUSAL_LM",
                "target_modules": ["q_proj", "v_proj", "down_proj"],
            },
            "sft_config_args": {},
            "dataset": {
                "select_range": None,
                "fraction_train": 1.0,
            },
            "loss_config_params": {
                "uniform_weights": True,
            },
        }
        super().__init__(config)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_sst2_datasets(config: SST2FineTunerConfig, tokenizer: AutoTokenizer):
    """Load SST-2 from the HuggingFace Hub and format for SFTTrainer.

    Each example is converted to a list of chat messages so that the SFTTrainer
    can apply the model's chat template automatically. The assistant turn contains
    only the label word ("positive" or "negative").

    Parameters
    ----------
    config : SST2FineTunerConfig
    tokenizer : AutoTokenizer
        Used only to verify the label tokens exist in the vocabulary.

    Returns
    -------
    train_dataset, eval_dataset : Dataset
    label_token_ids : dict
        Maps label word → token id, used later for accuracy computation.
    """
    raw = load_dataset("glue", "sst2")

    # Verify the label words are single tokens so accuracy computation is exact.
    label_token_ids = {}
    for word in LABEL_WORDS.values():
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) != 1:
            logger.warning(
                "Label word '%s' tokenises to %d tokens — accuracy may be approximate.",
                word, len(ids),
            )
        label_token_ids[word] = ids[0]
    logger.info("Label token ids: %s", label_token_ids)

    def format_example(example):
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": make_prompt(example["sentence"])},
                {"role": "assistant", "content": LABEL_WORDS[example["label"]]},
            ]
        }

    train_split = raw["train"]
    if config["dataset"]["select_range"] is not None:
        lo, hi = config["dataset"]["select_range"]
        train_split = train_split.select(range(lo, hi))

    train_dataset = train_split.map(format_example, remove_columns=train_split.column_names)

    # SST-2 official validation split ("validation" key in glue/sst2).
    eval_dataset = raw["validation"].map(format_example, remove_columns=raw["validation"].column_names)

    logger.info("Train size: %d  |  Eval size: %d", len(train_dataset), len(eval_dataset))
    return train_dataset, eval_dataset, label_token_ids


# ---------------------------------------------------------------------------
# Fine-tuner
# ---------------------------------------------------------------------------

class SST2FineTuner:
    """Fine-tunes an EELlamma model on SST-2 sentiment classification."""

    def __init__(self, config: SST2FineTunerConfig):
        self.config = config
        self.model, self.tokenizer = self._init_model()
        self.train_dataset, self.eval_dataset, self.label_token_ids = \
            prepare_sst2_datasets(config, self.tokenizer)

        sft_args: Dict[str, Any] = {
            "gradient_checkpointing": False,
            "gradient_accumulation_steps": 2,
            "per_device_train_batch_size": 8,
            "auto_find_batch_size": True,

            "max_length": 128,
            "packing": False,

            "num_train_epochs": 3,
            "learning_rate": 2e-4,

            "logging_steps": 50,
            "output_dir": config["output_path"],
            "report_to": "none",

            "bf16": torch.cuda.is_bf16_supported(including_emulation=False),
            "push_to_hub": False,
            "padding_free": False,
            "eval_strategy": "steps",
            "eval_steps": 200,
            "save_strategy": "best",
            "save_total_limit": 2,
            "metric_for_best_model": "eval_loss",
            "load_best_model_at_end": True,
            "eval_on_start": True,

            "seed": 42,
        }
        for key, val in config["sft_config_args"].items():
            sft_args[key] = val
        self.sft_config = SFTConfig(**sft_args)

        timestamp = int(time.time())
        tb_dir = f"{config['output_path']}/tensorboard_logs/{timestamp}"
        self.tb_writer = SummaryWriter(log_dir=tb_dir)
        os.environ["TENSORBOARD_LOGGING_DIR"] = tb_dir

        self.lora_config = LoraConfig(**config["lora_config_args"])

        self.trainer = SFTTrainer(
            model=self.model,
            peft_config=self.lora_config,
            processing_class=self.tokenizer,
            args=self.sft_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_loss_func=partial(
                self._compute_loss,
                self.model,
                config_params=config.get("loss_config_params"),
            ),
            callbacks=[TensorBoardCallback(self.tb_writer)],
        )

    def _init_model(self):
        """Load the pretrained EELlamma model and tokenizer."""
        model_cfg = self.config["model"]
        model_id = model_cfg["model_id"]
        skipdecode_enabled = model_cfg.get("skipdecode_enabled", False)

        # When SkipDecode is enabled, output_full_model must be False so that the
        # SkipDecode path in EELlamaModel.forward() is reached during training.
        # When it is disabled, output_full_model=True ensures all exit layers are
        # computed and their logits accumulated for the multi-exit loss.
        output_full_model = not skipdecode_enabled

        model = EeLlamaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=0,
            # EELlamma early-exit parameters
            exit_layers=model_cfg["exit_layers"],
            untied_heads=False,
            output_full_model=output_full_model,
            # SkipDecode parameters
            skipdecode_enabled=skipdecode_enabled,
            skipdecode_min_exit_layer=model_cfg.get("skipdecode_min_exit_layer", 0),
            skipdecode_max_exit_layer=model_cfg.get("skipdecode_max_exit_layer", None),
            skipdecode_num_warmup_layers=model_cfg.get("skipdecode_num_warmup_layers", 1),
            skipdecode_max_sequence_length=model_cfg.get("skipdecode_max_sequence_length", 128),
            skipdecode_prompt_size=model_cfg.get("skipdecode_prompt_size", 0),
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if model_cfg.get("copy_chat_template_model_id"):
            src_tok = AutoTokenizer.from_pretrained(model_cfg["copy_chat_template_model_id"])
            chat_template = src_tok.chat_template.replace(
                "<|eot_id|>", tokenizer.decode(model.config.eos_token_id)
            )
            tokenizer.chat_template = chat_template

        return model, tokenizer

    @staticmethod
    def _compute_loss(model, outputs, labels, num_items_in_batch=None, config_params=None):
        """Cross-entropy loss, adapted for both SkipDecode and standard EELlamma training.

        SkipDecode mode (skipdecode_enabled=True)
        -----------------------------------------
        The SkipDecode forward path skips intermediate layers entirely, so
        ``all_layers_logits`` is empty. In this case a single cross-entropy loss is
        computed from ``outputs.logits`` (the top-layer output). This directly trains
        the model to produce correct predictions from the layer subset that SkipDecode
        would use at inference time.

        Standard EELlamma mode (skipdecode_enabled=False)
        --------------------------------------------------
        All exit-layer logits are available in ``all_layers_logits``. Each exit layer
        contributes one cross-entropy loss and the results are combined with either
        uniform weights or depth-proportional weights (controlled by config_params).

        Parameters
        ----------
        model : EeLlamaForCausalLM
        outputs : CausalLMOutputWithPastAndEeLogits
        labels : torch.Tensor  shape (batch, seq_len)
        num_items_in_batch : int, optional
        config_params : dict, optional
            Supports ``uniform_weights`` (bool, default True).
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # SkipDecode path: all_layers_logits is empty because the forward pass
        # returns early before any intermediate exit-layer logits are collected.
        if model.config.skipdecode_enabled:
            return model.loss_function(
                logits=outputs.logits,
                labels=labels,
                vocab_size=model.config.vocab_size,
            )

        # Standard multi-exit EELlamma path.
        exit_layers = model.config.exit_layers
        num_exit_layers = len(exit_layers)
        assert num_exit_layers == outputs.all_layers_logits.shape[0], (
            f"Expected {num_exit_layers} exit layers in all_layers_logits, "
            f"got {outputs.all_layers_logits.shape[0]}"
        )

        losses = torch.empty((0,), device=device)
        for i in range(num_exit_layers):
            loss = model.loss_function(
                logits=outputs.all_layers_logits[i],
                labels=labels,
                vocab_size=model.config.vocab_size,
            )
            losses = torch.cat([losses, loss[None]])

        uniform = True
        if config_params and "uniform_weights" in config_params:
            uniform = config_params["uniform_weights"]

        if uniform:
            weights = torch.full((num_exit_layers,), 1.0 / num_exit_layers, device=device)
        else:
            layer_indices = torch.tensor(exit_layers, dtype=torch.float, device=device)
            weights = layer_indices / layer_indices.sum()

        return (losses * weights).sum()

    def evaluate_accuracy(self, split: str = "validation") -> float:
        """Compute classification accuracy on the requested split.

        For each example the model scores both label tokens at the last position
        of the prompt (before the answer) and picks the higher-scoring one.

        Parameters
        ----------
        split : str
            "validation" (default) uses self.eval_dataset.

        Returns
        -------
        float
            Accuracy in [0, 1].
        """
        dataset = self.eval_dataset if split == "validation" else self.train_dataset
        model = self.trainer.model
        model.eval()

        pos_id = self.label_token_ids["positive"]
        neg_id = self.label_token_ids["negative"]

        correct = 0
        total = 0

        with torch.inference_mode():
            for example in dataset:
                messages = example["messages"]
                # Re-derive the ground-truth label from the assistant turn.
                gt_word = messages[-1]["content"].strip().lower()
                gt_id = self.label_token_ids.get(gt_word)
                if gt_id is None:
                    continue

                # Build prompt WITHOUT the assistant answer.
                prompt_messages = [m for m in messages if m["role"] != "assistant"]
                prompt = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

                outputs = model(**inputs, logits_to_keep=1)
                # outputs.logits shape: (1, 1, vocab_size)
                last_logits = outputs.logits[0, -1]
                pred_id = pos_id if last_logits[pos_id] > last_logits[neg_id] else neg_id

                correct += int(pred_id == gt_id)
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        logger.info("Accuracy on %s: %.4f  (%d / %d)", split, accuracy, correct, total)
        return accuracy

    def fine_tune(self):
        """Run fine-tuning then evaluate accuracy on the validation set."""
        logger.info("Device: %s", self.trainer.args.device)
        logger.info("Starting SST-2 fine-tuning.")
        self.trainer.train()

        logger.info("Fine-tuning complete. Saving final adapter.")
        self.trainer.save_model(f"{self.config['output_path']}-final")

        logger.info("Evaluating accuracy on validation set.")
        accuracy = self.evaluate_accuracy("validation")
        self.tb_writer.add_scalar("accuracy/validation", accuracy)
        self.tb_writer.close()
        return accuracy


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune EELlamma on SST-2")
    parser.add_argument(
        "--config",
        default="configs/full_model_fine-tuning/eellama-skipdecode-sst2.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = SST2FineTunerConfig.load(args.config)
    logger.info("Configuration:\n%s", config.to_string())

    fine_tuner = SST2FineTuner(config)
    final_accuracy = fine_tuner.fine_tune()
    print(f"\nFinal validation accuracy: {final_accuracy:.4f}")
