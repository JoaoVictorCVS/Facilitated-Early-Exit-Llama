"""
Fine-tune EELlama on SST-2 sentiment classification.

Uses the same ModelPEFTTuner as fine-tuning.py but supplies a custom dataset
preparation function that formats SST-2 as instruction-following chat pairs:

  User:      Classify the sentiment of the following sentence as 'positive' or
             'negative'.\n\nSentence: {sentence}
  Assistant: positive  (or negative)

Run from the src/ directory:
    python3 fine-tune-sst2.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging

from datasets import load_dataset

from fine_tuning import ModelPEFTTuner, ModelPEFTTunerConfig
import utils.logging_utils  # noqa: F401 – sets up root logging

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "negative", 1: "positive"}

SYSTEM_PROMPT = (
    "Classify the sentiment of the following sentence as 'positive' or 'negative'."
)


def _format_sst2_row(example):
    """Convert an SST-2 row to the chat-message format expected by SFTTrainer."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sentence: {example['sentence'].strip()}"},
            {"role": "assistant", "content": LABEL_MAP[example["label"]]},
        ]
    }


def prepare_sst2_datasets(config):
    """Load SST-2 parquet files and return (train_dataset, eval_dataset)."""
    train_path = config["dataset"]["train_path"]
    eval_path = config["dataset"]["eval_path"]

    logger.info("Loading SST-2 train split from %s", train_path)
    train_dataset = load_dataset("parquet", data_files={"train": train_path}, split="train")

    logger.info("Loading SST-2 validation split from %s", eval_path)
    eval_dataset = load_dataset("parquet", data_files={"validation": eval_path}, split="validation")

    select_range = config["dataset"].get("select_range")
    if select_range is not None:
        train_dataset = train_dataset.select(range(select_range[0], select_range[1]))

    train_dataset = train_dataset.shuffle(seed=42)
    train_dataset = train_dataset.map(_format_sst2_row, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(_format_sst2_row, remove_columns=eval_dataset.column_names)

    logger.info("SST-2 train size: %d, eval size: %d", len(train_dataset), len(eval_dataset))
    return train_dataset, eval_dataset


if __name__ == "__main__":
    config = ModelPEFTTunerConfig.load(
        "configs/full_model_fine-tuning/eellama-3p2-1B-layerskip-adapter-sst2.yaml"
    )
    logger.info("Using configuration:\n%s", config.to_string())

    fine_tuner = ModelPEFTTuner(config, prepare_datasets_func=prepare_sst2_datasets)
    fine_tuner.fine_tune()
