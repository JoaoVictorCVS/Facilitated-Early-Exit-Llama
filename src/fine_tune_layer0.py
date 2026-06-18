"""
Fine-tune selected parts of an EELlama model, such as layer 0 (working as prompt 
modifier), to facilitate early exit.

For configuring the fine-tuning process, create a yaml config file suitable for 
class ModelFineTunerConfig from partial_model_fine_tuning.py and insert the path
in the main block at the end of this file or specify the path as command line argument 
`--config`. In addition, if using a difficulty_aware_loss, set environment variable
SHAPING_FCT_CONFIG to the desired number of shapings implemented in partial_model_fine_tuning.py.
For training a separate prompt modifier module, use file fine_tune_prompt_modifier.py
instead of this file.

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# can also execute with `CUDA_VISIBLE_DEVICES=0 python3 fine-tuning.py`

import logging
import argparse

import torch
import torch.nn.functional as F
from datasets import load_from_disk

from partial_model_fine_tuning import ModelFineTuner, ModelFineTunerConfig
import utils.logging_utils


logger = logging.getLogger(__name__)

def prepare_datasets(config):
    # We use the dataset with the full model's outputs as targets.
    # Therefore we directly pass the input_ids to the trainer (see https://huggingface.co/docs/trl/v0.27.0/en/sft_trainer#trl.SFTTrainer.train_dataset)
    def format_dataset_ids(instance):
        completion_mask = torch.cat([torch.zeros_like(torch.tensor(instance["input_ids"])), torch.ones_like(torch.tensor(instance["output_ids"]))])
        input_ids = torch.cat([torch.tensor(instance["input_ids"]), torch.tensor(instance["output_ids"])])

        return {
            "input_ids": input_ids,
            "completion_mask": completion_mask
        }
    dataset = load_from_disk(config["dataset"]["path"])
    dataset = dataset.map(format_dataset_ids)
    dataset = dataset.remove_columns(['output_ids'])

    if config["dataset"].get("shuffle_before_select", False):
        logger.info("Shuffling dataset before selecting a subset.")
        dataset = dataset.shuffle(seed=42)

    if config["dataset"]["select_range"] is not None:
        dataset = dataset.select(range(config["dataset"]["select_range"][0], config["dataset"]["select_range"][1]))

    fraction_train = config["dataset"]["fraction_train"]
    logger.info("Split dataset: #train / #eval = %d / %d", int(len(dataset)*fraction_train), len(dataset) - int(len(dataset)*fraction_train))
    train_dataset = dataset.select(range(0,int(len(dataset)*fraction_train)))
    eval_dataset = dataset.select(range(int(len(dataset)*fraction_train),len(dataset)))

    return train_dataset, eval_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune layer 0 of EELlama.")
    parser.add_argument("--config", type=str)
    cli_args = parser.parse_args()
    config_path = cli_args.config if cli_args.config is not None else "configs/fine_tune_layer0_config_eellama-3p2-1B-layerskip_01.yaml"
    mftconfig = ModelFineTunerConfig.load(config_path)
    logger.info("Using configuration:\n" + mftconfig.to_string())
    fine_tuner = ModelFineTuner(mftconfig, prepare_datasets_func=prepare_datasets)
    fine_tuner.fine_tune()