# https://huggingface.co/transformers/v3.0.2/training.html

"""Script for fine-tuning the embedding layer (and the lm_head) of an EELlama model."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# can also execute with `CUDA_VISIBLE_DEVICES=0 python3 fine-tuning.py`

from collections import OrderedDict
from functools import partial
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainerCallback
from peft import PeftModel
from trl import SFTConfig, SFTTrainer

from transformers_override.models.llama.configuration_eellama import EeLlamaConfig
from transformers_override.models.llama.modeling_eellama import EeLlamaForCausalLM

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
output_path = "../models/llama-3-eellama-3-8B-instruct-lora-tuned-embedding"
adapter_path = "../models/llama-3-eellama-3-8B-instruct/llama-3-eellama-3-8B-instruct-adapter-cnn-dm/checkpoint-4830"

tokenizer = AutoTokenizer.from_pretrained(model_id)
exit_layers = [7, 15, 23, 31]
model = EeLlamaForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=0,

    exit_layers=exit_layers,
    untied_heads=False,
    output_full_model=True
)

if model.model.config.untied_heads:
    # Initialize all lm heads to the same values as the final layers head
    for h in range(len(model.lm_heads)-1):
        model.lm_heads[h].load_state_dict(model.lm_head.state_dict(), assign=True)

model = PeftModel.from_pretrained(model, adapter_path)


# print(model)

# Select layers to tune
for name, param in model.named_parameters():
    if "embed_tokens" in name or "lm_head" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
    print(f"Training {name}: {param.requires_grad}")

model.train()


def format_dataset(example):
    return {"messages": [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]}
    
dataset = load_from_disk("../datasets/cnn-dm_short")
dataset = dataset.rename_column("article", "prompt")
dataset = dataset.rename_column("highlights", "completion")
dataset = dataset.map(format_dataset)
dataset = dataset.remove_columns(['prompt', 'completion'])

dataset = dataset.shuffle(seed=42)
fraction_train = 0.98
print("Split dataset: #train / #eval = ", int(len(dataset)*fraction_train), "/", len(dataset) - int(len(dataset)*fraction_train))
train_dataset = dataset.select(range(0,int(len(dataset)*fraction_train)))
eval_dataset = dataset.select(range(int(len(dataset)*fraction_train),len(dataset)))


def compute_loss(outputs, labels, num_items_in_batch=None):
    """
    Computation of the model loss.

    We use a weighted sum of cross-entropy losses of all exit layers (incl.
    the final layer).

    Parameters
    ----------
    outputs : list of CausalLMOutputWithPast
        Outputs returned by the model at each token.
    labels : torch.Tensor
        The target tokens. Shape (batch_size, sequence_length)
    num_items_in_batch : int
        The number of items in the entire accumulated batch 
        (batch_size * gradient_accumulation_steps)

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
    num_exit_layers = len(exit_layers)
    assert num_exit_layers==outputs.all_layers_logits.shape[0]

    # Combine all batches and tokens to a one-dimensional sequence of length N
    losses = torch.empty((0,), device=device)
    for exit_layer in range(num_exit_layers):
        assert labels.shape == outputs.all_layers_logits[exit_layer][:,:,0].shape
        loss = model.loss_function(logits=outputs.all_layers_logits[exit_layer], labels=labels, vocab_size=model.config.vocab_size) # The loss function is assigned in the PretrainedModel here: https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_utils.py#L5756
        losses = torch.cat([losses, loss[None]])
    assert losses.shape==(num_exit_layers,)

    weights = torch.tensor(exit_layers, dtype=float, device=device) / torch.sum(torch.tensor(exit_layers, device=device))  # i / sum_i (i)
    # weights = torch.tensor([1/len(exit_layers)] * len(exit_layers), device=device)    # uniform variant
    return torch.sum(losses*weights).to(device)

class SavePartialModelCallback(TrainerCallback):
    """
    A callback for the SFT Trainer used to save only a certain parameter of the 
    model if a new optimal loss is achieved instead of saving the whole model.
    Particularly, only the embedding layer is fine-tuned, so we only want to save
    the embedding layer.
    """
    def __init__(self, keys_to_save, save_dir):
        self.keys_to_save = keys_to_save
        if save_dir[-1] != "/": save_dir += "/"
        self.save_dir = save_dir
        self.best_metric = None
        self.start_timestamp = int(time.time())


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
            print(f"New optimum! Saved {self.keys_to_save} at {outfile}")

            # Prevent the trainer from saving the whole checkpoint
            control.should_save = False

sft_config = SFTConfig(
    gradient_checkpointing=False,
    gradient_accumulation_steps=4,  
    per_device_train_batch_size=4, 
    auto_find_batch_size=True,

    max_length=512,     # the maximum length of prompt+completion in cnn/dm is 16336 (check reduce(lambda m, item: max(m, item), dataset.map(lambda item: {'length': len(item['highlights']) + len(item['article'])})['length'])) . However, the average length is about 1000. We reduced the set to only short instances.
    packing=False,

    num_train_epochs=3,
    learning_rate=1e-5,
    
    logging_steps=20,
    logging_dir='./logs',
    output_dir=output_path,
    report_to='none',

    bf16=torch.cuda.is_bf16_supported(including_emulation=False),
    push_to_hub=False,
    padding_free=False,
    eval_strategy="steps",
    save_strategy="no",
    save_total_limit=1,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=False,
    eval_on_start=True,

    seed=42,
    # full_determinism=True,
)

spm_callback = SavePartialModelCallback(
    keys_to_save=["base_model.model.model.embed_tokens.weight", "base_model.model.lm_head.weight"],
    save_dir=output_path
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_loss_func=compute_loss,
    callbacks=[spm_callback],
)

print("Device:")
print(trainer.args.device)

print("Start Fine-Tuning.")
trainer.train()


print("Fine-Tuning completed.")