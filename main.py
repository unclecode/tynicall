import torch
from config import *
from utils import resize_vocab, load_model
from dataprep  import *
import os, subprocess
import gc
import torch
import wandb
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
# from peft import get_peft_model, prepare_model_for_int8_training

import bitsandbytes
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format, SFTTrainer
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# Defined in the secrets tab in Google Colab

os.environ["WANDB_NOTEBOOK_NAME"] = f"{__location__}/main.py"
wandb.login(key=os.environ.get("WANDB_API_KEY"))

# Load mdoel and tokenizer
if False and os.path.exists(MODEL_NAME.split('/')[1] + "-Extended"):
    os.system(f"rm -rf {MODEL_NAME.split('/')[1]}-Extended")
    # Remove wandb and outputs folders
    os.system(f"rm -rf wandb outputs")


if not os.path.exists(MODEL_NAME.split('/')[1] + "-Extended"):
    tokenizer, model = resize_vocab(MODEL_NAME, SPECIAL_TOKENS, save_pretrained=True)
else:
    tokenizer, model = load_model(f"local/{MODEL_NAME.split('/')[1]}-Extended")


# Load dataset (Processed one)
try:
    # Load again
    dataset = load_from_disk("local/" + DATASET_NAME + "-processed")
except:
    dataset = load(DATASET_NAME, split="all")
    dataset = dataset.map(process_row)
    dataset.save_to_disk(DATASET_NAME + "-processed")

# Check multiple tools
data = dataset
multi_tools_definition = sum([1 for d in data if len(json.loads(d['tool'])) > 1 ])
multi_tools_detected = sum([1 for d in data if len(json.loads(d['response'])) > 1 ])
len(data), multi_tools_definition, multi_tools_detected
# # QLoRA config
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch_dtype,
#     bnb_4bit_use_double_quant=True,
# )

# LoRA config
peft_config = LoraConfig(
    r=256, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    lora_alpha=512,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)


model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Ensure all parameters require gradients
# for param in model.parameters():
#     param.requires_grad = True


# # Verify that the model is ready for training
# for name, param in model.named_parameters():
#     if not param.requires_grad:
#         print(f"Parameter {name} does not require grad.")

dataset = dataset.train_test_split(test_size=0.05)

MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 16
GRADIENT = 2
EPOCH = 3
LEARNING_RATE = 2e-4
WEIGHT_DECAY= 0.01
MAX_STEPS = int(len(dataset["train"]) / BATCH_SIZE * EPOCH)
WARMUP_STEPS = int(MAX_STEPS * 0.1)

trainer = SFTTrainer(
    model = model,
    # peft_config=peft_config,
    tokenizer = tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        num_train_epochs=EPOCH,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRADIENT,
        learning_rate = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
        logging_steps=50,
        eval_strategy="steps",
        save_steps=200,
        max_steps = MAX_STEPS,
        lr_scheduler_type = "linear",
        optim = "adamw_8bit",
        warmup_steps = WARMUP_STEPS,
        tf32 = True,
        bf16 = True,
        seed = 3407,
        output_dir = "outputs", 
        report_to = "wandb",
    ),
)
model.config.use_cache = False

trainer.train() 

# only saves the incremental 🤗 PEFT weights (adapter_model.bin) that were trained, meaning it is super efficient to store, transfer, and load.
trainer.model.save_pretrained(MODEL_SAVE_FOLDER_NAME)
# save the full model and the training arguments
trainer.save_model(MODEL_SAVE_FOLDER_NAME)
trainer.model.config.save_pretrained(MODEL_SAVE_FOLDER_NAME)