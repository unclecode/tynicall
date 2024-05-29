import torch
from config import *
from utils import *
from dataprep  import *
import os, subprocess
import gc
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import wandb
from torch.utils.data import DataLoader
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
from trl import ORPOConfig, ORPOTrainer, setup_chat_format, SFTTrainer, DataCollatorForCompletionOnlyLM
from matplotlib import pyplot as plt
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# Defined in the secrets tab in Google Colab
os.environ["WANDB_NOTEBOOK_NAME"] = f"{__location__}/main.py"
wandb.login(key=os.environ.get("WANDB_API_KEY"))

False and remove_local_cache()

tokenizer, model = get_model_and_tokenizer(MODEL_NAME)
# tokenizer.pad_token, tokenizer.pad_token_id
# print(model.config)

# Load dataset (Processed one)
dataset = load_and_prepare(DATASET_NAME, tokenizer)
generate_token_histogram(dataset)
dataset = dataset.train_test_split(test_size=0.05)

# Check multiple tools
# # QLoRA config
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch_dtype,
#     bnb_4bit_use_double_quant=True,
# )

#  Get linear layer name 
# def list_linear_layers(model):
#     linear_layers = []
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             linear_layers.append(name)
#     return linear_layers
# linear_layer_names = list_linear_layers(model)

# LoRA config
peft_config = LoraConfig(
    r=512, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    lora_alpha=512,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules= [
        'c_proj', 
        'c_fc', 
        'k_proj', 
        'q_proj', 
        'v_proj', 
        'out_proj'
    ]
)
get_peft_model(model, peft_config).print_trainable_parameters()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

response_template = "<0x0A><0x0A>"
response_template_ids = tokenizer.encode(
    response_template, add_special_tokens=False
)[2:]
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

count_trainable_parameters(model)
# # Test the collator
# examples = [process_row(dataset["train"][0])['text']]
# encodings = [tokenizer(e) for e in examples]
# dataloader = DataLoader(encodings, collate_fn=collator, batch_size=1)
# batch = next(iter(dataloader))
# batch.keys()
# batch['labels']


# # No need i guess
# model, tokenizer = setup_chat_format(model, tokenizer)
# model = prepare_model_for_kbit_training(model)

MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 4
GRADIENT = 4
EPOCH = 3
LEARNING_RATE = 2e-4
WEIGHT_DECAY= 0.01
MAX_STEPS = 50 #int(len(dataset["train"]) / BATCH_SIZE * EPOCH)
WARMUP_STEPS = int(MAX_STEPS * 0.1)

trainer = SFTTrainer(
    model = model,
    # peft_config=peft_config,
    tokenizer = tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    data_collator=collator,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        num_train_epochs=EPOCH,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRADIENT,
        learning_rate = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
        logging_steps=200,
        eval_strategy="steps",
        max_grad_norm=1.0,
        save_steps=200,
        max_steps = MAX_STEPS,
        lr_scheduler_type = "cosine", # "linear",
        # optim = "adamw_8bit",
        optim = "adamw_torch",
        warmup_steps = WARMUP_STEPS,
        tf32 = False if device.type != "cuda" else True,
        # bf16 = torch.cuda.is_bf16_supported() if device.type == "cuda" else False,
        seed = 3407,
        output_dir = "outputs", 
        report_to = "wandb",
        save_safetensors=True,
    ),
)
model.config.use_cache = False

trainer.train() 

# only saves the incremental 🤗 PEFT weights (adapter_model.bin) that were trained, meaning it is super efficient to store, transfer, and load.
location = "local/resut/" 
NEW_MODEL_NAME = NEW_MODEL_NAME + '-2e-4-5000'
trainer.model.save_pretrained(location + NEW_MODEL_NAME)
tokenizer.save_pretrained(location + NEW_MODEL_NAME)
# save the full model and the training arguments
trainer.save_model(location + NEW_MODEL_NAME)
trainer.model.config.save_pretrained(location + NEW_MODEL_NAME)


# Test
# Test the pipeline with a sample input
i = 0
example = dataset["test"][i]
text = example["text"]
text = text.split("<0x0A><0x0A>")[0] + "<0x0A><0x0A>"

example["text"].split("<0x0A><0x0A>")[1]

# Ensure the input is correctly tokenized
inputs = tokenizer(text, return_tensors="pt").to(device)

# Can be ignorded
inputs = {key: val.half() if key != 'input_ids' else val for key, val in inputs.items()}

# Generate outputs
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# Inference and Test
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
)
 
model = PeftModel.from_pretrained(base_model, MODEL_SAVE_FOLDER_NAME)
model = model.merge_and_unload()
 
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_SAVE_FOLDER_NAME, add_eos_token=False
)
 
tokenizer.pad_token, tokenizer.pad_token_id, tokenizer.padding_side

model.pad_token_id = tokenizer.pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id


from transformers import pipeline, logging
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    return_full_text=False,
    torch_dtype=torch.float16,
)

logging.set_verbosity(logging.CRITICAL)

i = 0
for i in range(3):
    example = dataset["test"][i]
    print(f"Example {i + 1}\n")
    text = example["text"]
    text = text.split("<0x0A><0x0A>")[0] + "<0x0A><0x0A>"
    print(text)
    outputs = pipe(text)
    response = outputs[0]["generated_text"]
    '[TOOLS]' in response


