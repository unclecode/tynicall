import torch, os, textwrap, requests, json
from torch.utils.data import DataLoader
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from config import *
from utils import *
from dataprep import *
from tools import *
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments, pipeline, DataCollatorForLanguageModeling
from transformers import GPTNeoConfig, GPTNeoForCausalLM
from trl import SFTTrainer
import matplotlib.pyplot as plt

wrapper = textwrap.TextWrapper(width=80)
REPORT_TO_WANDB = False
if REPORT_TO_WANDB:
    import wandb
    os.environ["WANDB_NOTEBOOK_NAME"] = f"{__location__}/main.py"
    # os.environ["WANDB_ENTITY"] = "unclecode/alephnull"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_PROJECT"] = f"tinycall"
    wandb.login(key=os.environ.get("WANDB_API_KEY"))


model_name =  "roneneldan/TinyStories-1M"
model_name =  "roneneldan/TinyStories-33M"
model_name = "microsoft/Phi-3-mini-128k-instruct"
model_name =  "EleutherAI/gpt-neo-125m"


# https://github.com/huggingface/transformers/issues/22794
tokenizer, model = get_model(model_name, resize=False, pad = True, eos = False, add_eos_token = False)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# # Test the collator
# examples = ["hello" + tokenizer.eos_token]
# encodings = [tokenizer(e) for e in examples]
# dataloader = DataLoader(encodings, collate_fn=data_collator, batch_size=1)
# batch = next(iter(dataloader))
# batch.keys()
# labels = batch['labels']
# batch['labels']

#print(tokenizer.eos_token, tokenizer.eos_token_id, tokenizer.pad_token, tokenizer.pad_token_id, tokenizer.padding_side)
# model.config

# Build new structure
# num_layers, num_heads, num_layers, num_heads, hidden_size = 16, 16, 16, 16, 1024 #num_heads * 64
# model = build_model(num_layers=num_layers, num_heads=num_heads, hidden_size=hidden_size, tokenizer=tokenizer, device=device, resize=False)

print(f"Model parameters: {model.num_parameters()}")



# Prepare data
def data_row_formatter(system_message, user, tools, response):
    system_message = "For the given user query, only select the required tool(s). Return empty ([]) If no tool is required."
    text = f"""<system>\n{system_message}\n</system>\n\n<available_tools>\n{tools}\n</available_tools>\n\n<query>\n{user}\n</query>\n\n<selected_tools>\n{response}\n</selected_tools> {tokenizer.eos_token}"""
    # text = f"""## Instruction:\n{system_message}\n\n## Aavailable Tools:\n{tools}\n\n## User Query:\n{user}\n\n## Response:\n{response}\n<|endoftext|>"""
    return text

def data_row_formatter(data):
    system_message = "For the given user query, only select the required tool(s). Return empty ([]) If no tool is required."
    d = data['chat'].replace("\n\n\n", "\n\n")
    text =f"SYSTEM: {system_message}\n\n{data['tools']}\n{d}"
    return text

prefix = "-alpaca"
prefix = "-xml"
prefix = "-raw-PAD"
dataset = load_and_prepare(
    DATASET_NAME, tokenizer=tokenizer, formatter=data_row_formatter, 
    forced_reload=True, prefix=prefix
)
dataset = dataset.train_test_split(test_size=0.05, seed = 42)
dataset['train'] = dataset['train'].shuffle(seed=42)
# Shuffle the train dataset


    




# Save the test data
# dataset['test'].save_to_disk("data/test_data")

print(dataset['train'][4]['text'])

# Train
# prefix = "-v3"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 8
GRADIENT = 2
EPOCH = 3
LEARNING_RATE = 2e-3 
LEARNING_RATE = 2e-5 # 1e-5 # 2e-4
LEARNING_RATE = (2e-4 -  2e-5) / 2
LEARNING_RATE = 1e-5 # 1e-5 # 2e-4
LEARNING_RATE = 2e-4 # 1e-5 # 2e-4
# LEARNING_RATE = 1e-4
WEIGHT_DECAY= 0.01
MAX_STEPS = int(len(dataset["train"]) / BATCH_SIZE * EPOCH)
WARMUP_STEPS = int(MAX_STEPS * 0.1)
WARMUP_STEPS = min(WARMUP_STEPS, 2000)
UNIQU_NAME = f"{model_name.split('/')[1]}_{LEARNING_RATE}lr{prefix}".replace(".", "_").lower()
args = TrainingArguments(
        num_train_epochs=EPOCH,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size = BATCH_SIZE // 2,
        gradient_accumulation_steps = GRADIENT,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},

        learning_rate = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
        optim = "adamw_torch",
        warmup_steps = WARMUP_STEPS,
        lr_scheduler_type = "cosine", # "linear",

        eval_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_steps=100,
        save_strategy="steps", # "epoch", "no"

        max_steps = MAX_STEPS,
        # optim = "adamw_8bit",
        tf32 = False if device.type != "cuda" else True,
        # bf16 = torch.cuda.is_bf16_supported() if device.type == "cuda" else False,
        seed = 3407,

        
        # output_dir =  "outputs_gpt_neo_2_e_5", 
        output_dir =  f"outputs_{UNIQU_NAME}",
        # report_to = "wandb",
        report_to = "none",
        save_safetensors=True,

        load_best_model_at_end=True,
        save_total_limit=10,
)

USE_LORA = False

trainer = SFTTrainer(
    model = model,
    args=args,
    peft_config=peft_config if USE_LORA else None,
    tokenizer = tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    data_collator=data_collator,
    dataset_num_proc = 4,
    packing = False,
    callbacks=[PlotLossCallback(chart_name=f"chart_{UNIQU_NAME}")],
)
model.config.use_cache = False

trainer.train()
trainer.train(resume_from_checkpoint=True)

location = "results/" 
NEW_MODEL_NAME = "TinyCall-" + model_name.split("/")[1].split("-")[1]
M_NAME = NEW_MODEL_NAME + '-2e-5'

# Save the model
tokenizer.save_pretrained(location + M_NAME)
trainer.model.config.save_pretrained(location + M_NAME)
trainer.save_model(location + M_NAME)
trainer.model.save_pretrained(location + M_NAME)

# Save the 16-bit model
trainer.model.half().save_pretrained(f"{location}{M_NAME}-fp16")
tokenizer.save_pretrained(f"{location}{M_NAME}-fp16")
trainer.model.config.save_pretrained(location + M_NAME + "-fp16")


# Train from the last checkpoint
# trainer.train(resume_from_checkpoint=True)
# trainer.train("outputs/checkpoint-1000")

# Load model from a saved checkpoint
# tokenizer, model = get_model("outputs/checkpoint-1000")

# Test after fine-tuning
# print(wrapper.fill(test_basic(model, tokenizer)))
print(wrapper.fill(test_function_call(model, tokenizer)))
tests = []
with open('tests.json', 'r') as f:
    tests = json.load(f)
test = tests[1]
prompt = format_prompt_v2(test['messages'], test['tools'])
print(prompt)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)