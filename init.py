import torch, os, textwrap, wandb
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from config import *
from utils import *
from dataprep import *
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments, pipeline
from transformers import GPTNeoConfig, GPTNeoForCausalLM
from trl import SFTTrainer
import matplotlib.pyplot as plt
from transformers import TrainerCallback, TrainerState, TrainerControl

wrapper = textwrap.TextWrapper(width=80)
os.environ["WANDB_NOTEBOOK_NAME"] = f"{__location__}/main.py"
# os.environ["WANDB_ENTITY"] = "unclecode/alephnull"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_PROJECT"] = f"tinycall"
wandb.login(key=os.environ.get("WANDB_API_KEY"))

def quick_test(model, tokenizer, max_token = 100):
    prompt = "EleutherAI has"
    prompt = "Once on a time, there"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, 
        do_sample=True, #top_p=0.95, temperature=0.1,
        max_new_tokens=max_token, eos_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(wrapper.fill(response))