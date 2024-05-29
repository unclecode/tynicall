from config import *
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_eos_token=True,
        use_fast=True
    )

    if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
        attn_implementation = "flash_attention_2"
    else:
        torch_dtype = torch.float16
        attn_implementation = "eager"    

    config = AutoConfig.from_pretrained(
        model_name, 
        device_map="auto",
        attn_implementation=attn_implementation, 
        trust_remote_code=True,
        resume_download=None
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        attn_implementation=attn_implementation, 
        config=config, 
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
        resume_download=None
    )
    model.to(device)
    
    return tokenizer, model
    
def resize_vocab(model_name, special_tokens, save_pretrained=False,  **kwargs):
    tokenizer, model = load_model(model_name, **kwargs)
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print("We have added", num_added_toks, "tokens")
    
    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    tokenizer.padding_side = "right"
    print("Padding", tokenizer.pad_token, tokenizer.pad_token_id)

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    model.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if save_pretrained:
        tokenizer.save_pretrained(f"local/{model_name}-Extended")
        model.save_pretrained(f"local/{model_name}-Extended")
    return tokenizer, model


def get_model_and_tokenizer(model_name):
    if not os.path.exists("local/" + model_name.split('/')[1] + "-Extended"):
        print("Download the model, addd PAD and specialtokens, and save it locally, then return the tokenizer and model")
        tokenizer, model = resize_vocab(model_name, SPECIAL_TOKENS, save_pretrained=True)
    else:
        print("Load the model from local. Already exists.")
        tokenizer, model = load_model(f"local/{model_name.split('/')[1]}-Extended")    

    return tokenizer, model

def remove_local_cache():
    # Remove the local cache
    os.system(f"rm -rf wandb outputs local")    


def generate_token_histogram(formatter, data, tokenizer, bins = 25):
    token_counts = []
    for row in data:
        token_counts.append(
            len(
                tokenizer(
                    formatter(row),
                    add_special_tokens=True,
                    return_attention_mask=False,
                )["input_ids"]
            )
        )

    n, bins, patches = plt.hist(token_counts, bins= bins, facecolor='blue', alpha=0.5)

    # Create a histogram object
    histogram = {
        'n': n,  # The values of the histogram bins
        'bins': bins,  # The edges of the bins
        'patches': patches,  # Silent list of individual patches used to create the histogram
    }

    return token_counts, histogram




# ⚡ ~ pip install -qqq flash-attn
# ⚡ ~ pip install -qqq -U transformers datasets accelerate peft trl bitsandbytes wandb --progress-bar off
# ⚡ ~ pip install ternsorboard