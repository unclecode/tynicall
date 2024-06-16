from config import *
import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from dataprep import format_prompt, format_prompt_v2
from transformers import GPTNeoConfig, GPTNeoForCausalLM


from transformers import AutoTokenizer, GPT2TokenizerFast, GPT2Tokenizer
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

# tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True, use_fast=True)

# tokenizer._tokenizer.post_processor = TemplateProcessing(
#     single=tokenizer.bos_token + " $A " + tokenizer.eos_token,
#     special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id), (tokenizer.bos_token, tokenizer.bos_token_id)],
# )
# tokenizer.tokenize("Hello world", add_special_tokens = True)
# print(tokenizer.decode(tokenizer.encode("Hello world")))


# gtokenizer = GPT2TokenizerFast.from_pretrained(model_name, add_eos_token=True, use_fast=True)
# atokenizer.tokenize("Hello <|endoftext|> world", add_special_tokens = True)
# gtokenizer.tokenize("Hello <|endoftext|> world", add_special_tokens = True)
# print(gtokenizer.decode(gtokenizer.encode("GPT2TokenizerFast <|endoftext|>", add_special_tokens = True)))
# print(atokenizer.decode(atokenizer.encode("GPT2TokenizerFast <|endoftext|>", add_special_tokens = True)))



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
    if not os.path.exists("local/" + model_name + "-Extended"):
        print("Download the model, addd PAD and specialtokens, and save it locally, then return the tokenizer and model")
        tokenizer, model = resize_vocab(model_name, SPECIAL_TOKENS, save_pretrained=True)
    else:
        print("Load the model from local. Already exists.")
        tokenizer, model = load_model(f"local/{model_name}-Extended")    

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


def predict(messages, tools = []):
    prompt = format_prompt(messages, tools)
    print(wrapper.fill(prompt))
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = response.split('<0x0A><0x0A><0x0A>')[1].strip()
    print(prediction)
    return prediction

# ⚡ ~ pip install -qqq flash-attn
# ⚡ ~ pip install -qqq -U transformers datasets accelerate peft trl bitsandbytes wandb --progress-bar off
# ⚡ ~ pip install ternsorboard


# Inference and Test
def test_basic(model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt = "Once on a time, there"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, 
        do_sample=True, top_p=0.95, temperature=0.1,
        max_new_tokens=1024, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test function call
def test_function_call(model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tests = []
    with open('tests.json', 'r') as f:
        tests = json.load(f)
    test = tests[1]
    prompt = format_prompt_v2(test['messages'], test['tools'])
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def get_model(model_name, pad = True, resize = True, eos = True, add_eos_token = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            add_eos_token=add_eos_token,
            use_fast=True
    )
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
        device_map="cuda:0",  
        resume_download=None
    )
    model = model.to(device)

    if pad:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        with torch.no_grad():
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.pad_token_id
        # tokenizer.padding_side = "right"
    else:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    if resize:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        model.pad_token_id = tokenizer.pad_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    if eos:
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            # single=tokenizer.bos_token + " $A " + tokenizer.eos_token,
            # special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id), (tokenizer.bos_token, tokenizer.bos_token_id)],
            single="$A " + tokenizer.eos_token,
            special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id)],
        )

    return tokenizer, model


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def build_model( num_layers = 16, num_heads = 32, hidden_size = 1024, tokenizer = None, device = None, pad = True, resize = True):
    # hidden_size = 1024 # num_heads * 64 # Ensure this is divisible by num_heads

    # Create a new configuration
    new_config = GPTNeoConfig(
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_size,
        max_position_embeddings=2048,
        vocab_size=50264,
        activation_function='gelu_new',
        attention_dropout=0,
        resid_dropout=0,
        embed_dropout=0,
        use_cache=False,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        gradient_checkpointing=False,
        attention_types=[
            [
                [
                    "global",
                    "local"
                ],
                num_layers // 2  # Adjust the pattern to match the number of layers
            ]
        ],
        attention_layers=['global', 'local'] * (num_layers // 2),  # Ensure this matches num_layers
    )

    model = GPTNeoForCausalLM(config=new_config)
    model = model.to(device)
    if pad:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        tokenizer.padding_side = "right"
    if resize:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        model.pad_token_id = tokenizer.pad_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    
    return model

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
