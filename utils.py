from config import *
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
        attn_implementation = "flash_attention_2"
    else:
        torch_dtype = torch.float16
        attn_implementation = "eager"    

    config = AutoConfig.from_pretrained(model_name, attn_implementation=attn_implementation, resume_download=None)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        attn_implementation=attn_implementation, 
        config=config, 
        torch_dtype=torch.bfloat16,
    )
    model.to("cuda")
    
    return tokenizer, model
    
def resize_vocab(model_name, special_tokens, save_pretrained=False,  **kwargs):
    tokenizer, model = load_model(model_name, **kwargs)
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print("We have added", num_added_toks, "tokens")
    model.resize_token_embeddings(len(tokenizer))
    if save_pretrained:
        tokenizer.save_pretrained("TinyStories-1M-Extended")
        model.save_pretrained("TinyStories-1M-Extended")
    return tokenizer, model

# ⚡ ~ pip install -qqq flash-attn
# ⚡ ~ pip install -qqq -U transformers datasets accelerate peft trl bitsandbytes wandb --progress-bar off
# ⚡ ~ pip install ternsorboard