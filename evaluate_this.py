import torch
from config import *
from utils import *
from dataprep   import *
import os, subprocess
import torch
import textwrap
wrapper = textwrap.TextWrapper(width=80)
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib import pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "results/TinyCall-1M-2e-4-10000-fp16"
model_name = "results/TinyCall-3M-2e-3-fp16"
model_name = "results/TinyCall-3M-2e-3-fp16"
model_name = "results/TinyCall-33M-2e-4-fp16"
model_name = "/teamspace/studios/this_studio/outputs_33M_2_e_4/checkpoint-20000"
model_name = "/teamspace/studios/this_studio/outputs_50M_2_e_4/checkpoint-12600"
model_name = "/teamspace/studios/this_studio/outputs_tinystories-1m_0_002lr/checkpoint-15000"
model_name = "/teamspace/studios/this_studio/outputs_tinystories-1m_2e-05lr/checkpoint-17500"
model_name = "/teamspace/studios/this_studio/outputs_tinystories-33m_0_0002lr/checkpoint-20000"

model_name = "/teamspace/studios/this_studio/outputs_gpt_neo_2_e_4/checkpoint-2000"
model_name = "/teamspace/studios/this_studio/outputs_gpt_neo_2_e_5/checkpoint-18900"
model_name = "/teamspace/studios/this_studio/outputs_gpt-neo-125m_0_0002lr-xml/checkpoint-18500"
tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True, use_fast=True)
attn_implementation = "flash_attention_2"
config = AutoConfig.from_pretrained( model_name, device_map="auto", attn_implementation=attn_implementation)
model = AutoModelForCausalLM.from_pretrained( model_name, config=config, torch_dtype = torch.bfloat16, attn_implementation=attn_implementation)
model = model.to(device)

tests = []
with open('tests.json', 'r') as f:
    tests = json.load(f)

# test = tests[-1]
# # def data_row_formatter(system_message, user, tools, response = ""):
# #     system_message = "For the given user query, only select the required tool(s). Return emoty ([]) If not tool is required."
# #     text = f"""<system>\n{system_message}\n</system>\n\n<available_tools>\n{tools}\n</available_tools>\n\n<query>\n{user}\n</query>\n\n<selected_tools>\n{response}\n</selected_tools> <|endoftext|>"""
# #     # text = f"""## Instruction:\n{system_message}\n\n## Aavailable Tools:\n{tools}\n\n## User Query:\n{user}\n\n## Response:\n"""
# #     return text
# prompt = format_prompt_v2(test['messages'], test['tools'])
# print(prompt)

 
prompt = """<system>
For the given user query, only select the required tool(s). Return emoty ([]) If not tool is required.
</system>

<available_tools>
[{"name": "search_hotel", "description": "Search for a hotel based on given criteria", "parameters": {"type": "object", "properties": {"destination": {"type": "string", "description": "The destination of the hotel"}, "check_in_date": {"type": "string", "description": "The check-in date (YYYY-MM-DD)"}, "check_out_date": {"type": "string", "description": "The check-out date (YYYY-MM-DD)"}}, "required": ["destination", "check_in_date", "check_out_date"]}}]
</available_tools>

<query>
Hi, I need to find a hotel in Paris for my upcoming trip. I'll be there from 2022-05-01 to 2022-05-10.
</query>

<selected_tools>
"""


inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, 
    top_p=0.95, temperature=1,
    max_new_tokens=128, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
print(response.split('<selected_tools>')[1].split('</selected_tools>')[0])
print(len(response.split('</selected_tools>')[1]))


