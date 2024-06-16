
from torch.utils.data import DataLoader, Dataset
import evaluate
import json
import torch
from datasets import load_dataset
import textwrap
wrapper = textwrap.TextWrapper(width=80)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils import *
from tools import *

test_cases = []
try:
    with open('data/test_dataset.json', 'r') as f:
        test_cases = json.load(f)
except FileNotFoundError:
    dataset = load_dataset("fireworks-ai/function-calling-eval-dataset-v0")
    test_cases = []
    for row in dataset['multi_turn']:
        query = row['prompt'][0]['content'].split('\n')[-1]
        tools = [t['function'] for t in json.loads(row['tools'])]
        response = []
        if '<functioncall>' in row['completion']:
            response = json.loads(row['completion'].split('<functioncall>')[1])

        test_cases.append({
            'messages': [
                {'role': 'system', 'content': 'For the given user query, only select the required tool(s). Return emoty ([]) If not tool is required.'},
                {'role': 'user','content': query}
            ],
            'tools': tools,
            'response': response
        })
    # Save test_cases into test_dataset.json
    with open('data/test_dataset.json', 'w') as f:
        json.dump(test_cases, f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "/teamspace/studios/this_studio/outputs_gpt-neo-125m_0_0002lr-xml/checkpoint-18500"
model_name = "/teamspace/studios/this_studio/outputs_gpt-neo-125m_0_0002lr-v2/checkpoint-33600"
model_name = "/teamspace/studios/this_studio/outputs_gpt-neo-125m_0_0002lr-raw-pad/checkpoint-9200"
tokenizer, model = get_model(model_name, resize=False, pad = True, eos = False, add_eos_token = False)
model.eval()
# tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True, use_fast=True)
# tokenizer.padding_side = "right"
# attn_implementation = "flash_attention_2"
# config = AutoConfig.from_pretrained( model_name, device_map="auto", attn_implementation=attn_implementation)
# model = AutoModelForCausalLM.from_pretrained( model_name, config=config, torch_dtype = torch.bfloat16, attn_implementation=attn_implementation)
# model = model.to(device)

stop_token = '</selected_tools>'
stop_token_ids = tokenizer(stop_token, return_tensors="pt", padding=True, truncation=True)['input_ids'].squeeze().to(device)
stopping_criteria = StoppingCriteriaList([StopOnMultiToken(stop_token_ids)])

ix, case = 1, test_cases[1]
results = []
for ix, case in enumerate(test_cases):
    input = format_prompt_v2(case['messages'], case['tools'])
    inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True, add_special_tokens = False).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, 
                                 top_p=0.95, temperature=1,
                                 max_new_tokens=256, 
                                 stopping_criteria=stopping_criteria,
                                 pad_token_id=tokenizer.pad_token_id, 
                                 eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results.append({
        'ix': ix,
        'input': input,
        'query': case['messages'][-1]['content'],
        'response': response,
        'ground_truth': json.dumps([case['response']]),
        'prediction': response.replace(input, '').split('</selected_tools>')[0].strip(),
        'passed': response.replace(input, '').split('</selected_tools>')[0].strip() == json.dumps([case['response']])
    })
    print(f"Test case {ix} passed: {results[-1]['passed']}")

print(response)


# Save the results in data folder
with open('data/test_100_result.json', 'w') as f:
    json.dump(results, f)

accuracy_score = sum([r['passed'] for r in results]) / len(results)
accuracy_score

failed_cases = [r for r in results if not r['passed']]

ix = 0
print(failed_cases[ix]['ix'])
print(failed_cases[ix]['prediction'])
print(failed_cases[ix]['ground_truth'])
ix += 1
print(failed_cases[ix]['response'])
ix -= 1

failed_selected_index = {
    1: "Counting issue (dice)",
    13: "required issue",
    22: "Doesn't stop, when a repetitive pattern happens", 
    69: "Doesn't stop, when a repetitive pattern happens",
    71: "Counting",
    80: "Doesn't stop, when a repetitive pattern happens",
    89: "required issue",
    90: "Doesn't stop, when a repetitive pattern happens",
    91: "required issue",
}


from transformers import GenerationConfig
generation_config = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id, #tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    max_new_tokens=256,
    num_beams=1,
    do_sample=True,
    temperature=0.9,
    top_p=0.95,
)


prompt = """SYSTEM: For the given user query, only select the required tool(s). Return empty ([]) If no tool is required.

[{"name": "calculate_average", "description": "Calculate the average of a list of numbers", "parameters": {"type": "object", "properties": {"numbers": {"type": "array", "items": {"type": "number"}, "description": "The list of numbers to calculate the average of"}}, "required": ["numbers"]}}, {"name": "search_recipes", "description": "Search for recipes based on ingredients", "parameters": {"type": "object", "properties": {"ingredients": {"type": "array", "items": {"type": "string"}, "description": "The ingredients to search for in recipes"}, "dietary_restrictions": {"type": "array", "items": {"type": "string"}, "description": "Any dietary restrictions to consider"}}, "required": ["ingredients"]}}]
USER: Hi, I have a list of numbers and I need to find the average. The numbers are 5, 10, 15, 20, 25. And then search for a recipe with chicken and rice.

ASSISTANT:"""

prompt = """SYSTEM: For the given user query, only select the required tool(s). Return empty ([]) If no tool is required.

[{"name": "calculate_average", "description": "Calculate the average of a list of numbers", "parameters": {"type": "object", "properties": {"numbers": {"type": "array", "items": {"type": "number"}, "description": "The list of numbers to calculate the average of"}}, "required": ["numbers"]}}, {"name": "search_recipes", "description": "Search for recipes based on ingredients", "parameters": {"type": "object", "properties": {"ingredients": {"type": "array", "items": {"type": "string"}, "description": "The ingredients to search for in recipes"}, "dietary_restrictions": {"type": "array", "items": {"type": "string"}, "description": "Any dietary restrictions to consider"}}, "required": ["ingredients"]}}]
USER: Hi, I have a list of numbers and I need to find the average. The numbers are 5, 10, 15, 20, 25. And then search for a recipe with chicken and rice.

ASSISTANT: <functioncall> {"name": "calculate_average", "arguments": '{"numbers": [5, 10, 15, 20, 25]}'} <|endoftext|>

FUNCTION RESPONSE: {"average": 15.0}

ASSISTANT: The average of 15.0 for the numbers is 15. <|endoftext|>

USER: Okay, search for a recipe with chicken and rice.

ASSISTANT:"""
# ASSISTANT: <functioncall> {"name": "search_recipes"""

prompt = """SYSTEM: For the given user query, only select the required tool(s). Return empty ([]) If no tool is required.

[{"name": "search_movies", "description": "Search for movies based on title or genre", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The title or genre of the movie"}}, "required": ["search_query"]}}]
USER: Can you find me some horror movies? 

ASSISTANT: <functioncall> {"name": "search_movies", "arguments": '{"search_query": "horror"}'} <|endoftext|>

FUNCTION RESPONSE: {"movies": ["The Conjuring", "Insidious", "A Quiet Place", "Hereditary", "Get Out"]} 

ASSISTANT: Here are some horror movies you might like: "The Conjuring", "Insidious", "A Quiet Place", "Hereditary", "Get Out". <|endoftext|> 

USER: What about some romantic comedies? 

ASSISTANT:"""

prompt = """SYSTEM: For the given user query, only select the required tool(s). Return empty ([]) If no tool is required.

[{"name": "search_movies", "description": "Search for movies based on title or genre", "parameters": {"type": "object", "properties": {"search_query": {"type": "string", "description": "The title or genre of the movie" } }, "required": ["search_query"]}} ]
USER: Can you find me some horror movies? 

ASSISTANT:"""

prompt = """SYSTEM: For the given user query, only select the required tool(s). Return empty ([]) If no tool is required.

[
    {"name": "calculate_average", "description": "Calculate the average of a list of numbers", "parameters": {"type": "object", "properties": {"numbers": {"type": "array", "items": {"type": "number"}, "description": "The list of numbers to calculate the average of"}}, "required": ["numbers"]}}, 
    {"name": "search_recipes", "description": "Search for recipes based on ingredients", "parameters": {"type": "object", "properties": {"ingredients": {"type": "array", "items": {"type": "string"}, "description": "The ingredients to search for in recipes"}, "dietary_restrictions": {"type": "array", "items": {"type": "string"}, "description": "Any dietary restrictions to consider"}}, "required": ["ingredients"]}}
]
USER: Okay, search for a recipe with chicken and rice.

ASSISTANT:"""


inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, add_special_tokens = True).to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, generation_config=generation_config)
    # outputs = model.generate(**inputs,
    #                             top_p=0.95, temperature=1,
    #                             max_new_tokens=256, 
    #                             # stopping_criteria=stopping_criteria,
    #                             pad_token_id=tokenizer.pad_token_id, 
    #                             eos_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)


from transformers import AutoTokenizer, GPT2TokenizerFast, GPT2Tokenizer
from tokenizers import Tokenizer


