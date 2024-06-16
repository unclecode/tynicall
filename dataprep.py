from datasets import load_dataset, load_from_disk
from config import *
import json, re
from matplotlib import pyplot as plt
from functools import partial
import os

EOS = '<|endoftext|>'

def load_dataset_form_hf(dataset_name, tokenizer,  max_length = 2048, formatter = None):
    dataset = load(dataset_name, split="all")
    row_formatter = partial(process_row_batch, tokenizer=tokenizer, formatter=formatter)
    dataset = dataset.map(row_formatter, batched=True,  batch_size=10000, num_proc=4)
    dataset = dataset.filter(lambda x: x['token_count'] <= max_length)
    return dataset

def load_and_prepare(dataset_name, tokenizer, max_length = 2048, formatter = None, forced_reload = False, prefix = "-processed"):
    if not forced_reload and os.path.exists("local/" + dataset_name +  prefix):
        print("Loading from local cache")
        dataset = load_from_disk("local/" + dataset_name +  prefix)
    else:
        print("Loading from Huggingface, applying processing and saving to local cache")
        dataset = load_dataset_form_hf(dataset_name, tokenizer, max_length, formatter)
        dataset.save_to_disk("local/" + dataset_name +  prefix)
    return dataset

def generate_token_histogram(dataset):
    token_counts = dataset['token_count']  # Adjust for the appropriate split
    plt.figure(figsize=(10, 6))
    plt.hist(token_counts, bins=50, alpha=0.75)
    plt.title('Token Count Distribution')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def load(dataset_name, split="train", shuffle = False, select = None, seed = 42):
    dataset = load_dataset(dataset_name, split=split)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if select:
        dataset = dataset.select(range(select))
    return dataset

 
# def process_row(row):
#     system_functions = extract_functions(row['system'])
#     chat_function_calls = extract_function_calls(row['chat'])
#     row['tool'] = json.dumps(system_functions)
#     row['response'] = json.dumps(chat_function_calls)
#     return row

def extract_json_list(function_definitions):
    # Split by "}{" and handle missing braces
    json_objects = re.split(r'\}\s*\n*\s*\{', function_definitions)
    
    # Fix the JSON objects by adding the necessary braces
    functions = []
    try:
        if len(json_objects) == 1:
            obj = json_objects[0]
            if obj == '':
                obj = '{}'
            obj = re.sub(r'\\', r'\\\\', obj)
            obj = re.sub(r'\'', r'\\\'', obj)
            functions.append(json.loads(obj))
            return functions

        for i, obj in enumerate(json_objects):
            if i == 0:
                obj += '}'
            elif i == len(json_objects) - 1:
                obj = '{' + obj
            else:
                obj = '{' + obj + '}'
            # Escape special characters
            obj = re.sub(r'\\', r'\\\\', obj)
            obj = re.sub(r'\'', r'\\\'', obj)
            functions.append(json.loads(obj))
            # functions.append(obj)
    except:
        print(function_definitions)
    
    return functions

def extract_functions(system_message):
    # Split the system message by '-\n'
    parts = system_message.split('-\n')
    
    # If there's a second part, proceed to extract JSON objects
    if len(parts) > 1:
        function_definitions = parts[1].strip()
        
        return extract_json_list(function_definitions)
    
    return []

def extract_function_calls(chat_message):
    functioncall_start_index = chat_message.find("<functioncall>")
    functioncall_end_index = chat_message.find("<|endoftext|>", functioncall_start_index)
    function_calls = chat_message[functioncall_start_index + len("<functioncall>") : functioncall_end_index].strip()
    if function_calls and functioncall_start_index > -1 and functioncall_end_index > -1:
        try:
            function_calls = function_calls.replace("\'", '')
            function_calls = re.sub(r'\\', r'\\\\', function_calls)
            function_calls = re.sub(r'\'', r'\\\'', function_calls)
            json.loads(function_calls)
        except:
            print(function_calls)
            # print(function_calls)
            pass
        return extract_json_list(function_calls.strip())
    else:
        return []
    
def process_dataset(dataset):
    result = []
    ix = 0
    for entry in dataset:
        print("processing entry", ix)
        system_functions = extract_functions(entry['system'])
        chat_function_calls = extract_function_calls(entry['chat'])
        result.append({'tool': system_functions, 'response': chat_function_calls})
        ix += 1
    return result

def process_row_batch(batch, tokenizer, formatter = None):
    def token_count(row_text):
        return len(
                    tokenizer(
                        row_text,
                        add_special_tokens=True,
                        return_attention_mask=False,
                    )["input_ids"]
                )


    batch_result = {
        "text":[],
        "tool":[],
        "response":[],
        "user":[],
        "conversation":[],
        "token_count":[],
        "chat":[],
    }
    for ix in range(len(batch['system'])):
        system_functions = extract_functions(batch['system'][ix])
        chat_function_calls = extract_function_calls(batch['chat'][ix])
        first_user_message = [r for r in batch['conversations'][ix] if r['from'] == 'human']
        if not first_user_message:
            print("ERROR: NO USER MESSAGE", batch['conversations'][ix])
            continue
        user = first_user_message[0]['value']
        tools = json.dumps(system_functions)
        response = json.dumps(chat_function_calls)
        batch_result['user'].append(user)
        batch_result['conversation'].append(batch['conversations'][ix])
        batch_result['chat'].append(batch['chat'][ix])
        batch_result['tool'].append(tools)
        batch_result['response'].append(response)
        batch_result['token_count'].append(token_count(user))
        if formatter:
            data = {
                "system": batch['system'][ix],
                "chat": batch['chat'][ix],
                "user": user,
                "tools": tools,
                "response": response,
                "conversations": batch['conversations'][ix]
            }
            text = formatter(data)
            # text = formatter(SYSTEM_MESSAGE, user, tools, response, batch['conversations'][ix])
        else:
            text = f"""<s>[AVAILABLE_TOOLS] {tools} [/AVAILABLE_TOOLS][INST] {SYSTEM_MESSAGE}<0x0A><0x0A>{user}<0x0A><0x0A><0x0A>[TOOLS]{response}[/TOOLS][/INST]<|endoftext|>"""
            text = text.replace(' ', '▁')
        batch_result['text'].append(text)

    return batch_result

def process_row(row):
    global EOS
    system_functions = extract_functions(row['system'])
    chat_function_calls = extract_function_calls(row['chat'])
    row['tool'] = json.dumps(system_functions)
    row['response'] = json.dumps(chat_function_calls)

    first_user_message = [r for r in row['conversations'] if r['from'] == 'human']
    if not first_user_message:
        print("ERROR: NO USER MESSAGE", row['conversations'])

    row['user'] = first_user_message
    text = f"""<s>[AVAILABLE_TOOLS] {row['tool']} [/AVAILABLE_TOOLS][INST] {SYSTEM_MESSAGE}<0x0A><0x0A>{row['user']}[TOOLS]{row['response']}[/TOOLS][/INST]{EOS}"""
    row['text'] = text.replace(' ', '▁')
    return row

def format_prompt(messages, tools):
    system_message = [m['content'] for m in messages if m['role'] == 'system'] or [SYSTEM_MESSAGE]
    user = [m['content'] for m in messages if m['role'] == 'user']
    system_message = system_message[0]
    user = user[0]

    tools = json.dumps(tools)
    text = f"""<s>[AVAILABLE_TOOLS] {tools} [/AVAILABLE_TOOLS][INST] {system_message}<0x0A><0x0A>{user}<0x0A><0x0A><0x0A>"""
    text = text.replace(' ', '▁')
    return text
    pass

def format_prompt_v2(messages, tools):
    system_message = [m['content'] for m in messages if m['role'] == 'system'] or [SYSTEM_MESSAGE]
    user = [m['content'] for m in messages if m['role'] == 'user']
    system_message = system_message[0]
    user = user[0]
    tools = json.dumps(tools)
    text = f"""<system>\n{system_message}\n</system>\n\n<available_tools>\n{tools}\n</available_tools>\n\n<query>\n{user}\n</query>\n\n<selected_tools>\n"""
    return text
    pass

if __name__ == "__main__":
    dataset = load(DATASET_NAME, split="all")
    # dataset = dataset.train_test_split(test_size=0.05)
    ix = 0
    t = process_row(dataset[ix])
    import textwrap
    wrapper = textwrap.TextWrapper(width=80)
    print(wrapper.fill(t['text']))    
    # for row in dataset:
    #     if 'I need to ship a package to Canada.' in row['chat']:
    #         break
    #         ix += 1
    # print(ix)