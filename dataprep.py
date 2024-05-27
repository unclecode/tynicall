from datasets import load_dataset, load_from_disk
from config import *
import json, re

def load(dataset_name, split="train", shuffle = False, select = None, seed = 42):
    dataset = load_dataset(dataset_name, split=split)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if select:
        dataset = dataset.select(range(select))
    return dataset

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

def process_row(row):
    system_functions = extract_functions(row['system'])
    chat_function_calls = extract_function_calls(row['chat'])
    row['tool'] = json.dumps(system_functions)
    row['response'] = json.dumps(chat_function_calls)
    return row

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

def process_row(row):
    system_functions = extract_functions(row['system'])
    chat_function_calls = extract_function_calls(row['chat'])
    row['tool'] = json.dumps(system_functions)
    row['response'] = json.dumps(chat_function_calls)

    first_user_message = [r for r in row['conversations'] if r['from'] == 'human']
    if not first_user_message:
        print(row['conversations'])
        raise
        first_user_message = ''
    row['user'] = first_user_message
    text = f"""<s>[AVAILABLE_TOOLS] {row['tool']} [/AVAILABLE_TOOLS][INST] {SYSTEM_MESSAGE}<0x0A><0x0A>[TOOLS]{row['user']}[/TOOLS][/INST]"""
    row['text'] = text.replace(' ', '▁')
    return row



