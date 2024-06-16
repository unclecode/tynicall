from config import *
from utils import *
from dataprep import *
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class FunctionCallBlock:
    query: str
    function_call: List[Dict[str, Any]]
    function_response: str
    answer: str
    
    def __post_init__(self):
        # Parse the function_call string to a JSON list
        self.function_call = json.loads(self.function_call.replace("\'", ""))
        self.function_call['arguments'] = json.dumps(self.function_call['arguments'])
        self.function_call = [self.function_call]
    
    def dumps(self) -> List[str]:
        return [
            f"USER: {self.query}",
            f"ASSISTANT: <functioncall> {json.dumps(self.function_call)}",
            f"FUNCTION RESPONSE: {self.function_response}",
            f"ASSISTANT: {self.answer}"
        ]

@dataclass
class FunctionCallDataRow:
    system: str
    tools: Dict[str, Any]
    tools_count: int
    chat: List[str]
    calls: int
    parsed_blocks: List[FunctionCallBlock] = field(default_factory=list)
    
    def parse_chat(self):
        self.parsed_blocks = FunctionCallDataRow.parse_chat_to_blocks(self.chat)

        
    @staticmethod
    def parse_chat_to_blocks(chat: List[str]) -> List[FunctionCallBlock]:
        blocks = []
        i = 0
        while i < len(chat):
            if (chat[i].startswith("USER") and 
                (i + 3 < len(chat)) and 
                chat[i + 1].startswith("ASSISTANT: <functioncall>") and 
                chat[i + 2].startswith("FUNCTION RESPONSE") and 
                chat[i + 3].startswith("ASSISTANT")):
                
                block = FunctionCallBlock(
                    query=chat[i].replace("USER: ", "").strip(),
                    function_call=chat[i + 1].replace("ASSISTANT: <functioncall> ", "").replace("<|endoftext|>", "").strip(),
                    function_response=chat[i + 2].replace("FUNCTION RESPONSE: ", "").strip(),
                    answer=chat[i + 3].replace("ASSISTANT: ", "").strip()
                )

                blocks.append(block)
                i += 4
            else:
                i += 1
        return blocks

ds = load(DATASET_NAME, split="all")

def data_row_formatter(data) -> FunctionCallDataRow:
    system_message = "For the given user query, only select the required tool(s). Return empty ([]) If no tool is required."
    tools = extract_functions(data['system'])
    d = data['chat'].replace("\n\n\n", "\n\n")
    chat = [x for x in d.split('\n\n') if x]
    
    function_call_count = 0
    for msg in data['conversations']:
        if '<functioncall>' in msg['value']:
            function_call_count += 1
    
    formatted_data = FunctionCallDataRow(
        system=system_message,
        tools=tools,
        tools_count=len(tools),
        chat=chat,
        calls=function_call_count
    )
    
    formatted_data.parse_chat()
    return formatted_data

data_row_formatter(ds[5])

# rows = [data_row_formatter(d) for d in ds]

# multi_tools_single_call = [d for d in rows if d['tools_count'] > 1 and d['calls'] == 1]
# multi_tools_single_call = sorted(multi_tools_single_call, key=lambda x: x['tools_count'])

# multi_tools_multi_calls = [d for d in rows if d['tools_count'] > 1 and d['calls'] > 1]
# multi_tools_multi_calls = sorted(multi_tools_multi_calls, key=lambda x: x['calls'])

# single_tool_multi_calls = [d for d in rows if d['tools_count'] == 1 and d['calls'] > 1]
# single_tool_multi_calls = sorted(single_tool_multi_calls, key=lambda x: x['calls'])

# single_tools_single_call = [d for d in rows if d['tools_count'] == 1 and d['calls'] == 1]

# len(multi_tools_multi_calls), len(multi_tools_single_call), len(single_tool_multi_calls), len(single_tools_single_call)


# with open(f"data/multi_tools_multi_calls.json", "w") as f:
#     json.dump(multi_tools_multi_calls, f, indent=2)

# with open(f"data/single_tools_single_call.json", "w") as f:
#     json.dump(single_tools_single_call, f, indent=2)

# with open(f"data/multi_tools_single_call.json", "w") as f:
#     json.dump(multi_tools_single_call, f, indent=2)

# with open(f"data/single_tool_multi_calls.json", "w") as f:
#     json.dump(single_tool_multi_calls, f, indent=2)

# with open(f"data/multi_tools_multi_calls_100.json", "w") as f:
#     json.dump(multi_tools_multi_calls[-100:], f, indent=2)

# with open(f"data/single_tools_single_call_100.json", "w") as f:
#     json.dump(single_tools_single_call[-100:], f, indent=2)

# with open(f"data/multi_tools_single_call_100.json", "w") as f:
#     json.dump(multi_tools_single_call[-100:], f, indent=2)

# with open(f"data/single_tool_multi_calls_100.json", "w") as f:
#     json.dump(single_tool_multi_calls[-100:], f, indent=2)


multi_tools_multi_calls = []
with open(f"data/multi_tools_multi_calls_100.json", "r") as f:
    multi_tools_multi_calls = json.loads(f.read())


import random

def create_function_switcher_dataset(data, N, l, u, shuffle=False):
    new_dataset = []
    used_combinations = set()
    num_samples = len(data)
    
    # Assign IDs to each data item
    for i, item in enumerate(data):
        item['id'] = i + 1

    def get_unique_combination(num, used_combinations):
        combination = tuple(random.sample(range(1, num_samples + 1), num))
        while combination in used_combinations:
            combination = tuple(random.sample(range(1, num_samples + 1), num))
        return combination

    for _ in range(N):
        S = random.randint(l, u)
        selected_ids = get_unique_combination(S, used_combinations)
        used_combinations.add(selected_ids)
        
        all_blocks = []
        total_calls = 0
        
        for sid in selected_ids:
            sample = data[sid - 1]
            total_calls += sample['calls']
            chat = sample['chat']
            
            i = 0
            while i < len(chat):
                if chat[i].startswith("USER") and (i + 3 < len(chat)) and chat[i + 1].startswith("ASSISTANT: <functioncall>") and chat[i + 2].startswith("FUNCTION RESPONSE") and chat[i + 3].startswith("ASSISTANT"):
                    block = chat[i:i + 4]
                    all_blocks.append(block)
                    i += 4
                else:
                    i += 1
        
        if shuffle:
            random.shuffle(all_blocks)
        
        # Flatten the list of blocks back into a chat list
        new_chat = [msg for block in all_blocks for msg in block]
        
        # Add the final thank you and you're welcome messages
        new_chat.extend([
            "USER: Thank you for the information.",
            "ASSISTANT: You're welcome! If you have any other questions, feel free to ask."
        ])
        
        # Gather the first tool from each sample
        new_tools = [data[sid - 1]['tools'][0] for sid in selected_ids]
        
        # Create new data item
        new_data_item = {
            "system": "For the given user query, only select the required tool(s). Return empty ([]) If no tool is required.",
            "tools": new_tools,
            "tools_count": len(new_tools),
            "chat": new_chat,
            "calls": total_calls,
            "selected_indices": list(selected_ids)  # Add indices of selected samples
        }
        
        new_dataset.append(new_data_item)
    
    return new_dataset

N = 1  # Size of the new dataset
l = 2    # Lower bound of the number of switches
u = 3    # Upper bound of the number of switches

random.seed(412)
new_dataset = create_function_switcher_dataset(multi_tools_multi_calls, N, l, u, True)
new_dataset[0]
len(new_dataset)

multi_tools_multi_calls[204]


# Save in data/seq folder with _100 suffix
with open(f"data/seq/data_100.json", "w") as f:
    json.dump(new_dataset, f, indent=2)
    
    
    
    
