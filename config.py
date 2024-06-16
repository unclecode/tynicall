MODEL_NAME = "roneneldan/TinyStories-8M"
MODEL_NAME = "roneneldan/TinyStories-3M"
MODEL_NAME = "roneneldan/TinyStories-33M"
MODEL_NAME = "EleutherAI/gpt-neo-125m"
MODEL_NAME = "roneneldan/TinyStories-1M"
NEW_MODEL_NAME = "TinyCall-1M"
DATASET_NAME = "lilacai/glaive-function-calling-v2-sharegpt"
SPECIAL_TOKENS = ['[AVAILABLE_TOOLS]', '[/AVAILABLE_TOOLS]', '[INST]', '[/INST]', '<s>', '[TOOLS]', '[/TOOLS]']
SYSTEM_MESSAGE = "You are a helpful assistant. Your job is to select tools relevant to the user query. In the case of multiple tools, if the tools are dependent on each other, and one tool's input parameters come from another function, use @ followed by the function name for the parameter value. Remember you response type is List[Dict<String, Any>]."
PAD_TOKEN = "<pad>"

