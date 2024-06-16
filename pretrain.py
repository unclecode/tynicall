# Function to calculate parameters
def calculate_parameters(vocab_size, hidden_size, max_position_embeddings, num_hidden_layers, intermediate_size, num_attention_heads, num_key_value_heads):
    # Embedding parameters
    token_embeddings = vocab_size * hidden_size
    position_embeddings = max_position_embeddings * hidden_size
    embedding_parameters = token_embeddings + position_embeddings

    # Attention layer parameters
    attention_parameters_per_layer = (3 * hidden_size * hidden_size // num_attention_heads) + (hidden_size * hidden_size) + (4 * hidden_size)

    # Fully connected layer parameters
    fully_connected_parameters_per_layer = (hidden_size * intermediate_size) + (intermediate_size * hidden_size) + (2 * intermediate_size)

    # Layer normalization parameters
    layer_norm_parameters_per_layer = 2 * hidden_size

    # Additional final layer norm
    final_layer_norm_parameters = 2 * hidden_size

    # Total parameters
    total_parameters = embedding_parameters + num_hidden_layers * (
        attention_parameters_per_layer + fully_connected_parameters_per_layer + layer_norm_parameters_per_layer
    ) + final_layer_norm_parameters

    return total_parameters

# Initial configuration
vocab_size = 32064
max_position_embeddings = 2048
num_hidden_layers = 12
hidden_size = 512
intermediate_size = 2048
num_attention_heads = 8
num_key_value_heads = 8

vocab_size = 32064
hidden_size = 768
max_position_embeddings = 4096
num_hidden_layers = 16
intermediate_size = 3072
num_attention_heads = 12
num_key_value_heads = 12

# Calculate new parameters
new_parameters = calculate_parameters(vocab_size, hidden_size, max_position_embeddings, num_hidden_layers, intermediate_size, num_attention_heads, num_key_value_heads)
new_parameters
