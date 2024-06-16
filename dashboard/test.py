import time
import random
import requests

# Create a new chart
response = requests.post('http://127.0.0.1:5000/create_chart', json={'name': 'training_loss'})
chart_id = response.json()['chart_id']
print(f"View your training chart at: http://127.0.0.1:5000/chart/{chart_id}")

for i in range(100):
    # Generate random training and evaluation loss
    training_loss = random.uniform(0, 10)
    eval_loss = random.uniform(0, 10)
    
    # Send data to the Flask server
    data = {
        'loss': training_loss,
        'eval_loss': eval_loss
    }
    requests.post(f'http://127.0.0.1:5000/update_chart/{chart_id}', json=data)
    
    # Wait for a second before sending the next data point
    time.sleep(1)
