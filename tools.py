from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import TrainerCallback, TrainerState, TrainerControl
import torch
import requests


class StopOnMultiToken(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the generated sequence ends with the stop token sequence
        if input_ids.shape[1] < len(self.stop_token_ids):
            return False
        return torch.equal(input_ids[0, -len(self.stop_token_ids):], self.stop_token_ids)


class PlotLossCallback(TrainerCallback):
    def __init__(self, chart_name=None):
        self.chart_name = chart_name
        self.training_losses = []
        self.eval_losses = []
        
        # Create a new chart
        payload = {'name': chart_name} if chart_name else {}
        response = requests.post('http://127.0.0.1:5000/create_chart', json=payload)
        self.chart_id = response.json()['chart_id']
        print(f"View your training chart at: http://127.0.0.1:5000/chart/{self.chart_id}")

    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        logs = state.log_history[-1]
        data = {}
        if 'loss' in logs:
            self.training_losses.append(logs['loss'])
            data['loss'] = logs['loss']
        if 'eval_loss' in logs:
            self.eval_losses.append(logs['eval_loss'])
            data['eval_loss'] = logs['eval_loss']
        
        if data:
            # Send data to the Flask server
            requests.post(f'http://127.0.0.1:5000/update_chart/{self.chart_id}', json=data)