import pyautogui
import torch
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('Automation/.env')
debug_mode = os.getenv('DEBUG_MODE') == 'TRUE'

def simulate_mouse_movement_from_model(start_point, end_point, model, sequence_length=420):
    def preprocess_points(start_point, end_point, sequence_length):
        # Replace this with your actual preprocessing logic
        return np.random.rand(sequence_length, 9)  # Example: Random sequence

    # Debug mode check
    if debug_mode:
        print("Debug mode is ON. Running simulated mouse movement with debug information.")

    input_sequence = preprocess_points(start_point, end_point, sequence_length)
    input_tensor = torch.tensor([input_sequence], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predicted_sequence = model(input_tensor).numpy()[0]

    for i in range(len(predicted_sequence)):
        x, y, time_delta = predicted_sequence[i]
        
        time_delta = max(0.1, time_delta)  # Example minimum duration

        if debug_mode:
            print(f"Moving to ({x}, {y}) over {time_delta} seconds.")
        else:
            pyautogui.moveTo(x, y, duration=time_delta)

# Example usage
# simulate_mouse_movement_from_model((100, 100), (400, 400), your_model)
