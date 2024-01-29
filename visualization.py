import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import torch

def visualize_predictions(model, dataset, num_samples=4, epoch=0, save_viz=False, viz_dir=None):
    # If a directory name is provided, use it; otherwise, create a new directory based on the current date
    if viz_dir:
        viz_dir = f"{viz_dir}"
    else:
        today_date = datetime.now().strftime('%Y_%m_%d')
        viz_dir = f'viz_{today_date}'

    # Create the visualizations directory if it doesn't exist
    if save_viz and not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    for i in range(num_samples):
        sample_input, sample_target = dataset[i]
        model.eval()
        with torch.no_grad():
            sample_output = model(sample_input.unsqueeze(0))

        # Debugging information
        start_point = sample_input[0].numpy()
        end_point = sample_input[1].numpy()
        first_pred_point = sample_output[0, 0].numpy()
        print(f"Sample {i+1}: Start point: {start_point}, End point: {end_point}, First predicted point: {first_pred_point}")

        plt.figure(figsize=(10, 5))
        plt.plot(sample_target[:, 0].numpy(), sample_target[:, 1].numpy(), 'ro-', label='Actual Path')
        plt.plot(sample_output[0, :, 0].numpy(), sample_output[0, :, 1].numpy(), 'bs-', label='Predicted Path')
        plt.legend()
        plt.title(f'Epoch {epoch}, Sample {i+1}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.gca().invert_yaxis()

        if save_viz:
            file_name = f'epoch_{epoch}_sample_{i+1}.png'
            plt.savefig(os.path.join(viz_dir, file_name))
            plt.close()
        else:
            plt.show()
