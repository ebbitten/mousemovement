import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pickle
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import os
import pandas as pd

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train and visualize LSTM model for mouse movement prediction.')
parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to a checkpoint to load for continuing training.')
parser.add_argument('--test_checkpoint', type=str, default=None, help='Path to a checkpoint to load for testing and visualization.')
parser.add_argument('--save_viz', action='store_true', help='Save visualizations to a folder instead of displaying them.')
args = parser.parse_args()

# Load the standardized segments from the pickle file
with open('standardized_segments.pkl', 'rb') as f:
    standardized_segments = pickle.load(f)

# Define a custom dataset class
class MouseMovementDataset(Dataset):
    def __init__(self, segments):
        self.segments = segments

    def __len__(self):
        return len(self.segments)


    def __getitem__(self, idx):
        segment = self.segments[idx]
        # Extract the start and end points as input features
        input_features = torch.tensor(np.array([segment[0, :2], segment[-1, :2]]), dtype=torch.float32)
        # The target sequence is all the points in the segment except the first one
        targets = torch.tensor(segment[1:, :2], dtype=torch.float32)  # All points except the first
        return input_features, targets

# Convert list of Pandas DataFrames to list of tensors
tensor_segments = [segment[['x', 'y']].values for segment in standardized_segments]


# Create the dataset
dataset = MouseMovementDataset(tensor_segments)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#define the model
class MouseMovementLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, sequence_length):
        super(MouseMovementLSTM, self).__init__()
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        # Flatten the start and end points
        x = x.view(x.size(0), -1)  # Reshape to [batch_size, 4]

        # Repeat the flattened vector to form a sequence
        x = x.repeat(1, self.sequence_length).view(x.size(0), self.sequence_length, -1)

        lstm_out, _ = self.lstm(x)
        
        # Modify the output to match the target sequence length
        output = self.output_layer(lstm_out[:, :-1, :])  # Exclude the last time step

        return output

# Instantiate the model
model = MouseMovementLSTM(input_size=4, hidden_layer_size=128, num_layers=2, output_size=2, sequence_length=420)


# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load model from checkpoint if specified
if args.load_checkpoint:
    model.load_state_dict(torch.load(args.load_checkpoint))
    model.eval()
    print(f"Loaded checkpoint: {args.load_checkpoint}")


import matplotlib.pyplot as plt
import torch
import numpy as np
from datetime import datetime
import os

def visualize_predictions(model, dataset, num_samples=4, epoch=0, save_viz=False):
    # Determine the directory to save the visualizations
    today_date = datetime.now().strftime('%Y_%m_%d')
    viz_dir = f'gpt_mouse_move/viz_{today_date}'

    # Create the directory if it doesn't exist and save_viz is True
    if save_viz and not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    for i in range(num_samples):
        sample_input, sample_target = dataset[i]
        model.eval()
        with torch.no_grad():
            sample_output = model(sample_input.unsqueeze(0))

        plt.figure(figsize=(10, 5))
        plt.plot(sample_target[:, 0].numpy(), sample_target[:, 1].numpy(), 'ro-', label='Actual Path')
        plt.plot(sample_output[0, :, 0].numpy(), sample_output[0, :, 1].numpy(), 'bs-', label='Predicted Path')
        plt.legend()
        plt.title(f'Epoch {epoch}, Sample {i+1}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.gca().invert_yaxis()  # Invert y-axis to match screen coordinates

        # Save or display the plot
        if save_viz:
            file_name = f'sample_{i+1}_epoch_{epoch}_{today_date}.png'
            plt.savefig(os.path.join(viz_dir, file_name))
            plt.close()  # Close the figure after saving to avoid display
        else:
            plt.show()  # Display the plot



# Training loop for multiple epochs with checkpointing
num_epochs = 50
checkpoint_interval = 5

# Check if starting from scratch or a checkpoint
start_epoch = 0
if args.load_checkpoint:
    start_epoch = int(args.load_checkpoint.split('_')[-1].split('.')[0]) + 1

for epoch in range(start_epoch, num_epochs):
    model.train()
    # Training loop
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_targets).item()
    val_loss /= len(val_loader)

    print(f'Epoch {epoch+1}, Training Loss: {loss.item()}, Validation Loss: {val_loss}')

    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = f'model_epoch_{epoch}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Visualize predictions using the current model state
        visualize_predictions(model, val_dataset, num_samples=4, epoch=epoch, save_viz=args.save_viz)


if args.test_checkpoint:
    model.load_state_dict(torch.load(args.test_checkpoint))
    model.eval()
    visualize_predictions(model, val_dataset, num_samples=4)

print('Finished Training')
