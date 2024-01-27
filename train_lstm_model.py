import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split
import pickle
from datetime import datetime
import os
import pandas as pd
import numpy as np  # Add this line to import NumPy
from visualization import visualize_predictions

# Custom dataset class
class MouseMovementDataset(Dataset):
    def __init__(self, segments):
        self.segments = segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        input_features = torch.tensor(np.array([segment[0, :2], segment[-1, :2]]), dtype=torch.float32)
        targets = torch.tensor(segment[:, :2], dtype=torch.float32)  # Include all points
        return input_features, targets

# LSTM model definition
class MouseMovementLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, sequence_length, dropout_prob=0.5):
        super(MouseMovementLSTM, self).__init__()

        # Attributes
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout_prob)

        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_layer_size, hidden_layer_size * 2)
        self.fc2 = nn.Linear(hidden_layer_size * 2, hidden_layer_size)
        self.output_layer = nn.Linear(hidden_layer_size, output_size)

        # Dropout and Activation
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Ensure x is of the shape [batch_size, input_size]
        batch_size, _, input_size = x.shape
        x = x.view(batch_size, -1)  # Flatten the start and end points

        # Repeat x to form a sequence
        x = x.repeat(1, self.sequence_length).view(batch_size, self.sequence_length, -1)

        # LSTM layer
        lstm_out, _ = self.lstm(x)

        # Select the output for each time step
        lstm_out = lstm_out.contiguous().view(batch_size, self.sequence_length, -1)

        # Fully connected layers with dropout and ReLU activations
        out = self.dropout(self.relu(self.fc1(lstm_out)))
        out = self.dropout(self.relu(self.fc2(out)))

        # Final output layer
        predictions = self.output_layer(out)
        
        return predictions

# Custom loss function
class CustomSequenceLoss(nn.Module):
    def __init__(self, start_weight=1.0, end_weight=1.0, sequence_weight=1.0):
        super(CustomSequenceLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.sequence_weight = sequence_weight

    def forward(self, predictions, targets):
        sequence_loss = self.mse_loss(predictions, targets)
        start_loss = self.mse_loss(predictions[:, 0], targets[:, 0])
        end_loss = self.mse_loss(predictions[:, -1], targets[:, -1])
        combined_loss = (self.sequence_weight * sequence_loss + self.start_weight * start_loss + self.end_weight * end_loss)
        return combined_loss

# Training function
def train_lstm_model(directory_name, load_checkpoint=None, save_viz=False, num_epochs=500, batch_size=64, learning_rate=0.001, step_size=50, gamma=0.5, start_weight=5.0, end_weight=5.0, sequence_weight=1.0, checkpoint_interval=10):
    checkpoint_dir = f"{directory_name}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    viz_dir = f"{directory_name}/visualizations"
    os.makedirs(viz_dir, exist_ok=True)

    # Load data
    with open('standardized_segments.pkl', 'rb') as f:
        standardized_segments = pickle.load(f)
    dataset = MouseMovementDataset([segment[['x', 'y']].values for segment in standardized_segments])

    # Split data and create DataLoaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model
    model = MouseMovementLSTM(input_size=4, hidden_layer_size=128, num_layers=2, output_size=2, sequence_length=420, dropout_prob=0.5)
    if load_checkpoint:
        model.load_state_dict(torch.load(load_checkpoint))
        model.eval()

    # Define loss function and optimizer
    criterion = CustomSequenceLoss(start_weight=start_weight, end_weight=end_weight, sequence_weight=sequence_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation loop
        model.eval()
        val_loss = 0.0
        num_batches = 0
        for val_inputs, val_targets in val_loader:
            val_outputs = model(val_inputs)
            batch_loss = criterion(val_outputs, val_targets)
            val_loss += batch_loss.item()
            num_batches += 1
        val_loss /= num_batches

        # Save checkpoints and visualize predictions
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            if save_viz:
                visualize_predictions(model, val_dataset, num_samples=4, epoch=epoch, save_viz=True, viz_dir=viz_dir)

    print('Finished Training')


if __name__ == "__main__":
    train_lstm_model()
