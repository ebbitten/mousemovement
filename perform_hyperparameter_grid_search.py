import train_lstm_model
import itertools
import os
from datetime import datetime

def perform_hyperparameter_grid_search():
    # Define ranges for hyperparameters
    batch_sizes = [32, 64, 128]
    learning_rates = [0.001, 0.005, 0.01]
    step_sizes = [30, 50, 70]
    gammas = [0.5, 0.7, 0.9]
    start_weights = [1.0, 5.0, 10.0]
    end_weights = [1.0, 5.0, 10.0]
    sequence_weights = [1.0, 5.0, 10.0]

    # Generate all combinations of hyperparameters
    for batch_size, learning_rate, step_size, gamma, start_weight, end_weight, sequence_weight in itertools.product(batch_sizes, learning_rates, step_sizes, gammas, start_weights, end_weights, sequence_weights):
        # Create a unique directory name for each hyperparameter combination
        today_date = datetime.now().strftime('%Y_%m_%d')
        directory_name = f"hyperparam_search_{today_date}_bs{batch_size}_lr{learning_rate}_ss{step_size}_gamma{gamma}_sw{start_weight}_ew{end_weight}_seqw{sequence_weight}"
        os.makedirs(directory_name, exist_ok=True)

        print(f"Training with batch_size={batch_size}, learning_rate={learning_rate}, step_size={step_size}, gamma={gamma}, start_weight={start_weight}, end_weight={end_weight}, sequence_weight={sequence_weight} in directory {directory_name}")

        # Call the training function with the directory name and hyperparameters
        train_lstm_model.train_lstm_model(
            directory_name=directory_name,
            load_checkpoint='/home/adam/VScodeProjects/Automation/gpt_mouse_move/hyperparam_search_2024_01_27_bs32_lr0.001_ss30_gamma0.5_sw1.0_ew1.0_seqw1.0/checkpoints/model_epoch_9.pth',  # or path to a checkpoint if needed
            save_viz=True,  # Set to True to save visualizations
            num_epochs=100,  # Adjust as needed
            batch_size=batch_size,
            learning_rate=learning_rate,
            step_size=step_size,
            gamma=gamma,
            start_weight=start_weight,
            end_weight=end_weight,
            sequence_weight=sequence_weight,
            checkpoint_interval=20
        )
        # Add code to save or log the results as needed
        # ...

if __name__ == "__main__":
    perform_hyperparameter_grid_search()
