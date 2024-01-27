# perform_hyperparameter_grid_search.py
import datetime
import train_lstm_model

def perform_hyperparameter_grid_search():
    batch_sizes = [32, 64]
    learning_rates = [0.001, 0.0005]
    step_sizes = [30, 50]
    gammas = [0.5, 0.7]

    for batch_size in batch_sizes:
        for lr in learning_rates:
            for step_size in step_sizes:
                for gamma in gammas:
                    directory_name = f"{datetime.datetime.now().strftime('%Y_%m_%d')}_{batch_size}_{lr}_{step_size}_{gamma}"
                    train_lstm_model.train_lstm_model(batch_size=batch_size, learning_rate=lr, step_size=step_size, gamma=gamma, directory_name=directory_name)

if __name__ == "__main__":
    perform_hyperparameter_grid_search()
