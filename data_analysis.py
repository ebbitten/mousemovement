import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from dotenv import load_dotenv
import os

# Load the environment variables from .env file
load_dotenv()

# Function to load and preprocess data
def load_data(file_pattern):
    all_files = glob.glob(file_pattern)
    all_data = []

    for file in all_files:
        data = pd.read_csv(file)
        data.columns = ['x', 'y', 'timestamp', 'click']
        all_data.append(data)

    return pd.concat(all_data, ignore_index=True)

# Function to analyze start and end points
def analyze_points(data):
    start_points = data[data['click'] == 1][['x', 'y']]  # Assuming a click indicates a start point
    end_points = data[data['click'] == 1][['x', 'y']].shift(-1)  # Assuming the point after a click is an end point

    # Remove NaN values (which may appear due to the shift operation)
    end_points = end_points.dropna()

    return start_points, end_points

# Function to plot data
def plot_data(start_points, end_points):
    plt.figure(figsize=(12, 6))

    # Plot start points
    plt.subplot(1, 2, 1)
    plt.scatter(start_points['x'], start_points['y'], color='blue', alpha=0.5)
    plt.title('Start Points')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Plot end points
    plt.subplot(1, 2, 2)
    plt.scatter(end_points['x'], end_points['y'], color='red', alpha=0.5)
    plt.title('End Points')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    plt.tight_layout()
    plt.show()

# Main function
def main():
    directory = os.getenv('MOUSE_TRAINING_DIR')  # Update this with the path to your data files
    file_pattern = os.path.join(directory, '*.csv')
    data = load_data(file_pattern)
    start_points, end_points = analyze_points(data)
    plot_data(start_points, end_points)

    # Calculate and print mean and median
    print("Start Points Mean:", start_points.mean())
    print("Start Points Median:", start_points.median())
    print("End Points Mean:", end_points.mean())
    print("End Points Median:", end_points.median())

if __name__ == "__main__":
    main()
