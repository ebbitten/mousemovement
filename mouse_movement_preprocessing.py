import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import glob
import os
from dotenv import load_dotenv

load_dotenv('Automation/.env')
debug_mode = os.getenv('DEBUG_MODE') == 'TRUE'

def segment_and_standardize_data(df, sequence_length=420, min_length=50):
    """ Segments and standardizes data into fixed-length sequences. """
    click_indices = df[df['click'] == 1].index
    standardized_segments = []
    start_idx = 0

    for end_idx in click_indices:
        if end_idx > start_idx:
            segment = df.iloc[start_idx:end_idx]
            segment_length = len(segment)

            if segment_length >= min_length:
                if segment_length < sequence_length:
                    # Pad the sequence from the last known point
                    last_point = segment.iloc[-1]
                    padding = pd.DataFrame([last_point] * (sequence_length - segment_length), columns=df.columns)
                    segment = pd.concat([segment, padding], ignore_index=True)

                elif segment_length > sequence_length:
                    # Trim the sequence if it's too long
                    segment = segment.iloc[:sequence_length]

                standardized_segments.append(segment)
            start_idx = end_idx + 1

    return standardized_segments


def preprocess_file(file_path):
    """ Preprocesses a single file and returns standardized segments. """
    data = pd.read_csv(file_path)
    data.columns = ['x', 'y', 'timestamp', 'click']
    data['delta_x'] = data['x'].diff().fillna(0)
    data['delta_y'] = data['y'].diff().fillna(0)
    data['delta_time'] = data['timestamp'].diff().fillna(0)
    data['velocity_x'] = data['delta_x'] / data['delta_time']
    data['velocity_y'] = data['delta_y'] / data['delta_time']
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    return segment_and_standardize_data(data)


def plot_mouse_movements(segment, title):
    """ Plot a sequence of mouse movements. """
    plt.plot(segment['x'], segment['y'], marker='o')
    plt.scatter(segment['x'].iloc[0], segment['y'].iloc[0], color='green', label='Start')  # Start point
    plt.scatter(segment['x'].iloc[-1], segment['y'].iloc[-1], color='red', label='End')    # End point
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.legend()
    plt.gca().invert_yaxis()  # Inverting y-axis to match screen coordinates

def perform_sanity_checks(segments):
    """ Perform sanity checks by plotting random segments. """
    num_segments = len(segments)
    
    if num_segments < 4:
        print(f"Not enough segments for sanity checks. Found only {num_segments} segments.")
        return

    plt.figure(figsize=(12, 10))
    for i in range(1, 5):
        segment = random.choice(segments)
        plt.subplot(2, 2, i)
        plot_mouse_movements(segment, f'Segment {i}')
    plt.tight_layout()
    plt.show()


# Directory containing the data files
data_dir = '/home/adam/VScodeProjects/Automation/data/mouse_training'  # Update with the actual path

# Find all CSV files following the pattern DD_MM_YYYY.csv
file_pattern = os.path.join(data_dir, '[0-3][0-9]_[0-1][0-9]_[2][0][0-9][0-9].csv')
all_files = glob.glob(file_pattern)

# Process each file and aggregate the results
all_standardized_segments = []
for file_path in all_files:
    segments = preprocess_file(file_path)
    all_standardized_segments.extend(segments)

# Save the aggregated standardized segments to a pickle file
with open('standardized_segments.pkl', 'wb') as f:
    pickle.dump(all_standardized_segments, f)

print(f"Preprocessing complete. Aggregated standardized segments saved to 'standardized_segments.pkl'")

def inspect_segments(segments):
    """ Inspect the first few segments to confirm that the columns are lining up correctly. """
    print("Inspecting segments...")
    for i, segment in enumerate(segments[:5]):
        print(f"\nSegment {i+1} summary statistics:")
        print(segment[['x', 'y']].describe())
        print(segment[['x', 'y']].head())

def inspect_coordinate_ranges(segments):
    """ Inspect the coordinate ranges to confirm they are within expected screen bounds. """
    print("Inspecting coordinate ranges...")
    for i, segment in enumerate(segments[:5]):
        x_range = (segment['x'].min(), segment['x'].max())
        y_range = (segment['y'].min(), segment['y'].max())
        print(f"Segment {i+1} X range: {x_range}, Y range: {y_range}")



# Debug option to perform sanity checks
debug_mode = True  # Set to False to disable sanity checks

if debug_mode:
    perform_sanity_checks(all_standardized_segments)
    inspect_segments(all_standardized_segments)
    inspect_coordinate_ranges(all_standardized_segments)