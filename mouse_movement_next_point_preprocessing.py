def preprocess_mouse_movement_data(file_path):
    # Load the data with the specified headers
    data = pd.read_csv(file_path, header=None, names=['x', 'y', 'time', 'click'])

    # Assuming the final target point is the last point in the dataset
    final_target_point = data.iloc[-1][['x', 'y']].values

    # Feature engineering: calculate the distance to the final target point for each point
    data['distance_to_target_x'] = final_target_point[0] - data['x']
    data['distance_to_target_y'] = final_target_point[1] - data['y']

    # Normalize the features and target separately
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    features = ['x', 'y', 'distance_to_target_x', 'distance_to_target_y']
    target_features = ['x', 'y']
    data[features] = feature_scaler.fit_transform(data[features])
    data[target_features] = target_scaler.fit_transform(data[target_features])

    # Define a custom dataset class for PyTorch
    class MouseMovementDataset(Dataset):
        def __init__(self, data, feature_scaler, target_scaler):
            self.data = data
            self.feature_scaler = feature_scaler
            self.target_scaler = target_scaler

        def __len__(self):
            return len(self.data) - 1

        def __getitem__(self, idx):
            current_features = self.data.iloc[idx][features].values
            target = self.data.iloc[idx + 1][target_features].values
            return current_features, target

    # Create the dataset
    dataset = MouseMovementDataset(data, feature_scaler, target_scaler)

    return dataset, feature_scaler, target_scaler

# Process the data and get the dataset and scalers
dataset, feature_scaler, target_scaler = preprocess_mouse_movement_data(file_path)

# Inspect the first few entries of the dataset
for i in range(5):
    features, target = dataset[i]
    print(f"Features: {features}, Target: {target}")
