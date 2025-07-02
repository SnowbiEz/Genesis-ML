import os
import numpy as np
import pandas as pd
import keras
from keras import Sequential
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Constants
BASE_DIR = 'ML'
TEST_DIR = 'ML/test_set'
CATEGORIES = ['normal', 'DDOS', 'port_scan', 'syn_flood', 'icmp_flood']
label_encoder = LabelEncoder()
label_encoder.fit(CATEGORIES)

# Load and preprocess training data
def load_and_preprocess_data():
    all_data = []
    all_labels = []
    for category in CATEGORIES:
        category_dir = os.path.join(BASE_DIR, category)
        files = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.csv')]
        for file_path in files:
            print(f'now training model on the following datasets ', files)
            data = pd.read_csv(file_path)
            labels = np.array([category] * len(data))
            features = data[['packet_size', 'request_rate', '_ws.col.Protocol']]
            all_data.append(features)
            all_labels.extend(labels)

    # Combine data and labels
    X = pd.concat(all_data, ignore_index=True)
    y = label_encoder.transform(all_labels)
    y_one_hot = keras.utils.to_categorical(y, num_classes=len(CATEGORIES))
    
    # Encode the protocol column using LabelEncoder
    protocol_encoder = LabelEncoder()
    X['_ws.col.Protocol'] = protocol_encoder.fit_transform(X['_ws.col.Protocol'])
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))  # Reshape for LSTM input
    
    return X_scaled, y_one_hot, scaler, protocol_encoder

# Preprocess the unlabeled real-world traffic data (test set)
def preprocess_test_data(test_file, scaler, expected_columns, protocol_encoder):
    # Load the test data
    data = pd.read_csv(test_file)
    
    # Rename Columns
    data = data.rename(columns={'frame.time_epoch': 'timestamp', 'frame.len': 'packet_size', 'frame.protocols': '_ws.col.Protocol'})
    
    # Reset index after resampling
    # data.reset_index(drop=True, inplace=True)
    
    # Convert timestamp to datetime and sort by 'ip.src' and 'timestamp'
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    data.sort_values(by=['ip.src', 'timestamp'], inplace=True)

    # Calculate request rate: total count of packets per second for each IP
    data.set_index('timestamp', inplace=True)
    request_rate = data.groupby('ip.src').resample('1s').size().reset_index(name='request_rate')
    request_rate['timestamp'] = request_rate['timestamp'].dt.floor('S')  # Ensure timestamp is floored to the second

    # Merge the request rate back to the original data
    data = data.reset_index().merge(request_rate, on=['ip.src', 'timestamp'], how='left').fillna(0)
    
    # Reset index after resampling
    data.reset_index(drop=True, inplace=True)

    # Extract features
    required_columns = ['packet_size', 'request_rate', 'tcp.dstport', 'udp.dstport', '_ws.col.protocol']
    missing_columns = [col for col in required_columns if col not in data.columns]  

    # Check if all required columns are present
    if missing_columns:
        raise ValueError(f"The following required columns are missing from the test data: {missing_columns}")

    features = data[required_columns]
    
    # Encode the protocol column using the same LabelEncoder
    features['_ws.col.Protocol'] = protocol_encoder.transform(features['_ws.col.Protocol'])
    
    # Align columns with the training data
    features = features.reindex(columns=expected_columns, fill_value=0)

    # Check if features DataFrame is empty
    if features.empty:
        raise ValueError("The features DataFrame is empty. Please check the input data.")

    # Scale the features
    features_scaled = scaler.transform(features)
    features_scaled = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))  # Reshape for LSTM input
    
    return features_scaled

# Define the LSTM model
def create_model(input_shape, num_classes):
    model = Sequential([
        keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Training and validation
X_train, y_train, scaler, protocol_encoder = load_and_preprocess_data()
model = create_model((X_train.shape[1], X_train.shape[2]), len(CATEGORIES))
history = model.fit(X_train, y_train, epochs=1, validation_split=0.3)

# Use the trained model to predict the type of traffic in the real-world dataset (test set)
real_world_test_file = os.path.join(TEST_DIR, 'data.csv')
expected_columns = pd.get_dummies(pd.DataFrame(columns=['packet_size', 'request_rate', 'tcp.dstport', 'udp.dstport', '_ws.col.protocol'])).columns.tolist()
X_test_scaled = preprocess_test_data(real_world_test_file, scaler, expected_columns, protocol_encoder)

# Predict the categories for the real-world test data
y_pred = model.predict(X_test_scaled)

# Map predictions to categories and display results
predicted_classes = np.argmax(y_pred, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes)

for idx, label in enumerate(predicted_labels):  
    print(f"Packet {idx + 1}: Predicted Category: {label}")

# Visualize traffic distribution by category
traffic_counts = pd.Series(predicted_labels).value_counts()
print(traffic_counts)