import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import Sequential
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
            data = pd.read_csv(file_path)

            # # Calculate request rate: requests per second from the same IP source
            # data['timestamp'] = pd.to_datetime(data['frame.time'])
            # data.sort_values(by='timestamp', inplace=True)  # Sort by timestamp for accurate calculations
            # data['time_diff'] = data['timestamp'].diff().dt.total_seconds()  # Time diff in seconds
            # data['time_diff'].fillna(0, inplace=True)  # Fill NaN with 0 for first entry
            # data['request_rate'] = data.groupby('source_ip')['time_diff'].transform(lambda x: 1 / x.replace(0, np.nan).mean()).fillna(0)
            
            # Extract and scale relevant features
            features = data[['packet_size', 'request_rate']]  # Add any other relevant features here
            labels = np.array([category] * len(data))
            all_data.append(features)
            all_labels.extend(labels)
    
    # Combine data and labels
    X = pd.concat(all_data, ignore_index=True)
    y = label_encoder.transform(all_labels)
    y_one_hot = keras.utils.to_categorical(y, num_classes=len(CATEGORIES))
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))  # Reshape for LSTM input
    
    return X_scaled, y_one_hot, scaler

# Preprocess the unlabeled real-world traffic data (test set)
def preprocess_test_data(test_file, scaler):
    # Load the test data from tshark
    data = pd.read_csv(test_file)
    
    # for info in test_file:
    # Rename 'frame.len' to 'packet_size' if it exists
    if 'frame.len' in data.columns:
        data = data.rename(columns={'frame.len': 'packet_size'})
    
    # Verify that 'packet_size' is in the test data
    if 'packet_size' not in data.columns:
        raise ValueError("The 'packet_size' column is missing from the test data.")
    
    # Calculate request rate using packet timestamps if 'frame.time' is present
    if 'frame.time_epoch' in data.columns:
        data['timestamp'] = pd.to_datetime(data['frame.time_epoch'])
        data.sort_values(by='timestamp', inplace=True)
        
        # Compute the time difference between consecutive packets to estimate request rate
        data['time_diff'] = data['timestamp'].diff().dt.total_seconds().fillna(0)
        data['request_rate'] = data['time_diff'].apply(lambda x: 1 / x if x != 0 else 0)
    else:
        raise ValueError("The 'frame.time' column is missing from the test data.")
    
    # Extract and arrange features in the same order as the training data
    features = data[['packet_size', 'request_rate']]
    
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
X_train, y_train, scaler = load_and_preprocess_data()
model = create_model((X_train.shape[1], X_train.shape[2]), len(CATEGORIES))
history = model.fit(X_train, y_train, epochs=50, validation_split=0.3)

# Use the trained model to predict the type of traffic in the real-world dataset (test set)
real_world_test_file = os.path.join(TEST_DIR, 'data.csv')  # Update with the actual test data file
X_test_scaled = preprocess_test_data(real_world_test_file, scaler)

# Predict the categories for the real-world test data
y_pred = model.predict(X_test_scaled)

# The test data has no labels, so we cannot directly evaluate accuracy, but we can classify the traffic
predicted_classes = np.argmax(y_pred, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes)

# Print predictions (showing the predicted category for each packet)
for idx, label in enumerate(predicted_labels):
    print(f"Packet {idx + 1}: Predicted Category: {label}")

# Optionally, you can also visualize the prediction results (e.g., showing traffic distribution by category)
traffic_counts = pd.Series(predicted_labels).value_counts()
traffic_counts.plot(kind='bar', title='Traffic Distribution')
plt.ylabel('Count of Packets')
plt.xlabel('Traffic Category')
plt.show()

# Optionally save the model after training
# model.save('path_to_save_model.h5')