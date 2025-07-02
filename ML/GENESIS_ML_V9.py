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

# Load and preprocess training data with extended protocol handling
def load_and_preprocess_data():
    all_data = []
    all_labels = []
    protocol_list = []

    for category in CATEGORIES:
        category_dir = os.path.join(BASE_DIR, category)
        files = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.csv')]
        
        for file_path in files:
            print(f'Processing dataset: {file_path}')
            data = pd.read_csv(file_path)
            
            # Extract relevant features for training, excluding IPs
            features = data[['packet_size', 'request_rate', '_ws.col.protocol', 'tcp.dstport', 'udp.dstport']]
            protocol_list.extend(features['_ws.col.protocol'].unique())  # Collect all protocols seen in training
            all_data.append(features)
            all_labels.extend([category] * len(data))
    
    # Combine data and labels
    X = pd.concat(all_data, ignore_index=True)
    y = label_encoder.transform(all_labels)
    y_one_hot = keras.utils.to_categorical(y, num_classes=len(CATEGORIES))
    
    # Protocol encoding with "unknown" handling
    protocol_list = list(set(protocol_list))  # Unique protocols in training
    protocol_list.append('unknown')  # Add 'unknown' label for unseen protocols
    protocol_encoder = LabelEncoder()
    protocol_encoder.fit(protocol_list)  # Fit with added 'unknown' label
    
    X['_ws.col.protocol'] = X['_ws.col.protocol'].apply(lambda x: x if x in protocol_encoder.classes_ else 'unknown')
    X['_ws.col.protocol'] = protocol_encoder.transform(X['_ws.col.protocol'])
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))  # Reshape for LSTM input
    
    return X_scaled, y_one_hot, scaler, protocol_encoder

# Preprocess the test data with extended protocol handling
def preprocess_test_data(test_file, scaler, expected_columns, protocol_encoder):
    # Load the test data
    data = pd.read_csv(test_file)
    
    # Rename Columns
    data = data.rename(columns={'frame.time_epoch': 'timestamp', 'frame.len': 'packet_size', 'frame.protocols': '_ws.col.protocol'})
    
    # Calculate request rate per IP per second
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    data['request_rate'] = data.groupby(['ip.src', data['timestamp'].dt.floor('s')])['timestamp'].transform('size')
    
    # Encode the protocol column, handling unseen labels
    data['_ws.col.protocol'] = data['_ws.col.protocol'].apply(lambda x: x if x in protocol_encoder.classes_ else 'unknown')
    data['_ws.col.protocol'] = protocol_encoder.transform(data['_ws.col.protocol'])
    
    # Select and align features, handling missing columns
    features = data[['packet_size', 'request_rate', 'ip.src', 'ip.dst', '_ws.col.protocol', 'tcp.dstport', 'udp.dstport']]
    features = features.reindex(columns=expected_columns, fill_value=0)  # Align with expected columns
    
    # Scale the features for model input
    features_scaled = scaler.transform(features.drop(columns=['ip.src', 'ip.dst']))  # Drop IPs from scaling
    features_scaled = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))
    
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
history = model.fit(X_train, y_train, epochs=200, validation_split=0.3)

# Use the trained model to predict the type of traffic in the real-world dataset (test set)
real_world_test_file = os.path.join(TEST_DIR, 'data.csv')
expected_columns = ['packet_size', 'request_rate', 'ip.src', 'ip.dst', '_ws.col.protocol', 'tcp.dstport', 'udp.dstport']
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
