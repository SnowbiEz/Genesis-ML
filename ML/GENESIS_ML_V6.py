import os
import numpy as np
import pandas as pd
import keras
from keras import Sequential
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Constants
BASE_DIR = 'ML'
TEST_DIR = 'ML/test_set'
CATEGORIES = ['normal', 'DDOS', 'port_scan', 'syn_flood', 'icmp_flood']
label_encoder = LabelEncoder()
label_encoder.fit(CATEGORIES)

# Load and preprocess training data
# Load and preprocess training data without calculating request rate
def load_and_preprocess_data():
    all_data = []
    all_labels = []
    for category in CATEGORIES:
        category_dir = os.path.join(BASE_DIR, category)
        files = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.csv')]
        for file_path in files:
            data = pd.read_csv(file_path)
            
            # Extract relevant features directly from the dataset
            if 'packet_size' not in data.columns or 'request_rate' not in data.columns:
                raise ValueError(f"Required columns are missing in the file {file_path}")
            else:
                features = data[['packet_size', 'request_rate']]
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
    # Load the test data
    data = pd.read_csv(test_file)
    
    # Rename columns
    data = data.rename(columns={'frame.time_epoch': 'timestamp', 'frame.len': 'packet_size', '_ws.col.protocol': 'protocol'})
    
    # Verify that required columns are present
    if 'packet_size' not in data.columns:
        raise ValueError("The 'packet_size' column is missing from the test data.")
    if 'timestamp' not in data.columns:
        raise ValueError("The 'timestamp' column is missing from the test data.")

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
    
    # Extract features and scale them
    features = data[['packet_size', 'request_rate']]
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
history = model.fit(X_train, y_train, epochs=500, validation_split=0.3)

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Use the trained model to predict the type of traffic in the real-world dataset (test set)
real_world_test_file = os.path.join(TEST_DIR, 'data.csv')
X_test_scaled = preprocess_test_data(real_world_test_file, scaler)

# Predict the categories for the real-world test data
y_pred = model.predict(X_test_scaled)

# Map predictions to categories and display results
predicted_classes = np.argmax(y_pred, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes)

for idx, label in enumerate(predicted_labels):  
    print(f"Packet {idx + 1}: Predicted Category: {label}")

# Visualize traffic distribution by category
traffic_counts = pd.Series(predicted_labels).value_counts()
traffic_counts.plot(kind='bar', title='Traffic Distribution')

# for idx, label in enumerate(predicted_labels):  
#     print(f"Packet {idx + 1}: Predicted Category: {label}")

traffic_counts = pd.Series(predicted_labels).value_counts()
print(traffic_counts)

plt.ylabel('Count of Packets')
plt.xlabel('Traffic Category')
plt.show()

# Save the trained model if needed
# model.save('path_to_save_model.h5')