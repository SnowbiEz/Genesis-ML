import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Constants
BASE_DIR = 'ML'
TEST_DIR = 'ML/test_set'
CATEGORIES = ['normal', 'DDOS', 'port_scan']
label_encoder = LabelEncoder()
label_encoder.fit(CATEGORIES)

# Load and preprocess training data
def load_and_preprocess_data():
    all_data = []
    all_labels = []
    for category in CATEGORIES:
        category_dir = os.path.join(BASE_DIR, category)  # Cross-platform path handling
        files = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.csv')]
        for file_path in files:
            data = pd.read_csv(file_path)
            labels = np.array([category] * len(data))
            all_data.append(data[['packet_size', 'request_rate']])
            all_labels.extend(labels)
    X = pd.concat(all_data, ignore_index=True)
    y = label_encoder.transform(all_labels)
    y_one_hot = to_categorical(y, num_classes=len(CATEGORIES))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
    return X_scaled, y_one_hot, scaler

# Define the model
def create_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and evaluate test data for each category
def evaluate_each_category(model, scaler):
    accuracy_results = {}
    for category in CATEGORIES:
        test_files = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.endswith('.csv')]
        for file_path in test_files:
            data = pd.read_csv(file_path)
            category_from_file = os.path.splitext(os.path.basename(file_path))[0]
            print(f"Evaluating {file_path} as category: {category_from_file}...")
            X_test = data[['packet_size', 'request_rate']]
            X_test_scaled = scaler.transform(X_test)
            X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            y_test = np.array([category_from_file] * len(data))
            y_test_encoded = label_encoder.transform(y_test)
            y_test_one_hot = to_categorical(y_test_encoded, num_classes=len(CATEGORIES))
            
            loss, accuracy = model.evaluate(X_test_scaled, y_test_one_hot, verbose=0)
            accuracy_results[category_from_file] = accuracy * 100
            print(f"Accuracy for {category_from_file}: {accuracy * 100:.2f}%")
    return accuracy_results

# Main script
X_train, y_train, scaler = load_and_preprocess_data()
model = create_model((X_train.shape[1], X_train.shape[2]), len(CATEGORIES))
model.fit(X_train, y_train, epochs=10, validation_split=0.2)  # Reduced epochs for quicker testing

# Evaluate
accuracy_results = evaluate_each_category(model, scaler)
print("Final Accuracy Results:", accuracy_results)
