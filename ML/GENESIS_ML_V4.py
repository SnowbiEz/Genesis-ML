# Ensure the correct version of Python is used
# Python 3.8 or 3.9.13 is recommended

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import Sequential
# from keras import LSTM, Dense, Dropout
# from keras import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
BASE_DIR = 'ML'
TEST_DIR = 'ML/test_set'
CATEGORIES = ['normal', 'DDOS', 'port_scan', 'syn_flood', 'icmp_flood', ]  # Example of additional categories
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
            labels = np.array([category] * len(data))
            features = data[['packet_size', 'request_rate']]  # Add other relevant features here
            all_data.append(features)
            all_labels.extend(labels)
    X = pd.concat(all_data, ignore_index=True)  
    y = label_encoder.transform(all_labels)
    y_one_hot = keras.utils.to_categorical(y, num_classes=len(CATEGORIES))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))  # For LSTM input
    return X_scaled, y_one_hot, scaler

# Load combined test data from all categories
def load_combined_test_data(scaler):
    test_data = []
    test_labels = []
    files = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.endswith('.csv')]
    for file_path in files:
        data = pd.read_csv(file_path)
        category = file_path.split('_')[0].split('/')[-1]
        labels = np.array([category] * len(data))
        features = data[['packet_size', 'request_rate']]  # Add other relevant features here
        test_data.append(features)
        test_labels.extend(labels)
    
    # Check if test_data is not empty before concatenating
    if test_data:
        X_test = pd.concat(test_data, ignore_index=True)
        y_test = label_encoder.transform(test_labels)
        y_test_one_hot = keras.utils.to_categorical(y_test, num_classes=len(CATEGORIES))
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))  # For LSTM input
        return X_test_scaled, y_test_one_hot, y_test
    else:
        print("No test data found. Please ensure that the test data files are present in the TEST_DIR.")
        return None, None, None

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

# Evaluate the model on combined test data
X_test, y_test_one_hot, y_test = load_combined_test_data(scaler)
loss, accuracy = model.evaluate(X_test, y_test_one_hot)
print(f"Combined test accuracy: {accuracy * 100:.2f}%")

# Detailed evaluation metrics
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_classes, target_names=CATEGORIES))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Optionally save the model
# model.save('path_to_save_model.h5')
