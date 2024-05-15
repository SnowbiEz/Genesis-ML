import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Constants
BASE_DIR = 'ML'
CATEGORIES = ['normal', 'DDOS', 'port_scan']
label_encoder = LabelEncoder()
label_encoder.fit(CATEGORIES)  # Pre-fit label encoder to categories

# Helper function to load and preprocess data
def load_and_preprocess_data():
    all_data = []
    all_labels = []
    for category in CATEGORIES:
        category_dir = os.path.join(BASE_DIR, category)
        files = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.csv')]
        for file_path in files:
            data = pd.read_csv(file_path)
            labels = np.array([category] * len(data))
            all_data.append(data)
            all_labels.extend(labels)
    
    all_data = pd.concat(all_data, ignore_index=True)
    all_labels = np.array(all_labels)

    # Preprocessing
    feature_columns = ['packet_size', 'request_rate']
    X = all_data[feature_columns]
    y = label_encoder.transform(all_labels)
    y_one_hot = to_categorical(y, num_classes=len(CATEGORIES))

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for LSTM input
    X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
    return X_scaled, y_one_hot

# Define the model
def create_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess data from all categories
X, y = load_and_preprocess_data()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = create_model((X_train.shape[1], X_train.shape[2]), len(CATEGORIES))
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Overall test accuracy: {accuracy * 100:.2f}%")

# Plotting results
plt.figure(figsize=(10, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Save model if needed
# model.save('path_to_save')
