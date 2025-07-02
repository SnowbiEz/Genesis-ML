import os
import numpy as np
import pandas as pd
import keras
from keras import Sequential
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.src.callbacks.early_stopping import EarlyStopping
import matplotlib.pyplot as plt

# Constants
BASE_DIR = 'ML'
TEST_DIR = 'ML/test_set'
CATEGORIES = ['normal', 'DDOS', 'port_scan', 'syn_flood', 'icmp_flood']
FEATURESET = ['packet_size', 'request_rate', '_ws.col.protocol', 'tcp.dstport', 'udp.dstport', 'timestamp']
label_encoder = LabelEncoder()
label_encoder.fit(CATEGORIES)

def load_and_preprocess_data():
    all_data = []
    all_labels = []
    protocol_list = []
    tcp_port_list = []
    udp_port_list = []

    for category in CATEGORIES:
        category_dir = os.path.join(BASE_DIR, category)
        files = [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.csv')]
        
        for file_path in files:
            print(f'Preprocessing dataset: {file_path}')
            data = pd.read_csv(file_path)
            
            # Check if required columns are present
            required_columns = ['packet_size', 'request_rate', '_ws.col.protocol', 'tcp.dstport', 'udp.dstport', 'timestamp']
            if not all(col in data.columns for col in required_columns):
                print(f"Skipping {file_path} due to missing columns")
                continue
            
            data['tcp.dstport'] = data['tcp.dstport'].astype(int)
            data['udp.dstport'] = data['udp.dstport'].astype(int)
            
            # Convert timestamp to seconds since epoch
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data['timestamp'] = data['timestamp'].astype('int64') // 10**9
            
            # Sort data by 'ip.src'
            # data = data.sort_values(by='ip.src')
            
            # Extract relevant features for training, excluding IPs
            features = data[['packet_size', 'request_rate', '_ws.col.protocol', 'tcp.dstport', 'udp.dstport', 'timestamp']]
            protocol_list.extend(features['_ws.col.protocol'].unique())
            tcp_port_list.extend(features['tcp.dstport'].unique())
            udp_port_list.extend(features['udp.dstport'].unique())
            all_data.append(features)
            all_labels.extend([category] * len(data))
    
    if not all_data:
        raise ValueError("No data was loaded. Please check your CSV files and directories.")
    
    # Combine data and labels
    X = pd.concat(all_data, ignore_index=True)
    y = label_encoder.transform(all_labels)
    y_one_hot = keras.utils.to_categorical(y, num_classes=len(CATEGORIES))
    X.to_csv('./ML/test_set/preprocessedTrainingData.csv', index=False)
    
    # Protocol encoding with "unknown" handling
    protocol_list = list(set(protocol_list))
    protocol_list.append('unknown')
    protocol_encoder = LabelEncoder()
    protocol_encoder.fit(protocol_list)
    
    X['_ws.col.protocol'] = X['_ws.col.protocol'].apply(lambda x: x if x in protocol_encoder.classes_ else 'unknown')
    X['_ws.col.protocol'] = protocol_encoder.transform(X['_ws.col.protocol'])

    # TCP port encoding with "unknown" handling
    tcp_port_list = list(set(tcp_port_list))
    tcp_port_list.append('unknown')
    tcp_port_encoder = LabelEncoder()
    tcp_port_encoder.fit(tcp_port_list)
    
    X['tcp.dstport'] = X['tcp.dstport'].apply(lambda x: x if x in tcp_port_encoder.classes_ else 'unknown')
    X['tcp.dstport'] = tcp_port_encoder.transform(X['tcp.dstport'])

    # UDP port encoding with "unknown" handling
    udp_port_list = list(set(udp_port_list))
    udp_port_list.append('unknown')
    udp_port_encoder = LabelEncoder()
    udp_port_encoder.fit(udp_port_list)
    
    X['udp.dstport'] = X['udp.dstport'].apply(lambda x: x if x in udp_port_encoder.classes_ else 'unknown')
    X['udp.dstport'] = udp_port_encoder.transform(X['udp.dstport'])
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))  # Reshape for LSTM input
    
    return X_scaled, y_one_hot, scaler, protocol_encoder, tcp_port_encoder, udp_port_encoder

# Preprocess the test data with extended protocol handling
def preprocess_test_data(test_file, scaler, expected_columns):
    # Load the test data
    data = pd.read_csv(test_file)
    
    # Rename Columns
    data = data.rename(columns={'frame.time_epoch': 'timestamp', 'frame.len': 'packet_size', 'frame.protocols': '_ws.col.protocol'})
    
    # Convert timestamps to seconds since the start of the day
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    data['seconds_of_day'] = data['timestamp'].dt.hour * 3600 + data['timestamp'].dt.minute * 60 + data['timestamp'].dt.second
    data['timestamp'] = data['timestamp'].astype('int64') // 10**9

    # Calculate request rate grouped by IP and 1-second intervals
    request_rate_df = data.groupby(['ip.src', 'seconds_of_day']).size().reset_index(name='request_rate')
    
    # Merge request rates back into the original dataframe
    data = data.merge(request_rate_df, on=['ip.src', 'seconds_of_day'], how='left')
    data.drop(columns=['seconds_of_day'], inplace=True)
    
    # Remove entries with user IPs
    for ip in ["192.168.1.173", "1.1.1.1", "1.0.0.1"]:
        ip_indices = data[data['ip.src'] == ip].index
        if not ip_indices.empty:
            print(f"Dropped entries for IP {ip}: {ip_indices}")
            data = data.drop(ip_indices)
    
    # Encode categorical features
    data['_ws.col.protocol'] = data['_ws.col.protocol'].apply(lambda x: x if x in protocol_encoder.classes_ else 'unknown')
    data['_ws.col.protocol'] = protocol_encoder.transform(data['_ws.col.protocol'])
    data['tcp.dstport'] = data['tcp.dstport'].apply(lambda x: x if x in tcp_port_encoder.classes_ else 'unknown')
    data['tcp.dstport'] = tcp_port_encoder.transform(data['tcp.dstport'])
    data['udp.dstport'] = data['udp.dstport'].apply(lambda x: x if x in udp_port_encoder.classes_ else 'unknown')
    data['udp.dstport'] = udp_port_encoder.transform(data['udp.dstport'])
    
    save_preprocessedData = pd.DataFrame(data)
    save_preprocessedData.to_csv('./ML/test_set/preprocessedData.csv', index=False)
    
    # Align with expected columns, exclude timestamp, and scale
    features = data[['packet_size', 'request_rate', '_ws.col.protocol', 'tcp.dstport', 'udp.dstport', 'timestamp']]
    features = features.reindex(columns=expected_columns, fill_value=0)  # Align with expected columns
    features_scaled = scaler.transform(features)
    features_scaled = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))
    
    return features_scaled

# Define the LSTM model
def create_model(input_shape, num_classes):
    model = Sequential([
        keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.5),
        keras.layers.LSTM(32, return_sequences=False, activation='tanh', recurrent_activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Training and validation
X_train, y_train, scaler, protocol_encoder, tcp_port_encoder, udp_port_encoder = load_and_preprocess_data()
model = create_model((X_train.shape[1], X_train.shape[2]), len(CATEGORIES))

# Early stopping callback
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# include early stopping in the callback function if needed for the validation phase
trainmodel = model.fit(X_train, y_train, epochs=50, validation_split=0.5, callbacks=[])

# Use the trained model to predict the type of traffic in the real-world dataset (test set)
real_world_test_file = os.path.join(TEST_DIR, 'data.csv')
expected_columns = ['packet_size', 'request_rate', '_ws.col.protocol', 'tcp.dstport', 'udp.dstport', 'timestamp']
X_test_scaled = preprocess_test_data(real_world_test_file, scaler, expected_columns)
y_pred = model.predict(X_test_scaled)

# Map predictions to categories and display results
predicted_classes = np.argmax(y_pred, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes)

for idx, label in enumerate(predicted_labels):  
    print(f"Packet {idx + 1}: Predicted Category: {label}")

# Visualize traffic distribution by category
traffic_counts = pd.Series(predicted_labels).value_counts()
print(traffic_counts)

# Plot the accuracy
# plt.figure(figsize=(10, 6))
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
# plt.show()
