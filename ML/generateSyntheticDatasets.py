import os
import pandas as pd
import random

def generate_normal_traffic(num_samples):
    data = [{'packet_size': random.randint(64, 1500),
             'request_rate': random.uniform(0.1, 10),
             'source_ip': f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
             'destination_ip': "192.168.1.1",
             'protocol_type': random.choice(["TCP", "UDP", "ICMP"])}
            for _ in range(num_samples)]
    return data

def generate_ddos_traffic(num_samples, num_attackers):
    data = [{'packet_size': random.randint(64, 1500),
             'request_rate': random.uniform(100, 1000),
             'source_ip': f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, num_attackers)}",
             'destination_ip': "192.168.1.1",
             'protocol_type': "TCP"}
            for _ in range(num_samples)]
    return data

def generate_port_scan_traffic(num_samples):
    data = [{'packet_size': random.randint(40, 100),
             'request_rate': random.uniform(0.1, 1),
             'source_ip': f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
             'destination_ip': "192.168.1.1",
             'protocol_type': random.choice(["TCP", "UDP"])}
            for _ in range(num_samples)]
    return data

# Set base directory to your ML folder
base_dir = 'ML'
categories = ['normal', 'DDOS', 'port_scan']

# Ensure base directory exists
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for category in categories:
    category_dir = os.path.join(base_dir, category)
    # Ensure category directory exists
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)
    
    # Get the next available file index
    existing_files = [f for f in os.listdir(category_dir) if f.startswith(category)]
    if existing_files:
        latest_file = max(existing_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        next_index = int(latest_file.split('_')[-1].split('.')[0]) + 1
    else:
        next_index = 1

    # Generate the number of datasets per category
    num_datasets = 3
    for i in range(next_index, next_index + num_datasets):
        if category == 'normal':
            data = generate_normal_traffic(1000)
        elif category == 'DDOS':
            data = generate_ddos_traffic(200, 50)
        elif category == 'port_scan':
            data = generate_port_scan_traffic(200)
        
        df = pd.DataFrame(data)
        filename = os.path.join(category_dir, f"{category}_dataset_{i}.csv")
        df.to_csv(filename, index=False)

print("Datasets generated and stored.")
