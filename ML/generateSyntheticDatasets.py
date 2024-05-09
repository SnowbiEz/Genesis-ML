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
num_datasets = 3  # Number of datasets per category

# Ensure base directory exists
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for category in categories:
    category_dir = os.path.join(base_dir, category)
    # Ensure category directory exists
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)
        
    for i in range(1, num_datasets + 1):
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
