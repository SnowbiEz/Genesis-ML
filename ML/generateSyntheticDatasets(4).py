import os
import pandas as pd
import random
from datetime import datetime, timedelta

# Helper function to calculate request rates based on timestamps
def calculate_request_rate(data):
    data.sort(key=lambda x: (x['source_ip'], x['timestamp']))
    current_ip = None
    interval_start = None
    request_count = 0
    for i in range(len(data)):
        if data[i]['source_ip'] != current_ip:
            current_ip = data[i]['source_ip']
            interval_start = data[i]['timestamp']
            request_count = 1
        else:
            interval_duration = (data[i]['timestamp'] - interval_start).total_seconds()
            if interval_duration > 0:
                data[i]['request_rate'] = request_count / interval_duration
            request_count += 1
    return data

# Updated function to generate synthetic data with timestamps
def generate_synthetic_data(num_samples, packet_size_range, request_rate_range, num_attackers=None):
    data = []
    for _ in range(num_samples):
        timestamp = datetime.now() + timedelta(seconds=random.uniform(0, num_samples))
        data.append({
            'packet_size': random.randint(*packet_size_range),
            'source_ip': f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, num_attackers or 255)}",
            'destination_ip': "192.168.1.1",        
            'protocol_type': random.choice(["TCP", "UDP", "ICMP"]),
            'timestamp': timestamp
        })
    return calculate_request_rate(data)

# Functions for different categories
def generate_normal_traffic(num_samples):
    return generate_synthetic_data(num_samples, (64, 1500), (0.1, 10))

def generate_ddos_traffic(num_samples, num_attackers):
    return generate_synthetic_data(num_samples, (64, 1500), (100, 1000), num_attackers)

def generate_port_scan_traffic(num_samples):
    return generate_synthetic_data(num_samples, (40, 100), (0.1, 1))

def generate_syn_flood_traffic(num_samples, num_attackers):
    return generate_synthetic_data(num_samples, (40, 100), (100, 1000), num_attackers)

def generate_icmp_flood_traffic(num_samples):
    return generate_synthetic_data(num_samples, (40, 100), (100, 1000))

# Directory setup
base_dir = 'ML'
categories = ['normal', 'DDOS', 'port_scan', 'syn_flood', 'icmp_flood']

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for category in categories:
    category_dir = os.path.join(base_dir, category)
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)
    
    # Next file index and data generation
    existing_files = [f for f in os.listdir(category_dir) if f.startswith(category)]
    next_index = int(max([f.split('_')[-1].split('.')[0] for f in existing_files], default=0)) + 1

    num_datasets = 3
    for i in range(next_index, next_index + num_datasets):
        if category == 'normal':
            data = generate_normal_traffic(1000)
        elif category == 'DDOS':
            data = generate_ddos_traffic(200, 50)
        elif category == 'port_scan':
            data = generate_port_scan_traffic(200)
        elif category == 'syn_flood':
            data = generate_syn_flood_traffic(200, 50)
        elif category == 'icmp_flood':
            data = generate_icmp_flood_traffic(200)
        
        df = pd.DataFrame(data).drop(columns='timestamp')
        filename = os.path.join(category_dir, f"{category}_dataset_{i}.csv")
        df.to_csv(filename, index=False)

print("Datasets with time-based request rates generated and stored.")
