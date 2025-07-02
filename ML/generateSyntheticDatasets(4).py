import os
import pandas as pd
import random

def generate_normal_traffic(num_samples):
    data = [{'packet_size': random.randint(64, 1500),
             'request_rate': random.randint(1, 5),  # Lower request rate as integer
             'ip.src': f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
             'ip.dst': "192.168.1.1",
             '_ws.col.protocol': random.choice(["TCP", "UDP", "ICMP"]),
             'tcp.dstport': random.choice([80, 443, 8080]) if random.choice(["TCP", "UDP"]) == "TCP" else None,
             'udp.dstport': 53 if random.choice(["TCP", "UDP"]) == "UDP" else None}
            for _ in range(num_samples)]
    return data

def generate_ddos_traffic(num_samples, num_attackers):
    data = [{'packet_size': random.randint(500, 1500),
             'request_rate': random.randint(100, 1000),  # Request rate as integer
             'ip.src': f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, num_attackers)}",
             'ip.dst': "192.168.1.1",
             '_ws.col.protocol': "TCP",
             'tcp.dstport': 80,  # Common target for DDoS
             'udp.dstport': None}
            for _ in range(num_samples)]
    return data

def generate_port_scan_traffic(num_samples):
    data = [{'packet_size': random.randint(40, 100),
             'request_rate': random.randint(1, 10),  # Request rate as integer
             'ip.src': f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
             'ip.dst': "192.168.1.1",
             '_ws.col.protocol': random.choice(["TCP", "UDP"]),
             'tcp.dstport': random.randint(20, 1024) if random.choice(["TCP", "UDP"]) == "TCP" else None,
             'udp.dstport': random.randint(20, 1024) if random.choice(["TCP", "UDP"]) == "UDP" else None}
            for _ in range(num_samples)]
    return data

def generate_syn_flood_traffic(num_samples, num_attackers):
    data = [{'packet_size': random.randint(40, 100),  # Typical SYN packet size
             'request_rate': random.randint(100, 1000),  # Request rate as integer
             'ip.src': f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, num_attackers)}",
             'ip.dst': "192.168.1.1",
             '_ws.col.protocol': "TCP",
             'tcp.dstport': 80,  # Targeting common web server port
             'udp.dstport': None}
            for _ in range(num_samples)]
    return data

def generate_icmp_flood_traffic(num_samples):
    data = [{'packet_size': random.randint(40, 100),  # Typical ICMP echo packet size
             'request_rate': random.randint(100, 1000),  # Request rate as integer
             'ip.src': f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
             'ip.dst': "192.168.1.1",
             '_ws.col.protocol': "ICMP",
             'tcp.dstport': None,
             'udp.dstport': None}
            for _ in range(num_samples)]
    return data

# Set base directory to your ML folder
base_dir = 'ML'
categories = ['normal', 'DDOS', 'port_scan', 'syn_flood', 'icmp_flood']

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
        elif category == 'syn_flood':
            data = generate_syn_flood_traffic(200, 50)
        elif category == 'icmp_flood':
            data = generate_icmp_flood_traffic(200)
        
        df = pd.DataFrame(data)
        filename = os.path.join(category_dir, f"{category}_dataset_{i}.csv")
        df.to_csv(filename, index=False)

print("Datasets generated and stored.")