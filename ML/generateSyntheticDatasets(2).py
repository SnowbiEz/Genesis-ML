import os
import pandas as pd
import random
import datetime

# Function to generate a list of consistent IP addresses
def generate_ip_list(num_ips):
    return [f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{i}" for i in range(1, num_ips + 1)]

# Function to generate normal traffic
def generate_normal_traffic(num_samples, ip_list):
    protocols = {
        "TCP": [80, 443, 8080],
        "UDP": [53, 67, 123],
        "ICMP": [-1],
        "TLSv1.2": [443],
        "TLSv1.3": [443],
        "QUIC": [443],
        "HTTP": [80],
        "HTTPS": [443],
        "DNS": [53],
        "ICMPv6": [-1],
    }
    data = []
    start_time = datetime.datetime.now()
    for _ in range(num_samples):
        protocol = random.choice(list(protocols.keys()))
        port_choices = protocols[protocol]
        tcp_port = random.choice(port_choices) if protocol in ["TCP", "TLSv1.2", "TLSv1.3", "QUIC", "HTTP", "HTTPS"] else -1
        udp_port = random.choice(port_choices) if protocol == "UDP" or protocol == "DNS" else -1
        timestamp = start_time + datetime.timedelta(seconds=random.randint(0, 1000))
        ip_src = random.choice(ip_list)
        data.append({
            'timestamp': timestamp,
            'packet_size': random.randint(64, 1500),
            'request_rate': 0,  # Placeholder, will be calculated later
            'ip.src': ip_src,
            'ip.dst': "192.168.1.1",
            '_ws.col.protocol': protocol,
            'tcp.dstport': tcp_port,
            'udp.dstport': udp_port
        })
    return data

def generate_port_scan_traffic(num_samples, ip_list):
    protocols = ["TCP", "UDP", "ICMPv6", "DNS"]
    data = []
    start_time = datetime.datetime.now()
    for _ in range(num_samples):
        protocol = random.choice(protocols)
        tcp_port = random.randint(20, 1024) if protocol == "TCP" else -1
        udp_port = random.randint(20, 1024) if protocol == "UDP" else -1
        timestamp = start_time + datetime.timedelta(seconds=random.randint(0, 1000))
        ip_src = random.choice(ip_list)
        data.append({
            'timestamp': timestamp,
            'packet_size': random.randint(40, 100),
            'request_rate': 0,
            'ip.src': ip_src,
            'ip.dst': "192.168.1.1",
            '_ws.col.protocol': protocol,
            'tcp.dstport': tcp_port,
            'udp.dstport': udp_port
        })
    return data

def generate_ddos_traffic(num_samples, ip_list):
    protocols = ["TCP", "TLSv1.2", "TLSv1.3", "QUIC", "HTTPS"]
    data = []
    start_time = datetime.datetime.now()
    
    for _ in range(num_samples):
        protocol = random.choice(protocols)
        timestamp = start_time + datetime.timedelta(milliseconds=random.randint(0, 100))  # Denser timestamps
        ip_src = random.choice(ip_list)
        
        data.append({
            'timestamp': timestamp,
            'packet_size': random.randint(500, 1500),
            'request_rate': 0,  # Placeholder for calculation later
            'ip.src': ip_src,
            'ip.dst': "192.168.1.1",
            '_ws.col.protocol': protocol,
            'tcp.dstport': 80 if protocol in ["TCP", "HTTPS", "TLSv1.2", "TLSv1.3"] else -1,
            'udp.dstport': -1
        })
    
    return data

def generate_syn_flood_traffic(num_samples, ip_list):
    data = []
    start_time = datetime.datetime.now()
    
    for _ in range(num_samples):
        timestamp = start_time + datetime.timedelta(milliseconds=random.randint(0, 100))  # Denser timestamps
        ip_src = random.choice(ip_list)
        
        data.append({
            'timestamp': timestamp,
            'packet_size': random.randint(40, 100),
            'request_rate': 0,
            'ip.src': ip_src,
            'ip.dst': "192.168.1.1",
            '_ws.col.protocol': "TCP",
            'tcp.dstport': 80,
            'udp.dstport': -1
        })
    
    return data

def generate_icmp_flood_traffic(num_samples, ip_list):
    protocols = ["ICMP", "ICMPv6"]
    data = []
    start_time = datetime.datetime.now()
    
    for _ in range(num_samples):
        protocol = random.choice(protocols)
        timestamp = start_time + datetime.timedelta(milliseconds=random.randint(0, 100))  # Denser timestamps
        ip_src = random.choice(ip_list)
        
        data.append({
            'timestamp': timestamp,
            'packet_size': random.randint(40, 100),
            'request_rate': 0,
            'ip.src': ip_src,
            'ip.dst': "192.168.1.1",
            '_ws.col.protocol': protocol,
            'tcp.dstport': -1,
            'udp.dstport': -1
        })
    
    return data

def calculate_request_rate(data):
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp_rounded'] = df['timestamp'].dt.floor('1s')  # Round to 1-second intervals

    # Group by IP and 1-second intervals, then calculate the request rate
    request_rate_df = df.groupby(['ip.src', 'timestamp_rounded']).size().reset_index(name='request_rate')
    
    # Merge the calculated request rate back into the original DataFrame
    df = pd.merge(df, request_rate_df, on=['ip.src', 'timestamp_rounded'], how='left')
    
    # Clean up columns
    df.drop(columns=['timestamp_rounded'], inplace=True)
    
    # Remove any duplicate columns and helper columns
    df = df.drop(columns=['request_rate_x', 'timestamp_rounded'], errors='ignore')
    df = df.rename(columns={'request_rate_y': 'request_rate'})

    return df.to_dict('records')


# Set base directory to your ML folder
base_dir = 'ML'
categories = ['normal', 'DDOS', 'port_scan', 'syn_flood', 'icmp_flood']

# Ensure base directory exists
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Generate a list of IP addresses for clustering and consistency for each dataset
normal_ip_list = generate_ip_list(50)
ddos_ip_list = generate_ip_list(10)  # Smaller set to simulate fewer unique attackers
port_scan_ip_list = generate_ip_list(30)
syn_flood_ip_list = generate_ip_list(20)  # Smaller set to simulate fewer unique attackers
icmp_flood_ip_list = generate_ip_list(20)

# Generate datasets
for category in categories:
    category_dir = os.path.join(base_dir, category)
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)
    
    existing_files = [f for f in os.listdir(category_dir) if f.startswith(category)]
    next_index = int(existing_files[-1].split('_')[-1].split('.')[0]) + 1 if existing_files else 1

    num_datasets = 3
    for i in range(next_index, next_index + num_datasets):
        if category == 'normal':
            data = generate_normal_traffic(500, normal_ip_list)
        elif category == 'port_scan':
            data = generate_port_scan_traffic(200, port_scan_ip_list)
        elif category == 'DDOS':
            data = generate_ddos_traffic(400, ddos_ip_list)
        elif category == 'syn_flood':
            data = generate_syn_flood_traffic(200, ddos_ip_list)
        elif category == 'icmp_flood':
            data = generate_icmp_flood_traffic(200, ddos_ip_list)

        data_with_request_rate = calculate_request_rate(data)
        df = pd.DataFrame(data_with_request_rate)
        filename = os.path.join(category_dir, f"{category}_dataset_{i}.csv")
        df.to_csv(filename, index=False)

print("Datasets generated and stored.")