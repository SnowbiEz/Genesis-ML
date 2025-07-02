import pyshark
import csv
import os
import time
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta

# Dictionary to store timestamps of packets for each source IP
packet_timestamps = defaultdict(list)

def extract_packet_data(packet):
    try:
        # Check if the packet has TCP or UDP layer
        if packet.transport_layer in ['TCP', 'UDP']:
            packet_size = int(packet.length)
            source_ip = packet.ip.src if hasattr(packet, 'ip') else 'N/A'
            destination_ip = packet.ip.dst if hasattr(packet, 'ip') else 'N/A'
            protocol_type = packet.transport_layer
            
            # Record the current timestamp for the source IP
            current_time = datetime.fromtimestamp(float(packet.sniff_timestamp))
            packet_timestamps[source_ip].append(current_time)
            
            # Filter timestamps within the last second
            one_second_ago = current_time - timedelta(seconds=1)
            recent_timestamps = [ts for ts in packet_timestamps[source_ip] if ts >= one_second_ago]
            
            # Update the timestamps list to only keep recent ones
            packet_timestamps[source_ip] = recent_timestamps
            
            # Calculate request rate as the number of packets in the last second
            request_rate = len(recent_timestamps)
            
            return [packet_size, request_rate, source_ip, destination_ip, protocol_type]
    except AttributeError:
        return None  # Skip packets without required fields

async def process_packets_in_file(pcap_file, csv_file):
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['packet_size', 'request_rate', 'source_ip', 'destination_ip', 'protocol_type'])
    
    file_size = os.path.getsize(pcap_file)
    
    while True:
        new_size = os.path.getsize(pcap_file)
        if new_size > file_size:
            cap = pyshark.FileCapture(pcap_file, keep_packets=False, use_json=True)
            
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                for packet in cap:
                    packet_data = extract_packet_data(packet)
                    if packet_data:
                        writer.writerow(packet_data)
            
            cap.close()
            file_size = new_size
        else:
            print("No new packets. Checking again...")
        
        await asyncio.sleep(5)

if __name__ == "__main__":
    pcap_file = r'E:\Docs\CODEOUTPUTS\NEW\Python\genesisResearchProjectTest\test.cap'
    csv_file = r'E:\Docs\CODEOUTPUTS\NEW\Python\genesisResearchProjectTest\output.csv'
    
    asyncio.run(process_packets_in_file(pcap_file, csv_file))
