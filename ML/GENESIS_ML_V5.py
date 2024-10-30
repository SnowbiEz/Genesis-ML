import tensorflow as tf
import pyshark
import os
import time

print("TensorFlow version:", tf.__version__)

def process_packet(packet):
    print(f"Packet: {packet}")



def monitor_pcap(file_path):
    cap = pyshark.FileCapture(file_path, keep_packets=False)
    cap.load_packets(timeout=5)  # Initial sniff to catch up with existing packets

    file_size = os.path.getsize(file_path)

    while True:
        new_size = os.path.getsize(file_path)
        if new_size > file_size:
            # New packets have been added
            cap = pyshark.FileCapture(file_path, keep_packets=False)
            for packet in cap:
                process_packet(packet)
            file_size = new_size
        time.sleep(1)

if __name__ == "__main__":
    # Set the full path to the pcap file
    pcap_file = r'E:\Docs\CODEOUTPUTS\NEW\Python\genesisResearchProjectML\Genesis-ML\network_capture.pcap'
    
    monitor_pcap(pcap_file)