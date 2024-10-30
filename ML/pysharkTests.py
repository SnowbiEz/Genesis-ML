import pyshark
import os
import time

def display_packet(packet):
    print(f"Packet:" {packet})

def monitor_pcap(file_path):
    cap = pyshark.FileCapture(file_path, keep_packets = False)

    