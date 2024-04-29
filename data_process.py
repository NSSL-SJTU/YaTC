import os
import glob
import binascii
from PIL import Image
import scapy.all as scapy
from tqdm import tqdm
import numpy as np

def makedir(path):
    try:
        os.mkdir(path)
    except Exception as E:
        pass


def read_MFR_bytes(pcap_dir):
    packets = scapy.rdpcap(pcap_dir)
    data = []
    for packet in packets:
        header = (binascii.hexlify(bytes(packet['IP']))).decode()
        try:
            payload = (binascii.hexlify(bytes(packet['Raw']))).decode()
            header = header.replace(payload, '')
        except:
            payload = ''
        if len(header) > 160:
            header = header[:160]
        elif len(header) < 160:
            header += '0' * (160 - len(header))
        if len(payload) > 480:
            payload = payload[:480]
        elif len(payload) < 480:
            payload += '0' * (480 - len(payload))
        data.append((header, payload))
        if len(data) >= 5:
            break
    if len(data) < 5:
        for i in range(5-len(data)):
            data.append(('0'*160, '0'*480))
    final_data = ''
    for h, p in data:
        final_data += h
        final_data += p
    return final_data

def MFR_generator(flows_pcap_path, output_path):
    flows = glob.glob(flows_pcap_path + "/*/*/*.pcap")
    makedir(output_path)
    makedir(output_path + "/train")
    makedir(output_path + "/test")
    classes = glob.glob(flows_pcap_path + "/*/*")
    for cla in tqdm(classes):
        makedir(cla.replace(flows_pcap_path, output_path))
    for flow in tqdm(flows):
        content = read_MFR_bytes(flow)
        content = np.array([int(content[i:i + 2], 16) for i in range(0, len(content), 2)])
        fh = np.reshape(content, (40, 40))
        fh = np.uint8(fh)
        im = Image.fromarray(fh)
        im.save(flow.replace('.pcap', '.png').replace(flows_pcap_path, output_path))
