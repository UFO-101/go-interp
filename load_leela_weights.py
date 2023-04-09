#%%
import torch
import numpy as np
import requests
import os
import LeelaZero
from LeelaZero import Network
import importlib

#%%
import LeelaZero
importlib.reload(LeelaZero)
from LeelaZero import Network
import requests

url = "https://zero.sjeng.org/best-network"
# url = "https://github.com/yukw777/leela-zero-pytorch/raw/master/weights/leela-zero-pytorch-sm.txt"
cache_dir = ".cache"

# Create cache directory if it doesn't exist
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

filename = os.path.join(cache_dir, os.path.basename(url))
if os.path.exists(filename):
    print("File exists, reading from cache")
else:
    print("File does not exist, downloading from URL and save to cache")
    response = requests.get(url)
    content = response.content
    with open(filename, "wb") as f:
        f.write(content)

board_size = 19
in_channels = 18
residual_channels = 256  # 32
residual_layers = 40  # 8
network = Network(board_size, in_channels, residual_channels, residual_layers)
network.from_leela_weights(filename)

# Check that loading the weights works correctly
# temp_weights_file = os.path.join(cache_dir, "test_weights.txt.gz")
# try:
#     network.cpu().to_leela_weights(temp_weights_file)
#     compare_network = Network(board_size, in_channels, residual_channels, residual_layers)
#     compare_network.from_leela_weights(temp_weights_file)
#     for name, param in network.named_parameters():
#         try:
#             assert torch.allclose(param, compare_network.state_dict()[name])
#         except AssertionError:
#             print(f"Parameter {name} does not match")
#             print(f"Original: {param}")
#             print(f"Loaded: {compare_network.state_dict()[name]}")
# finally:
#     os.remove(temp_weights_file)

#%%
# Save the network to a file
torch.save(network.state_dict(), ".cache/best-network.pt")