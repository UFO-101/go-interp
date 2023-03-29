#%%
import torch
import numpy as np
import requests
import os
import LeelaZero
from LeelaZero import Network
import importlib

#%%
importlib.reload(LeelaZero)
from LeelaZero import Network

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
# %%

# Old code I wrote to load from SGF. Probably redundant now.
#%%
with open("MeVsLeela.sgf", "rb") as f:
    game = sgf.Sgf_game.from_bytes(f.read())
board = boards.Board(game.get_size())

black_tensors = []
white_tensors = []
color = 'b'
for node in game.main_sequence_iter():
    color, move = node.get_move()
    if move is not None:
        board.play(move[0], move[1], color)
    board_array = np.array(board.board)
    white_tensors.append(torch.from_numpy((board_array == 'w').astype(np.float32)))
    black_tensors.append(torch.from_numpy((board_array == 'b').astype(np.float32)))

input_tensors = []
side_to_move_tensors = white_tensors if color == 'b' else black_tensors
for i in range(8):
    try:
        input_tensors.append(side_to_move_tensors[-1 - i])
    except IndexError:
        input_tensors.append(torch.zeros(game.get_size(), game.get_size()))
other_size_tensors = black_tensors if color == 'b' else white_tensors
for i in range(8):
    try:
        input_tensors.append(other_size_tensors[-1 - i])
    except IndexError:
        input_tensors.append(torch.zeros(game.get_size(), game.get_size()))

if color == 'b':
    input_tensors.append(torch.zeros(game.get_size(), game.get_size()))
    input_tensors.append(torch.ones(game.get_size(), game.get_size()))
else:
    input_tensors.append(torch.ones(game.get_size(), game.get_size()))
    input_tensors.append(torch.zeros(game.get_size(), game.get_size()))

input = torch.stack(input_tensors, dim=0).unsqueeze(0).to(device)
print('input', input.shape)

network.to(device)
network.eval()
pol, val = network(input)

print('pol', pol)
print('val', val)


x, y = pol[0].argmax() % game.get_size(), pol[0].argmax() // game.get_size()
print('best move - x:', x, 'y:', y)
print('board value', val.item())
# %%