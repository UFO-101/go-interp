#%%
import torch
from LeelaZero import Network
from parse_leela_data import Dataset
import plotly.express as px
import pandas as pd
from einops import rearrange
import importlib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
memory_available = torch.cuda.get_device_properties(device).total_memory
print("Device: ", device, "- Memory available: ", memory_available)

def print_go_board_tensor(tensor, turn_number, show_n_turns_past=0, only_to_play=False, coords=None):
    tensor = tensor[0]
    color_to_play = 'b' if turn_number % 2 == 0 else 'w'
    print("Color to play:", color_to_play)
    for i in reversed(range(19)):
        for j in range(19):
            if coords is not None and (j, i) in coords:
                black_symbol = "🔴"
                white_symbol = "🟡"
                empty_symbol = "❌"
            else:
                black_symbol = "🟤"
                white_symbol = "⚪"
                empty_symbol = "➕"
            if tensor[0 + show_n_turns_past][i][j] == 1:
                print(black_symbol, end="") if color_to_play == 'b' else print(white_symbol, end="")
            elif tensor[8 + show_n_turns_past][i][j] == 1 and not only_to_play:
                print(white_symbol, end="") if color_to_play == 'b' else print(black_symbol, end="")
            else:
                print(empty_symbol, end="")
        print()

def show_layer_activations(activations, boards_width=16, boards_height=16):
    # output[0] has shape (16 * 16, 19, 19). Pad each 19x19 with a surrounding border of zeros.
    print('activations shape:', activations.shape)
    output_padded = torch.zeros((boards_width * boards_height, 20, 20)) - 1
    output_padded[:, :-1, :-1] = activations.detach()[0] if activations.shape[0] == 1 else activations.detach()
    output_padded = rearrange(output_padded, '(i j) h w -> (i h) (j w)', i=boards_width)
    px.imshow(output_padded.cpu().numpy(), origin="lower", height=800).show()
    

#%%
import LeelaZero
importlib.reload(LeelaZero)
from LeelaZero import Network
layer_count = 40
channel_count = 256
residualify = False
network = Network(19, 18, channel_count, layer_count, residualify=residualify)
network.load_state_dict(torch.load(".cache/best-network.pt"))
network.eval()
network.to(device)
print("Network loaded.")

#%%
dataset = Dataset(["game-data/better-long-range-atari.txt.0.gz"])
print("Number of input:", len(dataset))
#%%
turn_number = 50
print_go_board_tensor(dataset[turn_number], turn_number, show_n_turns_past=0, only_to_play=False)

#%%
# SHOW ACTIVATIONS and POLICY
network.eval()
input = dataset[turn_number][0].unsqueeze(0).to(device).requires_grad_(False)
print('input', input.shape)

def display_module_output(module, input, output):
    show_layer_activations(output)
    
channel_to_view = 166
channel_activations = []
for i in range(layer_count):
    # handle = network.residual_tower[i].register_forward_hook(display_module_output)
    try:
        with torch.inference_mode():
            pol, val, activations = network(input, get_layer_activations=i)
            channel_activations.append(activations[0, channel_to_view].detach().clone())
    finally:
        # handle.remove()
        pass
activations_combined = torch.stack(channel_activations)
print("Activations across layers for channel", channel_to_view)
show_layer_activations(activations_combined, boards_width=8, boards_height=5)

print("POLICY")
px.imshow(pol[0][:-1].reshape(19, 19).cpu().detach().numpy(), origin="lower").show()
print(f'max pol x:{pol[0].argmax() % 19}, y:{pol[0].argmax() // 19}.')
print(f'Board value: {((1 + val.item()) / 2.0) * 100}%')

#%%

# print('network.conv_input.conv.weight', network.conv_input.conv.weight.shape)
# px.imshow(network.conv_input.conv.weight[166].detach().cpu().numpy(), origin="lower", facet_col=0, facet_col_wrap=4).show()
resid_block = 0
print(f'network.residual_tower[{resid_block}].conv1.conv.weight', network.residual_tower[resid_block].conv1.conv.weight.shape)
# weights_to_plot = network.residual_tower[resid_block].conv1.conv.weight
weights_to_plot = network.conv_input.conv.weight
channel_to_plot = 166
print('weights_to_plot', weights_to_plot.shape)
fig = px.imshow(weights_to_plot[channel_to_plot].detach().cpu().numpy(), origin="lower", facet_col=0, facet_col_wrap=16, facet_col_spacing=0.001, facet_row_spacing=0.01, width=1000, height=1000)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("_")[-1]))
fig.show()

#%%
# ACTIVATION PATCHING
atari_turn = 50
print_go_board_tensor(dataset[atari_turn], atari_turn, show_n_turns_past=0, only_to_play=False)
safe_turn = 54
print_go_board_tensor(dataset[safe_turn], safe_turn, show_n_turns_past=0, only_to_play=False)
atari_input = dataset[atari_turn][0].unsqueeze(0).to(device).requires_grad_(False)
safe_input = dataset[safe_turn][0].unsqueeze(0).to(device).requires_grad_(False)
atari_pol_unpatched, _, _ = network(atari_input)
safe_pol_unpatched, _, _ = network(safe_input)
atari_escape_coords = (9, 15)
atari_escape_index = atari_escape_coords[1] * 19 + atari_escape_coords[0]

def plot_atari_escape_pol(*args, **kwargs):
    fig = px.line(*args, **kwargs)
    fig.add_hline(y=atari_pol_unpatched[0, atari_escape_index].item(), line_dash="dash", annotation_text="atari policy")
    fig.add_hline(y=safe_pol_unpatched[0, atari_escape_index].item(), line_dash="dash", annotation_text="safe policy")
    fig.show()

#%%
# LAYER PATCHING
network.eval()
layer_patching_vals = []
for layer_index in range(layer_count):
    atari_policy, _, atari_activations = network(atari_input, get_layer_activations=layer_index)
    patch_hook = lambda module, input, output: atari_activations
    handle = network.residual_tower[layer_index].register_forward_hook(patch_hook)
    try:
        with torch.inference_mode():
            pol, val, _ = network(safe_input)
            layer_patching_vals.append(pol[0, atari_escape_index].item())
    finally:
        handle.remove()
        pass

plot_atari_escape_pol(y=layer_patching_vals, labels={'x': 'Layer', 'y': 'Policy'})
#%%
# CHANNEL PATCHING
network.eval()
def patch_activations(module, input, output, atari_activations, channel):
    # Replace output with atari_activations at the given channel.
    output[:, channel] = atari_activations[:, channel]
    return output

# Create a pandas dataframe. Each row is a layer, each column is a channel.
layer_and_channel_patching_vals = pd.DataFrame(columns=range(channel_count))
for layer_index in range(layer_count):
    _, _, atari_activations = network(atari_input, get_layer_activations=layer_index)
    for channel_index in range(channel_count):
        patch_hook = lambda m, i, o: patch_activations(m, i, o, atari_activations, channel_index)
        handle = network.residual_tower[layer_index].register_forward_hook(patch_hook)
        try:
            with torch.inference_mode():
                pol, val, _ = network(safe_input)
                # layer_and_channel_patching_vals[layer_index].append(pol[0, atari_escape_index].item())
                layer_and_channel_patching_vals.loc[layer_index, channel_index] = pol[0, atari_escape_index].item()
        finally:
            handle.remove()
            pass

plot_atari_escape_pol(layer_and_channel_patching_vals, labels={'x': 'Layer', 'y': 'Policy', 'color': 'Channel'}, title=f'Channel Patching by Layer (residualified: {residualify})')

#%%
# TRAIN the INPUT to maximize a neuron's activation
class ParseTrainedInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # x = (torch.nn.functional.relu(x) * 2) - 1
        input = torch.zeros((1, 18, 19, 19), device=device)
        input[0, 0] = x.clamp(min=0, max=1)
        input[0, 8] = -x.clamp(max=0, min=-1)
        input[0, 15] = 1
        # Round anything above 0.8 to 1.0 and anything below to 0.0
        threshold = 0.8
        rounded_input = torch.where(input > threshold, torch.ones_like(input), torch.zeros_like(input))
        return input, rounded_input

trained_input = torch.randn((19, 19), requires_grad=True, device=device)
parser = ParseTrainedInput()
optimizer = torch.optim.Adam([trained_input], lr=0.15)
neuron_layer, neuron_channel = -1, 2 # layer -1 for input_cov
# coords = torch.tensor([(2, 15), (3, 15), (4, 15), (5, 15)], dtype=torch.long, device=device)
coords = [(2, 15)]
coords_tensor = torch.tensor(coords, dtype=torch.long, device=device)
loss_history = []
torch.autograd.set_detect_anomaly(True)

network.train()
for i in range(100):
    optimizer.zero_grad()
    input, rounded_input = parser(trained_input)
    # handle = network.residual_tower[neuron_layer].register_forward_hook(display_module_output)
    try:
        pol, val, activations = network(input, neuron_layer)
    finally:
        # handle.remove()
        pass
    # Sum the activations of coordinates in coords
    channel_activations = -activations[0, neuron_channel]
    coords_activations = channel_activations[tuple(coords_tensor)]
    loss = torch.sum(coords_activations)
    loss_history.append(loss.item())
    loss.backward()
    optimizer.step()

# print(f"Layer {neuron_layer} ACTIVATIONS (optimizing neuron {neuron_channel}, {neuron_x}, {neuron_y})")
show_layer_activations(activations)
# print("TRAINED INPUT")
# px.imshow(trained_input.detach().cpu().numpy(), origin="lower").show()
# print("Parsed trained input")
# px.imshow(input.detach().cpu().numpy()[0, 0], origin="lower").show()
# px.imshow(input.detach().cpu().numpy()[0, 8], origin="lower").show()
print("Rounded parsed trained input")
print_go_board_tensor(rounded_input, 0, coords=coords)
px.line(loss_history, height=300).show()

# %%
