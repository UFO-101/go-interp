#%%
import torch
import LeelaZero
from parse_leela_data import Dataset
import pandas as pd
from einops import rearrange
from torchviz import make_dot
import plotly.express as px
import importlib
from utils import print_go_board_tensor, display_tensor_grid, tensor_symmetries, point_symmetries

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
importlib.reload(LeelaZero)
layer_count, channel_count = 40, 256
network = LeelaZero.Network(19, 18, channel_count, layer_count)
network.load_state_dict(torch.load(".cache/best-network.pt"))
network.eval()
network.to(device)
print("LeelaZero.Network loaded.")
#%%
pos_infos = [("leelaoutput", 294, (2, 11)), ("better-long-range-atari", 50, (9, 15)), ("spiral-atari-1", 162, (3, 9))]
inputs = []
points, point_indices = [], []
for filename, turn, coords in pos_infos:
    input = Dataset([f"game-data/{filename}.txt.0.gz"])[turn][0]
    print_go_board_tensor(input, turn, coords=[coords])
    inputs.append(tensor_symmetries(input))
    points.extend(point_symmetries(*coords)[0])
    point_indices.extend(point_symmetries(*coords)[1])
inputs = torch.cat(inputs).to(device)
#%%
# SHOW ACTIVATIONS and POLICY
network.eval()
pol, val, activations, _ = network(inputs, leaky_relu=False)
print('activations', activations.shape)
display_tensor_grid(activations[:, 16], animate=True)

print("POLICY 0")
px.imshow(pol[0][:-1].reshape(19, 19).cpu().detach().numpy(), origin="lower").show()
print(f'Board value: {((1 + val[0].item()) / 2.0) * 100}%')
print("POLICY 1")
px.imshow(pol[1][:-1].reshape(19, 19).cpu().detach().numpy(), origin="lower").show()
print("POLICY 10")
px.imshow(pol[10][:-1].reshape(19, 19).cpu().detach().numpy(), origin="lower").show()

#%%
# GRADIENT PROBE
grad_network = LeelaZero.Network(19, 18, channel_count, layer_count)
grad_network.load_state_dict(torch.load(".cache/best-network.pt"))
grad_network.train()
for module in grad_network.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eval()

optimizer = torch.optim.Adam(grad_network.parameters(), lr=0.00001)
grad_network.zero_grad()
pol, val, activations, block_activations = grad_network(inputs)
point_pols = torch.gather(pol, dim=-1, index=torch.tensor(point_indices).unsqueeze(-1))
loss = -point_pols.mean()
print('loss', loss)
loss.backward()
optimizer.step()
pol, val, mod_activations, mod_block_outputs = grad_network(inputs)
activation_diff = mod_activations - activations
block_output_diff = mod_block_outputs - block_activations
print("activation diff", activation_diff.shape)
display_tensor_grid(activation_diff[:, 0], title="activation 1 grad step diff", animate=True)

# print("negative weight gradients")
# display_tensor_grid(-grad_network.residual_tower[26].conv2.conv.weight.grad)
# print("weights")
# display_tensor_grid(grad_network.residual_tower[39].conv2.conv.weight[83])
#%%
# display weights

weights_to_plot = network.residual_tower[39].conv2.conv.weight
# weights_to_plot = network.conv_input.conv.weight
channel_to_plot = 56
print('weights_to_plot', weights_to_plot.shape)
fig = px.imshow(weights_to_plot[channel_to_plot].detach().cpu().numpy(), origin="lower", facet_col=0, facet_col_wrap=16, facet_col_spacing=0.001, facet_row_spacing=0.01, width=1000, height=1000)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("_")[-1]))
fig.show()

#%%
# activation patching setup
atari_turn = 50
print_go_board_tensor(dataset[atari_turn], atari_turn, show_n_turns_past=0, only_to_play=false)
safe_turn = 54
print_go_board_tensor(dataset[safe_turn], safe_turn, show_n_turns_past=0, only_to_play=false)
atari_input = dataset[atari_turn][0].unsqueeze(0).to(device).requires_grad_(false)
safe_input = dataset[safe_turn][0].unsqueeze(0).to(device).requires_grad_(false)
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
# layer patching
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
        x = torch.nn.functional.tanh(x)
        layer_0 = torch.maximum(x, torch.zeros_like(x)).unsqueeze(0)
        layers_1_7 = torch.zeros_like(x).repeat(7, 1, 1)
        layer_8 = torch.maximum(-x, torch.zeros_like(x)).unsqueeze(0)
        layers_9_15 = torch.zeros_like(x).repeat(7, 1, 1)
        layer_16 = torch.full_like(x, fill_value=torch.max(x).item()).unsqueeze(0)
        # layer_16 = torch.ones_like(x).unsqueeze(0)
        layer_17 = torch.zeros_like(x).unsqueeze(0)
        input = torch.cat((layer_0, layers_1_7, layer_8, layers_9_15, layer_16, layer_17), dim=0).unsqueeze(0)
        threshold = 0.8
        rounded_input = torch.where(input > threshold, torch.ones_like(input), torch.zeros_like(input))
        return input, rounded_input

# class ParseTrainedInput(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         x = (torch.nn.functional.tanh(x) + 1) / 2
#         x = torch.cat([x, torch.ones((1, 1, 19, 19)), torch.zeros((1, 1, 19, 19))], dim=1)
#         return x

trained_input = torch.full((19, 19), 0.0, requires_grad=True, device=device) #, fill_value=0.0)
# trained_input = torch.full((1, 16, 19, 19), fill_value=-2.0, requires_grad=True, device=device)
parser = ParseTrainedInput()
optimizer = torch.optim.Adam([trained_input], lr=0.2)
neuron_layer, neuron_channel = 25, 166 # layer -1 for input_cov
# coords = torch.tensor([(2, 15), (3, 15), (4, 15), (5, 15)], dtype=torch.long, device=device)
# 1.4892 - 1.4893


coords = [(2, 15)]
coords_tensor = torch.tensor(coords, dtype=torch.long, device=device)
loss_history = []
torch.autograd.set_detect_anomaly(True)
network.eval()

print(f"Optimising for layer {neuron_layer}, channel {neuron_channel}, coords {coords}")
for i in range(100):
    optimizer.zero_grad()
    input, rounded_input = parser(trained_input)
    # input = parser(trained_input)
    pol, val, activations = network(input, neuron_layer, leaky_relu=False)
    channel_activations = activations[0, neuron_channel]
    coords_activations = channel_activations[tuple(coords_tensor[:, 1]), tuple(coords_tensor[:, 0])]
    # other_activations = torch.sum(activations) - torch.sum(coords_activations)
    loss = -torch.sum(coords_activations)
    # other_activations = torch.sum(activations) - torch.sum(coords_activations)
    # loss = other_activations * 0.01 - torch.mean(coords_activations)
    loss_history.append(loss.item())
    loss.backward()
    optimizer.step()

display_tensor_grid(activations)
print("TRAINED INPUT")
px.imshow(trained_input.detach().cpu().numpy(), origin="lower").show()
print_go_board_tensor(rounded_input, 0, coords=coords)
# show_layer_activations(parser(trained_input), boards_width=4, boards_height=5)
px.line(loss_history, height=300).show()
# %%
