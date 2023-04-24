#%%
import os
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
inputs = []
# pos_infos = [("leelaoutput", 294, (2, 11)), ("better-long-range-atari", 50, (9, 15)), ("spiral-atari-1", 162, (3, 9))]
# points, point_indices = [], []
# for filename, turn, coords in pos_infos:
#     input = Dataset([f"game-data/{filename}.txt.0.gz"])[turn][0]
#     print_go_board_tensor(input, turn, coords=[coords])
#     inputs.append(tensor_symmetries(input))
#     points.extend(point_symmetries(*coords)[0])
#     point_indices.extend(point_symmetries(*coords)[1])
# inputs = torch.cat(inputs).to(device)

directory = "game-data"
# filenames = os.listdir(directory)
filenames = ["adv-attack/connection_test_realgame2_a_wturn.txt.0.gz", "adv-attack/connection_test_realgame2_b_wturn.txt.0.gz"]
for filename in filenames: # + ["spiral-atari-1.txt.0.gz"]:
    print("filename", filename)
    input = Dataset([f"{directory}/{filename}"], fill_blanks=True)
    print("len input", len(input))
    input = input[len(input) - 1][0]
    print("input", input.shape)
    color_to_play = "w" if "wturn" in filename else "b"
    print_go_board_tensor(input, color_to_play)
    display_tensor_grid(input)
    inputs.append(input)
inputs = torch.stack(inputs).requires_grad_(False).to(device)
#%%
# SHOW ACTIVATIONS and POLICY
network.eval()
pol, val, activations, _ = network(inputs, leaky_relu=False)
print('activations', activations.shape)
for i, filename in enumerate(filenames):
    name_prefix = filename.split(".")[0]
    # display_tensor_grid(activations[:, i], animate=True, title=name_prefix,
                        # filename=f"output_figs/{name_prefix}.html")

activations_diff = activations[:, 0] - activations[:, 1]
display_tensor_grid(activations_diff, animate=True, title="activations diff",
                    filename=f"output_figs/realgame2_activations_diff.html")

for i in range(2):
    print("POLICY", i)
    px.imshow(pol[i][:-1].view(19, 19).cpu().detach().numpy(), origin="lower").show()
    print(f'Board value: {((1 + val[i].item()) / 2.0) * 100}%')
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
corrupted_input = inputs[0].unsqueeze(0).to(device).requires_grad_(False)
print_go_board_tensor(corrupted_input, color="w")
clean_input = inputs[1].unsqueeze(0).to(device).requires_grad_(False)
print_go_board_tensor(clean_input, color="w")
clean_pol_no_patch, clean_val_no_patch, _, _ = network(clean_input)
corrupted_pol_no_patch, corrupted_val_no_patch, _, _ = network(corrupted_input)
# atari_escape_coords = (9, 15)
# atari_escape_index = atari_escape_coords[1] * 19 + atari_escape_coords[0]

def plot_patch_results(*args, bar=False, **kwargs):
    if bar:
        fig = px.bar(*args, **kwargs)
    else:
        fig = px.line(*args, **kwargs)
    fig.add_hline(y=(corrupted_val_no_patch.item() + 1) / 2 * 100, line_dash="dash", annotation_text="Cycle, no patch")
    fig.add_hline(y=(clean_val_no_patch.item() + 1) / 2 * 100, line_dash="dash", annotation_text="Broken cycle, no patch")
    fig.show()

#%%
# layer patching
network.eval()
patched_vals = []
clean_policy, clean_val, clean_resid, clean_block_outs = network(clean_input)
print("clean_block_outs", clean_block_outs.shape)
for lyr_idx in range(layer_count):
    patch_hook = lambda m, i, o: (clean_block_outs[lyr_idx], None)
    handle = network.residual_tower[lyr_idx].register_forward_hook(patch_hook)
    try:
        with torch.inference_mode():
            pol, val, _, _ = network(corrupted_input)
            patched_vals.append((val.item() + 1) / 2 * 100)
    finally:
        handle.remove()
        pass
plot_patch_results(y=patched_vals, labels={'x': 'Layer', 'y': 'Board value'}, title="Patching broken cycle into cycle, by layer (residualified)")
#%%
# CHANNEL PATCHING
network.eval()
def patch_activations(module, input, output, clean_block_outs, layer, channel):
    # Replace output with atari_activations at the given channel.
    output[0][:, channel] = clean_block_outs[layer, :, channel]
    return output

# Create a pandas dataframe. Each row is a layer, each column is a channel.
_, _, _, clean_block_outs = network(clean_input)
# layer_chan_patch_vals = pd.DataFrame(columns=range(channel_count))
patched_vals = []
for chan_idx in range(channel_count):
    handles = []
    for lyr_idx in range(layer_count):
        patch_hook = lambda m, i, o: patch_activations(m, i, o, clean_block_outs, lyr_idx, chan_idx)
        handle = network.residual_tower[lyr_idx].register_forward_hook(patch_hook)
        handles.append(handle)
    try:
        with torch.inference_mode():
            pol, val, _, _ = network(corrupted_input)
            # layer_chan_patch_vals.loc[chan_idx] = (val.item() + 1) / 2 * 100
            patched_vals.append((val.item() + 1) / 2 * 100)
    finally:
        for handle in handles:
            handle.remove()
        pass

plot_patch_results(bar=True, y=patched_vals, labels={'x': 'Channel', 'y': 'Board value'}, title=f'Patching broken cycle into cycle, by channel (at every layer) (residualified)')

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
        # input = tensor_symmetries(input) # 8 symmetries
        return input, rounded_input

trained_input = torch.randn((19, 19), requires_grad=True, device=device) #, fill_value=0.0)
parser = ParseTrainedInput()
optimizer = torch.optim.Adam([trained_input], lr=0.2)
neuron_layer, neuron_channel = 10, 1 # layer -1 for input_cov
# coords = torch.tensor([(2, 15), (3, 15), (4, 15), (5, 15)], dtype=torch.long, device=device)
# coords = [(2, 15)]
# coords = point_symmetries(2, 15)[0]
# print("coords", coords)
# coords_tensor = torch.tensor(coords, dtype=torch.long, device=device)
loss_history = []
other_activations_history = []
network.train()

print(f"Optimising for layer {neuron_layer}, channel {neuron_channel}, coords {None}")
for i in range(5000):
    optimizer.zero_grad()
    input, rounded_input = parser(trained_input)
    _, _, resids, _ = network(input)
    channel_activations = resids[neuron_layer + 1, :, neuron_channel]
    # coords_activations = channel_activations[tuple(range(len(coords))), tuple(coords_tensor[:, 1]), tuple(coords_tensor[:, 0])]
    # print("coords activations", coords_activations.shape)
    # other_activations = torch.sum(resids[neuron_layer + 1]) - torch.sum(coords_activations)
    loss = -torch.sum(channel_activations)
    # other_activations = torch.sum(resids[neuron_layer + 1]) - torch.sum(channel_activations)
    # loss = other_activations * 0.0005 - torch.mean(channel_activations)
    loss_history.append(loss.item())
    # other_activations_history.append(-other_activations.item())
    loss.backward()
    optimizer.step()

display_tensor_grid(resids[neuron_layer + 1, 0])
print("TRAINED INPUT")
px.imshow(trained_input.detach().cpu().numpy(), origin="lower").show()
print_go_board_tensor(rounded_input, 'b', coords=None)
# show_layer_activations(parser(trained_input), boards_width=4, boards_height=5)
px.line(loss_history, height=300).show()
px.line(other_activations_history, height=300).show()
# %%
