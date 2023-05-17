#%%
import torch as t
import plotly.express as px
import LeelaZero
import importlib
importlib.reload(LeelaZero)
from utils import print_go_board_tensor, display_tensor_grid, tensor_symmetries

device = t.device("cuda" if t.cuda.is_available() else "cpu")
layer_count, channel_count = 40, 256
model = LeelaZero.Network(19, 18, channel_count, layer_count).to(device)
model.load_state_dict(t.load(".cache/best-network.pt"))
model.train()
for module in model.modules():
    if isinstance(module, t.nn.BatchNorm2d):
        module.eval()

# TRAIN the INPUT to maximize a neuron's activation
class ParseTrainedInput(t.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = t.nn.functional.tanh(x)
        layer_0 = t.maximum(x, t.zeros_like(x)).unsqueeze(0)
        layers_1_7 = t.zeros_like(x).repeat(7, 1, 1)
        layer_8 = t.maximum(-x, t.zeros_like(x)).unsqueeze(0)
        layers_9_15 = t.zeros_like(x).repeat(7, 1, 1)
        # layer_16 = t.full_like(x, fill_value=t.max(x).item()).unsqueeze(0)
        layer_16 = t.ones_like(x).unsqueeze(0)
        layer_17 = t.zeros_like(x).unsqueeze(0)
        input = t.cat((layer_0, layers_1_7, layer_8, layers_9_15, layer_16, layer_17), dim=0).unsqueeze(0)
        threshold = 0.8
        rounded_input = t.where(input > threshold, t.ones_like(input), t.zeros_like(input))
        # input = tensor_symmetries(input) # 8 symmetries
        return input, rounded_input

trained_input = t.randn((19, 19), requires_grad=True, device=device) #, fill_value=0.0)
parser = ParseTrainedInput()
optimizer = t.optim.Adam([trained_input], lr=0.2)
neuron_layer, neuron_channel = 23, 234 # layer -1 for input_cov
# coords = t.tensor([(2, 15), (3, 15), (4, 15), (5, 15)], dtype=t.long, device=device)
coords = [(3, 9)]
# coords = point_symmetries(2, 15)[0]
# print("coords", coords)
coords_tensor = t.tensor(coords, dtype=t.long, device=device)
loss_history = []
other_activations_history = []

print(f"Optimising for layer {neuron_layer}, channel {neuron_channel}, coords {None}")
for i in range(1000):
    optimizer.zero_grad()
    input, rounded_input = parser(trained_input)
    _, _, resids, _ = model(input)
    channel_activations = resids[neuron_layer + 1, :, neuron_channel]
    # channel_activations = t.flatten(channel_activations, start_dim=-2)

    # top_activations = t.topk(channel_activations, k=5)
    coords_activations = channel_activations[tuple(range(len(coords))), tuple(coords_tensor[:, 1]), tuple(coords_tensor[:, 0])]
    # print("coords activations", coords_activations.shape)
    # other_activations = t.sum(resids[neuron_layer + 1]) - t.sum(coords_activations)
    # loss = -t.sum(channel_activations)
    loss = -t.sum(coords_activations)
    # loss = -t.sum(top_activations.values)
    # other_activations = t.sum(resids[neuron_layer + 1]) - t.sum(channel_activations)
    # loss = other_activations * 0.0005 - t.mean(channel_activations)
    loss_history.append(loss.item())
    # other_activations_history.append(-other_activations.item())
    loss.backward()
    optimizer.step()

display_tensor_grid(resids[neuron_layer + 1, 0])
print("TRAINED INPUT")
px.imshow(trained_input.detach().cpu().numpy(), origin="lower").show()
print_go_board_tensor(rounded_input, 'b', coords=coords)
# show_layer_activations(parser(trained_input), boards_width=4, boards_height=5)
px.line(loss_history, height=300).show()
# px.line(other_activations_history, height=300).show()
# %%
import importlib
import utils
importlib.reload(utils)
from utils import print_go_board_tensor, tensor_symmetries, inverse_tensor_symmetries
# print_go_board_tensor(rounded_input, 'b', coords=top_activations.indices.squeeze().tolist())

x = t.arange(16).reshape(4, 4)
print('x', x)
symmetries = tensor_symmetries(x)
print('symmetries', symmetries)
print()
print('inverse', inverse_tensor_symmetries(symmetries))

