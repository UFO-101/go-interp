#%%
import torch as t
import plotly.express as px
import pandas as pd
import LeelaZero
from parse_leela_data import Dataset
import importlib
importlib.reload(LeelaZero)
from utils import print_go_board_tensor, display_tensor_grid

device = t.device("cuda" if t.cuda.is_available() else "cpu")
layer_count, channel_count = 40, 256

# coords = (13, 2)
coords = (10, 10)
flat_coord = coords[1] * 19 + coords[0]
folder = "game-data/adv-attack"
# input1 = Dataset([f"{folder}/connection_test_realgame2_b_wturn.txt.0.gz"], fill_blanks=True)[5][0]
input1 = Dataset(["game-data/ladder-with-breaker.txt.0.gz"])[12][0]
input = input1.unsqueeze(0).to(device)

trained_model = LeelaZero.Network(19, 18, channel_count, layer_count).to(device)
trained_model.load_state_dict(t.load(".cache/best-network.pt"))
trained_model.eval()
trained_params = trained_model.state_dict()
trained_pol, trained_val, _, _ = trained_model(input)

print("Input 0") or print_go_board_tensor(input, color='w', coords=[coords])
px.imshow(trained_pol[0, :-1].view(19, 19).detach().cpu(), origin="lower").show()

base_model = LeelaZero.Network(19, 18, channel_count, layer_count).to(device)
base_model.load_state_dict(t.load(".cache/best-network.pt"))
base_model.eval()
for name, parameter in base_model.named_parameters():
    if "conv.weight" in name and "residual_tower" in name:
        parameter.data = t.full_like(parameter.data, t.mean(parameter).item())
base_params = base_model.state_dict()
base_pol, base_val, _, _ = base_model(input)

lerp_model = LeelaZero.Network(19, 18, channel_count, layer_count).to(device)
lerp_model.train()
for module in lerp_model.modules():
    if isinstance(module, t.nn.BatchNorm2d):
        module.eval()

lerp_params = lerp_model.state_dict()
samples = 500
prev_loss = None
prev_grads = None
integrated_grads = t.stack([t.zeros_like(param) for name, param in lerp_params.items() if "conv.weight" in name and "residual_tower" in name])
for i in range(0, samples + 1):
    new_params = {}
    for name, parameter in lerp_model.state_dict().items():
        if "conv.weight" in name and "residual_tower" in name:
            new_params[name] = base_params[name] + i / samples * (trained_params[name] - base_params[name])
        else:
            new_params[name] = trained_params[name]
    lerp_model.load_state_dict(new_params)

    lerp_pol, lerp_val, _, _ = lerp_model(input)
    loss = lerp_pol[0, flat_coord]
    # loss = lerp_val[0]
    loss.backward()

    if prev_loss is not None and prev_grads is not None:
        loss_diff = loss.item() - prev_loss
        integrated_grads += (prev_grads ** 2 * loss_diff) / t.sum(prev_grads ** 2)
    
    prev_loss = loss.item()
    prev_grads = t.stack([param.grad for name, param in lerp_model.named_parameters() if "conv.weight" in name and "residual_tower" in name])

# print("Base diff:", trained_val.item() - base_val.item())
print("Base diff:", trained_pol[0, flat_coord].item() - base_pol[0, flat_coord].item())
print("Integrated total:", t.sum(integrated_grads))
#%%

#%%
print('integrated_grads.shape', integrated_grads.shape)
px.imshow(integrated_grads.sum(dim=(-1, -2, -3)).reshape(80, 16, 16).detach().cpu(), origin="lower", animation_frame=0).show()
px.imshow(integrated_grads.sum(dim=(-1, -2, -4)).reshape(80, 16, 16).detach().cpu(), origin="lower", animation_frame=0).show()
px.imshow(integrated_grads.sum(dim=(0, -1, -2, -3)).reshape(16, 16).detach().cpu(), origin="lower").show()

#%%
display_tensor_grid(integrated_grads[:, 157], animate=True, bord_const=1e-9)