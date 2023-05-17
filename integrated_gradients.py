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
model = LeelaZero.Network(19, 18, channel_count, layer_count).to(device)
model.load_state_dict(t.load(".cache/best-network.pt"))
model.eval()

COUNTERFACTUAL = True

turn0, turn1 = 12, 12
# turn0, turn1 = 160, 162
# turn0, turn1 = 5, 5
# turn0, turn1 = 48, 50
# coords = (3, 9)
# coords = (9, 15)
coords = (10, 10)
# coords = (13, 2)
flat_coord = coords[1] * 19 + coords[0]
folder = "game-data/adv-attack"
# input0 = Dataset(["game-data/better-long-range-atari.txt.0.gz"])[turn0][0]
input0 = Dataset(["game-data/ladder-with-breaker.txt.0.gz"])[turn0][0]
# input0 = Dataset([f"{folder}/connection_test_realgame2_a_wturn.txt.0.gz"], fill_blanks=True)[turn0][0]
# input1 = Dataset(["game-data/better-long-range-atari.txt.0.gz"])[turn1][0]
input1 = Dataset(["game-data/ladder-no-breaker.txt.0.gz"])[turn1][0]
# input1 = Dataset([f"{folder}/connection_test_realgame2_b_wturn.txt.0.gz"], fill_blanks=True)[turn1][0]
inputs = t.stack([input0, input1]).to(device)
pols, vals, resids, block_outs = model(inputs)
base_diff =  pols[1, flat_coord] - pols[0, flat_coord]

print("Input 0") or print_go_board_tensor(input0, color='w', coords=[coords])
# px.imshow(pols[0, :-1].view(19, 19).detach().cpu(), origin="lower").show()
print("Input 1") or print_go_board_tensor(input1, color='w', coords=[coords])
# px.imshow(pols[1, :-1].view(19, 19).detach().cpu(), origin="lower").show()
print("Input 0 value", vals[0].item(), "Input 1 value", vals[1].item())

#%%

integrated_grads = t.zeros_like(resids[:, 0:1])
samples = 50
baseline_diffs = []
for lyr in range(0, layer_count-1):
    prev_loss = None
    prev_grad = None
    resid_1 = resids[lyr+1, 1:2]
    resid_0 = resids[lyr+1, 0:1] if COUNTERFACTUAL else t.zeros_like(resid_1)
    _, val_1, _, _ = model(resid_1, input_block=lyr+1)
    _, val_0, _, _ = model(resid_0, input_block=lyr+1)
    baseline_diffs.append(val_1.item() - val_0.item())
    for i in range(0, samples + 1):
        model.zero_grad()
        lerp_input = (resid_0 + (resid_1 - resid_0) * i / samples).detach()
        lerp_input.requires_grad_(True)

        pol, val, _, _ = model(lerp_input, input_block=lyr+1)
        # loss = pol[0, flat_coord]
        loss = val[0]
        loss.backward()

        if prev_loss is not None and prev_grad is not None:
            loss_diff = loss - prev_loss
            integrated_grads[lyr] += (prev_grad**2 *loss_diff)/t.sum(prev_grad**2)

        prev_loss = loss
        prev_grad = lerp_input.grad

print("COUNTERFACTUAL:", COUNTERFACTUAL)
print("Base diffs:", baseline_diffs)
print("Integrated grad sum:", integrated_grads.sum(dim=(1, 2, 3, 4)))
#%%
# print("Resid0") or display_tensor_grid(resids[:, 0], animate=True)
# print("Resid1") or display_tensor_grid(resids[:, 1], animate=True)
# print("Resid1 - Resid0") or display_tensor_grid(resids[:, 0] - resids[:, 1], animate=True)
print("Integrated grads") or display_tensor_grid(integrated_grads, bord_const=0.01, animate=True)
#%%
print("Resids 1") or display_tensor_grid(resids[:, 1], bord_const=0.01, animate=True)
#%%
from torch.nn.functional import softmax
points = [("234: (9, 3)", t.tensor([234, 9, 3])),
          ("234: (15, 9)", t.tensor([234, 15, 9])),
          ("Channel 107", 107),
          ("Channel 28", 28),
          ("Channel 166", 166),
          ("Channel 192", 192),
          ("Channel 148", 148),
          ("Channel 126", 126),
          ("Channel 195", 195),
          ("Channel 187", 187),
          ("Channel 230", 230),
          ("Channel 157", 157),
          ("Channel 191", 191),
          ("Channel 205", 205),
          ("Channel 175", 175),
          ("Channel 4", 4),
          ("100: (9, 3)", t.tensor([100, 9, 3])),
          ("195: (9, 3)", t.tensor([195, 9, 3])),
        #   ("33, 3, 3", t.tensor([33, 3, 3])),
        #   ("234: (10, 3)", t.tensor([234, 10, 3])),
        #   ("234: (9, 4)", t.tensor([234, 9, 4])),
        #   ("230: (9, 3)", t.tensor([230, 9, 3])),
        #   ("166: (9, 3)", t.tensor([166, 9, 3])),
        #   ("166: (15, 9)", t.tensor([166, 15, 9])),
        #   ("34, (4, 4", t.tensor([34, 4, 4])),
        #   ("28, (11, 7)", t.tensor([0, 11, 7])),
        #   ("28, (7, 11)", t.tensor([0, 7, 11])),
        #   ("166, (11, 7)", t.tensor([166, 11, 7])),
        #   ("166, (7, 11)", t.tensor([166, 7, 11])),
        #   ("28, (11, 9)", t.tensor([0, 11, 9])),
        #   ("28, (9, 11)", t.tensor([0, 9, 11])),
        #   ("166, (11, 9)", t.tensor([166, 11, 9])),
        #   ("166, (9, 11)", t.tensor([166, 9, 11])),
          ("Channel 233", 233),
        #   ("Channel 234", 234),
        #   ("Channel 199", 199),
        #   ("Channel 168", 168),
        #   ("Channel 168", 107),
        #   ("Channel 166", 166),
        #   ("Channel 118", 118),
        #   ("Control", None),
]
lines = pd.DataFrame(columns=[name for name, _ in points]) 
for name, max_idx in points:
    ablated_pols = []
    for lyr in range(0, layer_count-1):
        # max_idx = t.argmax(integrated_grads)
        # max_idx = t.tensor([max_idx//(19*19*256), max_idx//(19*19)%256, 1+max_idx%256//19, max_idx%19])
        # max_idx = t.tensor([20, max_idx//(19*19)%256, 1+max_idx%256//19, max_idx%19])

        ablated_resid1 = resids[lyr + 1, 1:].clone()
        resid_0 = resids[lyr + 1, 0]
        if max_idx is not None:
            if isinstance(max_idx, int):
                ablated_resid1[0, max_idx] = resid_0[max_idx]
                # ablated_resid1[0, max_idx] = t.zeros_like(resid_0[max_idx])
            else:
                ablated_resid1[0, max_idx[0], max_idx[1], max_idx[2]] = resid_0[max_idx[0], max_idx[1], max_idx[2]]
                # ablated_resid1[0, max_idx[0], max_idx[1], max_idx[2]] = t.zeros_like(resid_0[max_idx[0], max_idx[1], max_idx[2]])
        ablated_pol, ablated_val, _, _ = model(ablated_resid1, input_block=lyr+1)
        # ablated_pol = softmax(ablated_pol, dim=1)
        # ablated_pols.append(ablated_pol[0, flat_coord].item())
        ablated_pols.append(ablated_val[0].item())
        # print("Resid 0", pols[0, flat_coord].item(), "Ablated pol:", ablated_pol[0, flat_coord].item(), "Resid 1", pols[1, flat_coord].item())
    lines[name] = ablated_pols

fig = px.line(lines, x=lines.index, y=lines.columns)
soft_pols = softmax(pols, dim=1)
# fig.add_hline(y=soft_pols[0, flat_coord].item(), line_dash="dot", line_color="red")
# fig.add_hline(y=soft_pols[1, flat_coord].item(), line_dash="dot", line_color="red")
fig.add_hline(y=vals[0].item(), line_dash="dot", line_color="red")
fig.add_hline(y=vals[1].item(), line_dash="dot", line_color="red")
fig.show()
#%%
# Trying and failing to reproduce Redwood's results
hooks = []
for patch_layer in range(5):
    def patch_hook(module, input, output):
        block_out, mid = output
        block_out[:1, 118, :, :] = block_outs[patch_layer, 1:2, 168, :, :]
        return block_out, mid
    handle = model.residual_tower[patch_layer].register_forward_hook(patch_hook)
    hooks.append(handle)

print("pol[0]", pols[0, flat_coord].item(), "pol[1]", pols[1, flat_coord].item())
try:
    patched_pol, _, _, _ = model(inputs)
    print("patched_pol", patched_pol[0, flat_coord].item())
finally:
    for handle in hooks:
        handle.remove()

unpatched_pol, _, _, _ = model(inputs)
print("unpatched_pol", unpatched_pol[0, flat_coord].item())