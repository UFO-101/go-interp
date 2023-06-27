#%%
import torch as t
import plotly.express as px
import LeelaZero
from parse_leela_data import Dataset
import importlib
importlib.reload(LeelaZero)
from utils import print_go_board_tensor, display_tensor_grid, shifted_tensors, tensor_symmetries
from torch.nn import functional as F
import plotly.graph_objects as go
import itertools

device = t.device("cuda" if t.cuda.is_available() else "cpu")
layer_count, channel_count = 40, 256

coords = (10, 10)
flat_coord = coords[1] * 19 + coords[0]
input0 = Dataset(["game-data/ladder-no-breaker.txt.0.gz"])[12][0]
input1 = Dataset(["game-data/ladder-with-breaker.txt.0.gz"])[12][0]
input0 = input0.unsqueeze(0).to(device)
input1 = input1.unsqueeze(0).to(device)
inputs = t.cat([input0, input1]).to(device)

trained_model = LeelaZero.Network(19, 18, channel_count, layer_count).to(device)
trained_model.load_state_dict(t.load(".cache/best-network.pt"))
trained_model.eval()
trained_params = trained_model.state_dict()
trained_pol, _, trained_resids, _, trained_resids_and_mids = trained_model(inputs)
trained_pol_prob = F.softmax(trained_pol, dim=1)

print("Input 0") or print_go_board_tensor(input0, color=12, coords=[coords])
print("Input 1") or print_go_board_tensor(input1, color=12, coords=[coords])
print("Trained")
px.imshow(trained_pol_prob[:, :-1].view(2, 19, 19).detach().cpu(), origin="lower", facet_col=0).show()

base_model = LeelaZero.Network(19, 18, channel_count, layer_count).to(device)
base_model.load_state_dict(t.load(".cache/best-network.pt"))
base_model.eval()
for name, parameter in base_model.named_parameters():
    if "conv.weight" in name and "residual_tower" in name:
        parameter.data = t.full_like(parameter.data, t.mean(parameter).item())
base_params = base_model.state_dict()
base_pol, _, _, _, _ = base_model(inputs)
base_pol_prob = F.softmax(base_pol, dim=1)

print("Base Input 0")
px.imshow(base_pol_prob[:, :-1].view(2, 19, 19).detach().cpu(), origin="lower", facet_col=0).show()

lerp_model = LeelaZero.Network(19, 18, channel_count, layer_count).to(device)
lerp_model.train()
for module in lerp_model.modules():
    if isinstance(module, t.nn.BatchNorm2d):
        module.eval()
lerp_params = lerp_model.state_dict()

#%%
param_samples = 200
integrated_grad_list = []
for i in range(2):
    integrated_grads = t.stack([t.zeros_like(param) for name, param in lerp_params.items() if "conv.weight" in name and "residual_tower" in name])
    prev_loss = None
    prev_grads = None
    input = inputs[i:i+1]
    for j in range(0, param_samples + 1):
        lerp_model.zero_grad()
        new_params = {}
        for name, _ in lerp_model.state_dict().items():
            if "conv.weight" in name and "residual_tower" in name:
                new_params[name] = base_params[name] + j / param_samples * (trained_params[name] - base_params[name])
            else:
                new_params[name] = trained_params[name]
        lerp_model.load_state_dict(new_params)

        lerp_pol, lerp_val, _, _, _ = lerp_model(input)
        lerp_pol = F.softmax(lerp_pol, dim=1)
        loss = lerp_pol[0, flat_coord]
        loss.backward()

        if prev_loss is not None and prev_grads is not None:
            loss_diff = loss.item() - prev_loss
            clamp_prev_grads_sqr = t.clamp_min(t.sum(prev_grads ** 2), 1e-3)
            integrated_grads += (prev_grads ** 2 * loss_diff) / clamp_prev_grads_sqr
        
        prev_loss = loss.item()
        prev_grads = t.stack([param.grad for name, param in lerp_model.named_parameters() if "conv.weight" in name and "residual_tower" in name])
    integrated_grad_list.append(integrated_grads)

#%%
print("Baselines:")
train0, train1 = trained_pol_prob[0, flat_coord].item(), trained_pol_prob[1, flat_coord].item()
base0, base1 = base_pol_prob[0, flat_coord].item(), base_pol_prob[1, flat_coord].item()
print("Trained 1 - base 1:", train1 - base1)
print("Trained 0 - base 0:", train0 - base0)
print() or print("Integrated gradients[0]:", integrated_grad_list[0].sum())
print() or print("Integrated gradients[1]:", integrated_grad_list[1].sum())
integrated_grad_sum = (integrated_grad_list[0] + integrated_grad_list[1]).sum(dim=(-1, -2))
#%%
display_tensor_grid(1000 * (integrated_grad_list[0].sum(dim=(-1, -2)) + integrated_grad_list[1].sum(dim=(-1, -2))).view(80, 256, 16, 16), animate=True)

#%%
layer_strs = [f"{i}" for i in range(0, layer_count * 2)]
channel_strs = [f"{i}" for i in range(0, channel_count)]
node_labels = list(itertools.product(layer_strs, channel_strs))
node_xs = t.linspace(1e-6, 1-(1e-6), layer_count * 2).unsqueeze(-1).repeat(1, channel_count).flatten()
node_ys = t.linspace(1e-6, 1-(1e-6), channel_count).repeat(layer_count * 2)
src_nodes = t.arange(0, ((layer_count * 2) -1) * channel_count).view(-1, channel_count).repeat(1, channel_count)
print("src_nodes", src_nodes.shape)
src_nodes = src_nodes.flatten()
trgt_nodes = t.arange(channel_count, (layer_count*2) * channel_count).unsqueeze(-1).repeat(1, channel_count)
print("trgt_nodes", trgt_nodes.shape)
trgt_nodes = trgt_nodes.flatten()
filtered_topk_integrated_grad_sum = t.where(integrated_grad_sum[:-1] >= integrated_grad_sum[:-1].flatten().topk(20).values[-1], integrated_grad_sum[:-1], 0).flatten()
# filtered_topk_integrated_grad_sum = integrated_grad_sum[:-1].flatten().tolist()
# Remove unused elements to spot plotly breaking
indices = t.nonzero(filtered_topk_integrated_grad_sum).flatten()
src_nodes = src_nodes[indices].tolist()
trgt_nodes = trgt_nodes[indices].tolist()
filtered_topk_integrated_grad_sum = filtered_topk_integrated_grad_sum[indices]
normalized_topk = filtered_topk_integrated_grad_sum / filtered_topk_integrated_grad_sum.max().item()
filtered_topk_integrated_grad_sum = filtered_topk_integrated_grad_sum.tolist()
normalized_topk = normalized_topk.tolist()
used_nodes = list(set(src_nodes + trgt_nodes))
used_nodes.sort()
node_xs = node_xs[used_nodes].tolist()
node_ys = node_ys[used_nodes].tolist()

# filtered_topk_integrated_grad_sum = t.zeros_like(integrated_grad_sum[:-1]).flatten()
# filtered_topk_integrated_grad_sum[0:30] = 1
# filtered_topk_integrated_grad_sum = filtered_topk_integrated_grad_sum.tolist()
print("src_nodes", len(src_nodes), "trgt_nodes", len(trgt_nodes), "nodes_vals", len(filtered_topk_integrated_grad_sum))
fig = go.Figure(data=[go.Sankey(
    node = dict(
    #   pad = 15,
        thickness = 5,
    #   line = dict(color = "black", width = 0.5),
        label = [':'.join(tup) for tup in node_labels],
        x = node_xs,
        y = node_ys,
    ),
    link = dict(
        source = src_nodes,
        target = trgt_nodes,
        value = filtered_topk_integrated_grad_sum,
        color = [f"rgba(0, 0, 0, {a})" for a in normalized_topk]
  ))],
    layout=go.Layout(
        width=18000,
        height=1500,
                ))

fig.show(renderer="png")

#%%
def top_indices(x, k=10):
    topk = x.sum(dim=(-1, -2)).flatten().topk(k)
    top_indices = topk.indices
    top_indices = t.stack([top_indices // 256 // 256, top_indices // 256 % 256, top_indices % 256], dim=1)
    return top_indices
grad_list_add_top_idx = top_indices(integrated_grad_list[0] + integrated_grad_list[1])

multiplier = -1
ys = []
print("grad_list_add_top_idx")
print(grad_list_add_top_idx)
for top_idx in grad_list_add_top_idx:
    lerp_model.load_state_dict(t.load(".cache/best-network.pt"))
    if top_idx[0] % 2 == 0:
        lerp_model.residual_tower[top_idx[0] // 2].conv1.conv.weight.data[top_idx[1], top_idx[2]] *= multiplier
    else:
        lerp_model.residual_tower[top_idx[0] // 2].conv2.conv.weight.data[top_idx[1], top_idx[2]] *= multiplier
    mod_pol, _, _, _, _ = lerp_model(inputs)
    mod_pol_softmax = t.softmax(mod_pol, dim=1)
    ys.append(mod_pol_softmax[1, flat_coord].item())
fig = px.bar(x=list(range(len(ys))), y=ys)
fig.add_hline(y=trained_pol_prob[0, flat_coord].item(), line_dash="dash", line_color="red")
fig.add_hline(y=trained_pol_prob[1, flat_coord].item(), line_dash="dash", line_color="red")
fig.show()

#%%
integrated_grad_sum[0, 90, 166].shape

#%%
shifted_inputs = tensor_symmetries(inputs)
lerp_model.load_state_dict(t.load(".cache/best-network.pt"))
lerp_model.residual_tower[0].conv1.conv.weight.data[90, 166] *= -1
# lerp_model.residual_tower[1].conv2.conv.weight.data[4, 191] *= -1
# lerp_model.residual_tower[3].conv2.conv.weight.data[168, 227] *= -1
mod_pol, _, mod_resids, _, mod_resids_and_mids = lerp_model(shifted_inputs)
mod_pol_softmax = t.softmax(mod_pol, dim=1)
px.imshow(mod_pol_softmax[:, :-1].view(-1, 2, 19, 19).detach().cpu(), origin="lower", animation_frame=0, facet_col=1).show()

#%%
# resid_diff = trained_resids - mod_resids
# display_tensor_grid(resid_diff[:, 0], animate=True)
# display_tensor_grid(resid_diff[:, 1], animate=True)

resids_and_mids_diff = trained_resids_and_mids - mod_resids_and_mids
display_tensor_grid(resids_and_mids_diff[:, 0], animate=True)
display_tensor_grid(resids_and_mids_diff[:, 1], animate=True)
