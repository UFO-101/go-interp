#%%
import torch as t
import plotly.express as px
import LeelaZero
from parse_leela_data import Dataset
import importlib
importlib.reload(LeelaZero)
from utils import print_go_board_tensor, display_tensor_grid, shifted_tensors, tensor_symmetries
from torch.nn import functional as F

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
trained_pol, _, trained_resids, _, trained_midlayers = trained_model(inputs)
trained_pol_prob = F.softmax(trained_pol, dim=1)

loss = trained_pol_prob[0, flat_coord]
loss.backward()
prev_grads0 = t.stack([param.grad for name, param in trained_model.named_parameters() if "conv.weight" in name and "residual_tower" in name])
top = print_top_kernels(prev_grads0, k=50000)
top_prod = (top+1).prod(dim=1)
print("Top kernels 0")
print ((top_prod==(0+1)*(90+1)*(166+1)).nonzero(as_tuple=True)[0])
print ((top_prod==(3+1)*(4+1)*(191+1)).nonzero(as_tuple=True)[0])

trained_model.zero_grad()
trained_pol, _, trained_resids, _, trained_midlayers = trained_model(inputs)
trained_pol_prob = F.softmax(trained_pol, dim=1)
loss = trained_pol_prob[1, flat_coord]
loss.backward()
prev_grads1 = t.stack([param.grad for name, param in trained_model.named_parameters() if "conv.weight" in name and "residual_tower" in name])
top = print_top_kernels(prev_grads1, k=50000)
top_prod = (top+1).prod(dim=1)
print() or print("Top kernels 1")
print ((top_prod==(0+1)*(90+1)*(166+1)).nonzero(as_tuple=True)[0])
print ((top_prod==(3+1)*(4+1)*(191+1)).nonzero(as_tuple=True)[0])

grad_diff1 = prev_grads0 - prev_grads1
print() or print("Top kernels sub 1")
top = print_top_kernels(grad_diff1, k=50000)
top_prod = (top+1).prod(dim=1)
print ((top_prod==(0+1)*(90+1)*(166+1)).nonzero(as_tuple=True)[0])
print ((top_prod==(3+1)*(4+1)*(191+1)).nonzero(as_tuple=True)[0])

grad_diff2 = prev_grads1 - prev_grads0
top = print_top_kernels(grad_diff2, k=50000)
top_prod = (top+1).prod(dim=1)
print() or print("Top kernels sub 2")
print ((top_prod==(0+1)*(90+1)*(166+1)).nonzero(as_tuple=True)[0])
print ((top_prod==(3+1)*(4+1)*(191+1)).nonzero(as_tuple=True)[0])

grad_add = prev_grads0 + prev_grads1
top_indices_simple_grad_add = print_top_kernels(grad_add, k=50000)
top_prod = (top_indices_simple_grad_add+1).prod(dim=1)
print() or print("Top kernels add")
print ((top_prod==(0+1)*(90+1)*(166+1)).nonzero(as_tuple=True)[0])
print ((top_prod==(3+1)*(4+1)*(191+1)).nonzero(as_tuple=True)[0])

#%%


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

# input_samples = 30
param_samples = 400
integrated_grad_list = []
# for i in range(0, input_samples + 1):
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
            integrated_grads += (prev_grads ** 2 * loss_diff) / t.sum(prev_grads ** 2)
        
        prev_loss = loss.item()
        prev_grads = t.stack([param.grad for name, param in lerp_model.named_parameters() if "conv.weight" in name and "residual_tower" in name])
    integrated_grad_list.append(integrated_grads)

#%%
print("Baselines:")
train0, train1 = trained_pol_prob[0, flat_coord].item(), trained_pol_prob[1, flat_coord].item()
base0, base1 = base_pol_prob[0, flat_coord].item(), base_pol_prob[1, flat_coord].item()
print("Trained 1 - trained 0:", train1 - train0)
print("Base 1 - base 0:", base1 - base0)
print("Trained 1 - base 1:", train1 - base1)
print("Trained 0 - base 0:", train0 - base0)
print() or print("Integrated gradients[0]:", integrated_grad_list[0].sum())
print() or print("Integrated gradients[1]:", integrated_grad_list[1].sum())
#%%
display_tensor_grid(1000 * (integrated_grad_list[0].sum(dim=(-1, -2)) + integrated_grad_list[1].sum(dim=(-1, -2))).view(80, 256, 16, 16), animate=True)

#%%
norm_integrated_grad0 = integrated_grad_list[0] / integrated_grad_list[0].sum()
norm_integrated_grad1 = integrated_grad_list[1] / integrated_grad_list[1].sum()
# print("norm_integrated_grad0:", norm_integrated_grad0.sum(), "norm_integrated_grad1:", norm_integrated_grad1.sum())
# display_tensor_grid(100 * norm_integrated_grad0.sum(dim=(-1, -2)).view(80, 256, 16, 16), animate=True)
# display_tensor_grid(-100 * norm_integrated_grad1.sum(dim=(-1, -2)).view(80, 256, 16, 16), animate=True)
# integrated_grad_diff = norm_integrated_grad0 - norm_integrated_grad1

soft_integrated_grad0 = F.softmax(integrated_grad_list[0].flatten(), dim=0).view(integrated_grad_list[0].shape)
soft_integrated_grad1 = F.softmax(integrated_grad_list[1].flatten(), dim=0).view(integrated_grad_list[1].shape)
print("soft_integrated_grad0:", soft_integrated_grad0.sum(), "soft_integrated_grad1:", soft_integrated_grad1.sum())
# display_tensor_grid(100 * soft_integrated_grad0.sum(dim=(-1, -2)).view(80, 256, 16, 16), animate=True)
# display_tensor_grid(-100 * soft_integrated_grad1.sum(dim=(-1, -2)).view(80, 256, 16, 16), animate=True)
# display_tensor_grid(100 * (soft_integrated_grad0 - soft_integrated_grad1).sum(dim=(-1, -2)).view(80, 256, 16, 16), animate=True)
# display_tensor_grid(100 * (soft_integrated_grad0 + soft_integrated_grad1).sum(dim=(-1, -2)).view(80, 256, 16, 16), animate=True)

#%%
def print_top_kernels(x, k=10):
    topk = x.sum(dim=(-1, -2)).flatten().topk(k)
    top_indices = topk.indices
    # Convert indices from flat idx tensor to [80, 256, 256] shape indices
    top_indices = t.stack([top_indices // 256 // 256, top_indices // 256 % 256, top_indices % 256], dim=1)
    return top_indices
norm_integrated_grad0_top_idx = print_top_kernels(norm_integrated_grad0)
norm_integrated_grad1_top_idx = print_top_kernels(norm_integrated_grad1)
norm_sub_top_idx = print_top_kernels(norm_integrated_grad0 - norm_integrated_grad1)
norm_add_top_idx = print_top_kernels(norm_integrated_grad0 + norm_integrated_grad1)
grad_list_sub_top_idx = print_top_kernels(integrated_grad_list[0] - integrated_grad_list[1])
grad_list_add_top_idx = print_top_kernels(integrated_grad_list[0] + integrated_grad_list[1])
soft_sub_top_idx = print_top_kernels(soft_integrated_grad0 - soft_integrated_grad1)
soft_add_top_idx = print_top_kernels(soft_integrated_grad0 + soft_integrated_grad1)

# It seems like changing residual_tower[1].conv2 [x, 191] has a huge effect!!!!
# out_chan, in_chan = 4, w
# print("tower[0], conv 1, out_chan", out_chan, "in_chan", in_chan)

multiplier = -1
for name, top_indices in [["norm_integrated_grad0", norm_integrated_grad0_top_idx], ["norm_integrated_grad1", norm_integrated_grad1_top_idx], ["norm_sub", norm_sub_top_idx], ["norm_add", norm_add_top_idx], ["grad_list_sub", grad_list_sub_top_idx], ["grad_list_add", grad_list_add_top_idx], ["soft_sub", soft_sub_top_idx], ["soft_add", soft_add_top_idx], ["Naive grads add", top_indices_simple_grad_add[:1150]]]:
    ys = []
    print(name)
    print(top_indices)
    for top_idx in top_indices:
        lerp_model.load_state_dict(t.load(".cache/best-network.pt"))
        if top_idx[0] % 2 == 0:
            lerp_model.residual_tower[top_idx[0] // 2].conv1.conv.weight.data[top_idx[1], top_idx[2]] *= multiplier
        else:
            lerp_model.residual_tower[top_idx[0] // 2].conv2.conv.weight.data[top_idx[1], top_idx[2]] *= multiplier
        mod_pol, _, _, _, _ = lerp_model(inputs)
        mod_pol_softmax = t.softmax(mod_pol, dim=1)
        ys.append(mod_pol_softmax[1, flat_coord].item())
    # lerp_model.load_state_dict(t.load(".cache/best-network.pt"))
    # for top_idx in top_indices:
    #     if top_idx[0] % 2 == 0:
    #         lerp_model.residual_tower[top_idx[0] // 2].conv1.conv.weight.data[top_idx[1], top_idx[2]] *= multiplier
    #     else:
    #         lerp_model.residual_tower[top_idx[0] // 2].conv2.conv.weight.data[top_idx[1], top_idx[2]] *= multiplier
    # # lerp_model.residual_tower[0].conv1.conv.weight.data[230, 166] *= multiplier
    # mod_pol, _, _, _, _ = lerp_model(inputs)
    # mod_pol_softmax = t.softmax(mod_pol, dim=1)
    # ys.append(mod_pol_softmax[1, flat_coord].item())
    fig = px.bar(x=list(range(len(ys))), y=ys)
    fig.add_hline(y=trained_pol_prob[0, flat_coord].item(), line_dash="dash", line_color="red")
    fig.add_hline(y=trained_pol_prob[1, flat_coord].item(), line_dash="dash", line_color="red")
    fig.show()
    # px.imshow(mod_pol_softmax[:, :-1].view(2, 19, 19).detach().cpu(), origin="lower", facet_col=0).show()

#%%
lerp_model.load_state_dict(t.load(".cache/best-network.pt"))
print("3, 4, 191")
# print("0, 90, 166")
# print("None")
lerp_model.residual_tower[1].conv2.conv.weight.data[4, 191] *= -1
# lerp_model.residual_tower[0].conv1.conv.weight.data[90, 166] *= -1
mod_pol, _, mod_resids, _, _ = lerp_model(inputs)
mod_pol_softmax = t.softmax(mod_pol, dim=1)
px.imshow(mod_pol_softmax[:, :-1].view(2, 19, 19).detach().cpu(), origin="lower", facet_col=0).show()

#%%
resid_diff = trained_resids - mod_resids
display_tensor_grid(resid_diff[:, 0], animate=True)
display_tensor_grid(resid_diff[:, 1], animate=True)

#%%
shifted_inputs = tensor_symmetries(inputs)
# display_tensor_grid(shifted_inputs, animate=True)
print("7, 168, 227")
lerp_model.load_state_dict(t.load(".cache/best-network.pt"))
# lerp_model.residual_tower[0].conv1.conv.weight.data[90, 166] *= -1
# lerp_model.residual_tower[1].conv2.conv.weight.data[4, 191] *= -1
lerp_model.residual_tower[3].conv2.conv.weight.data[168, 227] *= -1
mod_pol, _, mod_resids, _, _ = lerp_model(shifted_inputs)
mod_pol_softmax = t.softmax(mod_pol, dim=1)
px.imshow(mod_pol_softmax[:, :-1].view(-1, 2, 19, 19).detach().cpu(), origin="lower", animation_frame=0, facet_col=1).show()

#%%
lerp_model.load_state_dict(t.load(".cache/best-network.pt"))
lerp_model.residual_tower[1].conv2.conv.weight.data[4, 191]