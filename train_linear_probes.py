#%%
import torch as t
from utils import display_tensor_grid
import plotly.express as px
from gather_data import get_data_loader
import LeelaZero
import torch.nn as nn

liberties_train_loader, liberties_test_loader = get_data_loader("liberties", "/weka/linear-probing/katago-training-parsed", 32)
print("len(liberties_train_loader)", len(liberties_train_loader))
#%%
device = t.device("cuda" if t.cuda.is_available() else "cpu")
layer_count, channel_count = 40, 256
model = LeelaZero.Network(19, 18, channel_count, layer_count).to(device)
model.to(device)
model.load_state_dict(t.load(".cache/best-network.pt"))
model.eval()
class Probes(nn.Module):
    def __init__(self):
        super().__init__()
        self.probes = nn.ModuleList([nn.Linear(256 * 19 * 19, 19 * 19) for _ in range(layer_count + 1)])
        
    def forward(self, activations):
        layer_outs = []
        for layer_acts, probe in zip(activations, self.probes):
            layer_outs.append(probe(layer_acts.flatten(start_dim=1)))
        out = t.stack(layer_outs)
        return out.view(41, -1, 19, 19)

probes = Probes().to(device)
#%%
probes.train()
optimizer = t.optim.Adam(probes.parameters(), lr=1e-5)

EPOCHS = 10
loss_history = []
for epoch in range(EPOCHS):
    for batch_idx, (data, target) in enumerate(liberties_train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        with t.no_grad():
            _, _, activations, _, _ = model(data)
        probes_out = probes(activations.detach())
        loss = (target - probes_out).abs().mean()
        loss.backward()
        loss_history.append(loss.item())
        optimizer.step()
        if batch_idx % 200 == 0:
            print(f"Epoch {epoch}, batch {batch_idx}, loss {loss.item()}")

px.line(y=loss_history).show()

#%%
display_tensor_grid(probes.probes[0].weight[180].view(256, 19, 19).detach().cpu())
#%%
# Test probes
t.cuda.empty_cache()
probes.eval()
total_error = t.zeros(41, 19, 19).to(device)
for batch_idx, (data, target) in enumerate(liberties_test_loader):
    data, target = data.to(device), target.to(device)
    with t.no_grad():
        _, _, activations, _, _ = model(data)
        probes_out = probes(activations.detach())
    error = (probes_out - target).abs().sum(dim=1)
    total_error += error
    del probes_out, error
    t.cuda.empty_cache()

average_error = total_error / len(liberties_test_loader.dataset)

print("total_error", total_error.shape)
display_tensor_grid(total_error)

print("average layer error")
px.line(average_error.cpu().mean(dim=(1, 2)))

#%%
# Random probes
random_probes = Probes().to(device)

t.cuda.empty_cache()
random_probes.eval()
random_total_error = t.zeros(41, 19, 19).to(device)
for batch_idx, (data, target) in enumerate(liberties_test_loader):
    data, target = data.to(device), target.to(device)
    with t.no_grad():
        _, _, activations, _, _ = model(data)
        random_probes_out = random_probes(activations.detach())
    error = (random_probes_out - target).abs().sum(dim=1)
    random_total_error += error
    del random_probes_out, error
    t.cuda.empty_cache()

random_average_error = random_total_error / len(liberties_test_loader.dataset)

# Plot random and trained probes on the same graph
#%%
fig = px.line(t.stack([random_average_error.cpu().mean(dim=(1, 2)), average_error.cpu().mean(dim=(1, 2)), t.zeros(41)]).T, labels={"index": "Layer", "value": "Average Absolute Error (Across All Positions)", "variable": "Probe Type"}, title="Linear Probe Performance for Counting Chain Liberties")

newnames = {'0': 'Random', '1': 'Trained', '2': 'Perfect'} # From the other post
fig.for_each_trace(lambda t: t.update(name = newnames[t.name])).show()

#%%
actual_liberties = t.stack([target for _, target in liberties_test_loader]).view(-1, 19, 19)
print("actual_liberties", actual_liberties.shape, "len(liberties_test_loader.dataset)", len(liberties_test_loader.dataset))
average_actual_liberties = actual_liberties.mean(dim=0, dtype=t.float32)
px.imshow(average_actual_liberties.cpu(), title="Average Actual Liberties")
print("average_actual_liberties", average_actual_liberties.mean())