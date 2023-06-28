#%%
import torch as t
from utils import display_tensor_grid
import plotly.express as px
from gather_data import get_data_loader
import LeelaZero
import torch.nn as nn

liberties_train_loader, liberties_test_loader = get_data_loader("liberties", 16)
print("len(liberties_train_loader)", len(liberties_train_loader))
liberties_train_loader

device = t.device("cuda" if t.cuda.is_available() else "cpu")
layer_count, channel_count = 40, 256
model = LeelaZero.Network(19, 18, channel_count, layer_count).to(device)
model.load_state_dict(t.load(".cache/best-network.pt"))
model.eval()

class Probes(nn.Module):
    def __init__(self):
        super().__init__()
        self.probes = [nn.Linear(256 * 19 * 19, 19 * 19)] * (layer_count + 1)
        
    def forward(self, activations):
        layer_outs = []
        for layer_acts, probe in zip(activations, self.probes):
            layer_outs.append(probe(layer_acts.flatten(start_dim=1)))
        out = t.stack(layer_outs)
        return out.view(41, -1, 19, 19)

probes = Probes().to(device)
probes.train()
optimizer = t.optim.Adam(model.parameters(), lr=1e-2)

#%%
EPOCHS = 1
loss_history = []
for epoch in range(EPOCHS):
    for batch_idx, (data, target) in enumerate(liberties_train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        _, _, activations, _, _ = model(data)
        probes_out = probes(activations.detach().clone())
        loss = (target - probes_out).abs().sum()
        loss.backward()
        loss_history.append(loss.item())
        optimizer.step()
        print(f"Epoch {epoch}, batch {batch_idx}, loss {loss.item()}")

px.line(y=loss_history).show()

#%%
# Test probes

probes.eval()
for batch_idx, (data, target) in enumerate(liberties_train_loader):
    data, target = data.to(device), target.to(device)
    _, _, activations, _, _ = model(data)
    probes_out = probes(activations.detach().clone())