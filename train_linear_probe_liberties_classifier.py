#%%
import torch as t
from utils import display_tensor_grid
import plotly.express as px
from gather_data import get_data_loader
import LeelaZero
import plotly.graph_objects as go

import torch.nn as nn
import torch as t

liberties_train_loader, liberties_test_loader = get_data_loader("liberties", "/weka/linear-probing/katago-training-parsed", 8)
print("len(liberties_train_loader)", len(liberties_train_loader))
#%%
device = t.device("cuda" if t.cuda.is_available() else "cpu")
layer_count, channel_count = 40, 256
model = LeelaZero.Network(19, 18, channel_count, layer_count).to(device)
model.to(device)
model.load_state_dict(t.load(".cache/best-network.pt"))
model.eval()
LAYERS = [0, 1, 2, 3, 6, 10, 15, 20, 25, 30, 35, 40]
LAYERS_LEN = len(LAYERS)
MAX_LIBS = 8
class Probes(nn.Module):
    def __init__(self):
        super().__init__()
        self.probes = nn.ModuleList([nn.Linear(256 * 19 * 19, MAX_LIBS * 19 * 19) for _ in range(LAYERS_LEN)])
        
    def forward(self, activations):
        layer_outs = []
        for i, layer_idx in enumerate(LAYERS):
            probe = self.probes[i]
            acts = activations[layer_idx].flatten(start_dim=1)
            layer_outs.append(probe(acts))
        out = t.stack(layer_outs)
        return out.view(-1, MAX_LIBS, 19, 19)

probes = Probes().to(device)
#%%
probes.train()
optimizer = t.optim.Adam(probes.parameters(), lr=1e-5)
loss_func = t.nn.CrossEntropyLoss()

EPOCHS = 2
loss_history = []
for epoch in range(EPOCHS):
    for batch_idx, (data, target) in enumerate(liberties_train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        target = t.clamp(target, max=MAX_LIBS -1) # Clamp to 7+ liberties
        target = target.repeat(LAYERS_LEN, 1, 1, 1)
        target = target.view(-1, 19, 19).long()
        with t.no_grad():
            _, _, activations, _, _ = model(data)
        probes_out = probes(activations.detach())
        loss = loss_func(probes_out, target)
        loss.backward()
        loss_history.append(loss.item())
        optimizer.step()
        if batch_idx % 500 == 0:
            print(f"Epoch {epoch}, batch {batch_idx}, loss {loss.item()}")

px.line(y=loss_history).show()

#%%
board_pos, val = (7, 8), 4
display_tensor_grid(probes.probes[0].weight.view(MAX_LIBS, 19, 19, 256, 19, 19)[val, board_pos[0], board_pos[1]].detach().cpu(), file="figs/liberties-classifier-probe-weights-pos-7-8-val-4.html")
#%%
# Test probes
t.cuda.empty_cache()
probes.eval()
total_correct = t.zeros(LAYERS_LEN, 19, 19).to(device)
for batch_idx, (data, target) in enumerate(liberties_test_loader):
    data, target = data.to(device), target.to(device)
    target = target.repeat(LAYERS_LEN, 1, 1, 1)
    target = target.view(-1, 19, 19).long()
    with t.no_grad():
        _, _, activations, _, _ = model(data)
        probes_out = probes(activations.detach())
    top1 = probes_out.argmax(dim=1)
    correct = (top1 == target).view(LAYERS_LEN, -1, 19, 19).sum(dim=1)
    total_correct += correct
    del probes_out
    t.cuda.empty_cache()

correct_rate = total_correct / len(liberties_test_loader.dataset)

print("correct rate", correct_rate.shape)
display_tensor_grid(correct_rate)

#%%
print("average layer error")
px.line(y=correct_rate.cpu().mean(dim=(1, 2)), x=LAYERS)

#%%
# Random probes
random_probes = Probes().to(device)

t.cuda.empty_cache()
random_probes.eval()
random_total_correct = t.zeros(LAYERS_LEN, 19, 19).to(device)
for batch_idx, (data, target) in enumerate(liberties_test_loader):
    data, target = data.to(device), target.to(device)
    target = target.repeat(LAYERS_LEN, 1, 1, 1)
    target = target.view(-1, 19, 19).long()
    with t.no_grad():
        _, _, activations, _, _ = model(data)
        random_probes_out = random_probes(activations.detach())
    top1 = random_probes_out.argmax(dim=1)
    correct = (top1 == target).view(LAYERS_LEN, -1, 19, 19).sum(dim=1)
    random_total_correct += correct
    del random_probes_out
    t.cuda.empty_cache()

random_correct_rate = random_total_correct / len(liberties_test_loader.dataset)

# Plot random and trained probes on the same graph
#%%
fig = go.Figure()
trace1 = go.Scatter(
    y=t.ones(LAYERS_LEN),
    x=LAYERS,
    mode='lines',
    name='Perfect'
)
trace2 = go.Scatter(
    y=correct_rate.cpu().mean(dim=(1, 2)),
    x=LAYERS,
    mode='lines',
    name='Trained'
)
trace3 = go.Scatter(
    y=random_correct_rate.cpu().mean(dim=(1, 2)),
    x=LAYERS,
    mode='lines',
    name='Random'
)
fig.add_trace(trace1)
fig.add_trace(trace2)
fig.add_trace(trace3)
fig.update_layout(title="Linear Probe Performance for Classifying Chain Liberties", xaxis_title="Layer", yaxis_title="Average Correct Rate (Across All Positions)")
fig.show()
#%%
fig.write_html("figs/liberties_classifier.html", include_plotlyjs="cdn")

#%%
actual_liberties = t.stack([target for _, target in liberties_test_loader]).view(-1, 19, 19)
print("actual_liberties", actual_liberties.shape, "len(liberties_test_loader.dataset)", len(liberties_test_loader.dataset))
average_actual_liberties = actual_liberties.mean(dim=0, dtype=t.float32)
px.imshow(average_actual_liberties.cpu(), title="Average Actual Liberties")
print("average_actual_liberties", average_actual_liberties.mean())