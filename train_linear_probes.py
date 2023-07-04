#%%
import torch as t
from utils import display_tensor_grid
import plotly.express as px
from gather_data import get_data_loader
import LeelaZero
import torch.nn as nn
import pandas as pd
import plotly.graph_objects as go
#%%

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

#%%
px.line(y=loss_history).show()
#%%
# Save/load probes to/from disk
# t.save(probes.state_dict(), ".cache/liberties-value-probes.pt")
probes.load_state_dict(t.load(".cache/liberties-value-probes.pt"))

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

#%%
# Test on cycles from the other dataset
inputs, liberties, cycles = get_data_loader("all-raw", "/weka/linear-probing/goattack-training-parsed", 32)
#%%
from tqdm import tqdm

print("len(inputs)", len(inputs), "len(liberties)", len(liberties), "len(cycles)", len(cycles))

cyclic_points = []
noncyclic_points = []
batch_size = 248
for board_idx in tqdm(range(0, len(inputs), batch_size)):
    input_boards = t.stack(inputs[board_idx:board_idx + batch_size])
    liberties_boards = t.stack(liberties[board_idx:board_idx + batch_size])
    cycle_boards = t.stack(cycles[board_idx:board_idx + batch_size])
    with t.inference_mode():
        _, _, activations, _, _ = model(input_boards.to(device))
        probes_out = probes(activations.detach())
        probes_out = probes_out[2] # Only look at the probes for the 3rd layer
    # Get non-zero positions
    cyclic_indices = t.nonzero(cycle_boards)
    noncyclic_indices = t.nonzero(cycle_boards == 0)
    cyclic_indices_liberties = liberties_boards[cyclic_indices[:, 0], cyclic_indices[:, 1], cyclic_indices[:, 2]]
    cyclic_indices_probes = probes_out[cyclic_indices[:, 0], cyclic_indices[:, 1], cyclic_indices[:, 2]]
    new_cyclic_points = t.stack([cyclic_indices_liberties, cyclic_indices_probes.cpu()], dim=1)
    cyclic_points.append(new_cyclic_points)

    noncyclic_indices_liberties = liberties_boards[noncyclic_indices[:, 0], noncyclic_indices[:, 1], noncyclic_indices[:, 2]]
    noncyclic_indices_probes = probes_out[noncyclic_indices[:, 0], noncyclic_indices[:, 1], noncyclic_indices[:, 2]]
    new_noncylic_points = t.stack([noncyclic_indices_liberties, noncyclic_indices_probes.cpu()], dim=1)
    noncyclic_points.append(new_noncylic_points)

cyclic_points = t.cat(cyclic_points)
noncyclic_points = t.cat(noncyclic_points)
print("cyclic   points", cyclic_points.shape, "noncyclic_points", noncyclic_points.shape)
# %%

# Group the points by liberties and plot the average probe output for each group
cyclic_points_df = pd.DataFrame(cyclic_points.cpu().numpy(), columns=["Liberties", "Probe Output"])
noncyclic_points_df = pd.DataFrame(noncyclic_points.cpu().numpy(), columns=["Liberties", "Probe Output"])

# Compute the average probe output for each group
cyclic_points_df = cyclic_points_df.groupby("Liberties").mean().reset_index()
noncyclic_points_df = noncyclic_points_df.groupby("Liberties").mean().reset_index()

# Plot as two line plots on the same graph
fig = px.line(cyclic_points_df, x="Liberties", y="Probe Output", title="Average Probe Output vs. Actual Number of Liberties", labels={"Liberties": "Liberties", "Probe Output": "Probe Output"})
# Label the first line as Cyclic Points
fig.add_scatter(x=noncyclic_points_df["Liberties"], y=noncyclic_points_df["Probe Output"], mode="lines", name="Noncyclic Points")
fig.show()

line1 = go.Scatter(x=cyclic_points_df["Liberties"], y=cyclic_points_df["Probe Output"], mode='lines', name='Cyclic Points')

# Create a line for the 'noncyclic_points_df'
line2 = go.Scatter(x=noncyclic_points_df["Liberties"], y=noncyclic_points_df["Probe Output"], mode='lines', name='Noncyclic Points')

# Create a layout
layout = go.Layout(title="Average Probe Output vs. Actual Number of Liberties", 
                   xaxis_title="Liberties", 
                   yaxis_title="Probe Output")

# Create a figure
fig = go.Figure(data=[line1, line2], layout=layout)

# Show the plot
fig.show()
