#%%
import torch as t
from utils import display_tensor_grid
import plotly.express as px
from gather_data import get_data_loader
import LeelaZero
import plotly.graph_objects as go

import torch.nn as nn
import torch as t

liberties_train_loader, liberties_test_loader = get_data_loader("cycles", "/weka/linear-probing/goattack-training-parsed", 32)
print("len(liberties_train_loader)", len(liberties_train_loader))
print("len(liberties_train_loader.dataset)", len(liberties_train_loader.dataset))
#%%
# for batch_idx, (data, target) in enumerate(liberties_train_loader):
    # non_zero = target.view(-1, 19 * 19).any(dim=1).sum()
    # print("non_zero count", non_zero)
    # display_tensor_grid(target, bord_const=0.1)

#%%
device = t.device("cuda" if t.cuda.is_available() else "cpu")
layer_count, channel_count = 40, 256
model = LeelaZero.Network(19, 18, channel_count, layer_count).to(device)
model.to(device)
model.load_state_dict(t.load(".cache/best-network.pt"))
model.eval()
LAYERS = [0, 1, 2, 3, 6, 10, 15, 20, 25, 30, 35, 40]
LAYERS_LEN = len(LAYERS)
BINARY = 2
class Probes(nn.Module):
    def __init__(self):
        super().__init__()
        self.probes = nn.ModuleList([nn.Linear(256 * 19 * 19, BINARY * 19 * 19) for _ in range(LAYERS_LEN)])
        
    def forward(self, activations):
        layer_outs = []
        for i, layer_idx in enumerate(LAYERS):
            probe = self.probes[i]
            acts = activations[layer_idx].flatten(start_dim=1)
            layer_outs.append(probe(acts))
        out = t.stack(layer_outs)
        return out.view(-1, BINARY, 19, 19)

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
board_pos, val = (11, 8), 0
display_tensor_grid(probes.probes[3].weight.view(BINARY, 19, 19, 256, 19, 19)[val, board_pos[0], board_pos[1]].detach().cpu())
#%%
# Test probes
t.cuda.empty_cache()
probes.eval()
total_correct = t.zeros(LAYERS_LEN, 19, 19).to(device)
true_positives = t.zeros(LAYERS_LEN, 19, 19).to(device)
false_positives = t.zeros(LAYERS_LEN, 19, 19).to(device)
true_negatives = t.zeros(LAYERS_LEN, 19, 19).to(device)
false_negatives = t.zeros(LAYERS_LEN, 19, 19).to(device)
total_positives = t.zeros(LAYERS_LEN, 19, 19).to(device)
total_negatives = t.zeros(LAYERS_LEN, 19, 19).to(device)
for batch_idx, (data, target) in enumerate(liberties_test_loader):
    data, target = data.to(device), target.to(device)
    target = target.repeat(LAYERS_LEN, 1, 1, 1)
    target = target.view(-1, 19, 19).long()
    with t.no_grad():
        _, _, activations, _, _ = model(data)
        probes_out = probes(activations.detach())
    top1 = probes_out.argmax(dim=1)
    correct = (top1 == target).view(LAYERS_LEN, -1, 19, 19)
    incorrect = (top1 != target).view(LAYERS_LEN, -1, 19, 19)
    positives = (target == 1).view(LAYERS_LEN, -1, 19, 19)
    negatives = (target == 0).view(LAYERS_LEN, -1, 19, 19)

    total_correct += correct.sum(dim=1)
    true_positives += (correct * positives).sum(dim=1)
    false_positives += (incorrect * positives).sum(dim=1)
    true_negatives += (correct * negatives).sum(dim=1)
    false_negatives += (incorrect * negatives).sum(dim=1)

    total_positives += positives.sum(dim=1)
    total_negatives += negatives.sum(dim=1)

    del probes_out
    t.cuda.empty_cache()

#%%
correct_rate = total_correct / len(liberties_test_loader.dataset)
true_positive_rate = true_positives / total_positives
false_positive_rate = false_positives / total_positives
true_negative_rate = true_negatives / total_negatives
false_negative_rate = false_negatives / total_negatives

precision = true_positives / (true_positives + false_positives)
sensitivity = true_positives / (true_positives + false_negatives)
# sensitivity gives nan when there are no positives, so replace with 1 if true_positives is 0 and 0 otherwise
sensitivity = t.where(t.isnan(sensitivity), t.where(true_positives == 0, t.zeros_like(sensitivity), t.ones_like(sensitivity)), sensitivity)
specificity = true_negatives / (true_negatives + false_positives)

print("correct rate", correct_rate.shape)
display_tensor_grid(correct_rate)
print("true positive rate", true_positive_rate.shape)
display_tensor_grid(true_positive_rate)
print("false positive rate", false_positive_rate.shape)
display_tensor_grid(false_positive_rate)
print("true negative rate", true_negative_rate.shape)
display_tensor_grid(true_negative_rate)

#%%
print("average layer error")
px.line(t.stack([true_positive_rate.mean(dim=(1,2)), false_positive_rate.mean(dim=(1,2)), true_negative_rate.mean(dim=(1,2)), false_negative_rate.mean(dim=(1,2))]).cpu().T).show()
px.line(y=correct_rate.cpu().mean(dim=(1, 2)), x=LAYERS)

# Plot precision and sensitivity
px.line(y=precision.cpu().mean(dim=(1, 2)), x=LAYERS).show()
px.line(y=sensitivity.cpu().mean(dim=(1, 2)), x=LAYERS).show()
px.line(y=specificity.cpu().mean(dim=(1, 2)), x=LAYERS).show()

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
fig.update_layout(title="Linear Probe Performance for Classifying Cyclic Groups", xaxis_title="Layer", yaxis_title="Average Correct Rate (Across All Positions)")
fig.show()
#%%
fig.write_html("figs/liberties_classifier.html", include_plotlyjs="cdn")

#%%
# Plot precision, sensitivity and specificity on the same graph
fig = go.Figure()
trace1 = go.Scatter(
    y=precision.cpu().mean(dim=(1, 2)),
    x=LAYERS,
    mode='lines',
    name='Precision<br>True Positive / Total Positive Predictions'
)
trace2 = go.Scatter(
    y=sensitivity.cpu().mean(dim=(1, 2)),
    x=LAYERS,
    mode='lines',
    name='Sensitivity<br>True Positive / Total Actual Positives'
)
trace3 = go.Scatter(
    y=specificity.cpu().mean(dim=(1, 2)),
    x=LAYERS,
    mode='lines',
    name='Specificity<br>True Negative / Total Actual Negatives'
)
fig.add_trace(trace3)
fig.add_trace(trace2)
fig.add_trace(trace1)
fig.update_layout(title="Linear Probe Performance for Classifying Cyclic Groups", xaxis_title="Layer", yaxis_title="Average Across All Positions")
fig.show()


# %%

#%%
actual_liberties = t.stack([target for _, target in liberties_test_loader]).view(-1, 19, 19)
print("actual_liberties", actual_liberties.shape, "len(liberties_test_loader.dataset)", len(liberties_test_loader.dataset))
average_actual_liberties = actual_liberties.mean(dim=0, dtype=t.float32)
px.imshow(average_actual_liberties.cpu(), title="Average Actual Liberties")
print("average_actual_liberties", average_actual_liberties.mean())