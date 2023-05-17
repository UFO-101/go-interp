#%%
import math
import torch
import torchvision
import plotly.express as px

def print_go_board_tensor(tensor, color, show_n_turns_past=0, only_to_play=False, coords=None):
    tensor = tensor[0] if isinstance(tensor, tuple) else tensor
    tensor = tensor.squeeze()
    assert isinstance(coords, list) or coords is None
    color = color if isinstance(color, str) else ('b' if color % 2 == 0 else 'w')
    print("Color to play:", color)
    print("coords", coords)
    for i in reversed(range(19)):
        for j in range(19):
            if coords is not None and (j, i) in coords or j * 19 + i in coords:
                black_symbol = "🔴"
                white_symbol = "🟡"
                empty_symbol = "❌"
            else:
                black_symbol = "🟤"
                white_symbol = "⚪"
                empty_symbol = "➕"
            if tensor[0 + show_n_turns_past][i][j] == 1:
                print(black_symbol, end="") if color == 'b' else print(white_symbol, end="")
            elif tensor[8 + show_n_turns_past][i][j] == 1 and not only_to_play:
                print(white_symbol, end="") if color == 'b' else print(black_symbol, end="")
            else:
                print(empty_symbol, end="")
        print()

def custom_make_grid(tensor, border=0):
    nrow = math.floor(math.sqrt(tensor.shape[0]))
    height, width = tensor.shape[-2], tensor.shape[-1]
    tensor = tensor.unsqueeze(-3)
    tensor = torchvision.utils.make_grid(tensor, nrow, 1, pad_value=border)
    return tensor[..., 0, :, :], height + 1, width + 1, nrow

def leq_4d_to_grid(tensor, bord_const=0):
    assert len(tensor.shape) <= 4
    layers, border = [], torch.min(tensor).item() * 1.1 - bord_const
    for i in range(tensor.shape[0]):
        grid, _, _, _ = custom_make_grid(tensor[i], border=border)
        layers.append(grid)
    return custom_make_grid(torch.stack(layers), border=border)

def display_tensor_grid(act, title=None, animate=False, file=None, bord_const=0):
    print('displaying tensor grid:', act.shape)
    act = act.detach().cpu().clone().squeeze()
    assert len(act.shape) <= 4 or (len(act.shape) == 5 and animate)
    if animate:
        layers = []
        for i in range(act.shape[0]):
            grid, h, w, nrow = leq_4d_to_grid(act[i], bord_const)
            layers.append(grid)
        grid_h, grid_w = grid.shape[0], grid.shape[1]
        grid = torch.stack(layers)
        fig = px.imshow(grid.cpu().numpy(), origin="lower", animation_frame=0, height=800, title=title)
    else:
        grid, h, w, nrow = leq_4d_to_grid(act, bord_const)
        grid_h, grid_w = grid.shape[0], grid.shape[1]
        fig = px.imshow(grid.cpu().numpy(), origin="lower", height=800, title=title)

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(int(w / 2), grid_w, w)),
            ticktext=list(range(0, nrow)),
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(int(h / 2), grid_h, h)),
            ticktext=list(range(0, grid_h * nrow, nrow)),
        )
    )
    if file is None:
        fig.update_layout(margin=dict(l=0, r=105, t=0, b=0, pad=0))
        fig.show()
    else:
        fig.write_html(file, auto_play=False, include_plotlyjs='cdn')

def tensor_symmetries(x):
    x = x.squeeze()
    return torch.cat(
        [torch.rot90(x, i, [-1, -2]) for i in range(4)] +
        [torch.flip(x, [-1]),
         torch.flip(x, [-2]),
         torch.transpose(x, -2, -1),
         torch.flip(torch.transpose(x, -1, -2), [-1, -2])])
    
def shifted_tensors(x):
    x = x.squeeze()
    return torch.cat(
        [torch.roll(x, i, -1) for i in [1, -1]] +
        [torch.roll(x, i, -2) for i in [1, -1]])
    
def inverse_tensor_symmetries(x):
    return torch.cat(
        [torch.rot90(x[i], i, [-2, -1]) for i in range(4)] +
        [torch.flip(x[4], [-1]),
         torch.flip(x[5], [-2]),
         torch.transpose(x[6], -2, -1),
         torch.transpose(torch.flip(x[7], [-1, -2]), -1, -2)])
# Score
# Edited: 8 (5W, 3B)
# Default: 9 (8W, 1B)

# Ladder losses
# 1W, 2B

def point_symmetries(x, y):
    points = [(x, y)]
    points.append((18 - y, x))
    points.append((18 - x, 18 - y))
    points.append((y, 18 - x))
    points.append((18 - x, y))
    points.append((x, 18 - y))
    points.append((y, x))
    points.append((18 - y, 18 - x))
    total_indices = [19 * y + x for x, y in points]
    return points, total_indices
