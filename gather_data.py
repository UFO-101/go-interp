#%%
from sgfmill import sgf
from sgfmill import sgf_moves
import random
from sgfmill import ascii_boards
import torch as t
from utils import display_tensor_grid, print_go_board_tensor
import plotly.express as px
import time
import os
from multiprocessing import Pool, cpu_count
from torch.utils.data import Dataset, DataLoader, random_split
#%%

t.set_printoptions(linewidth=200)

start_time = time.time()

PATH = "/Users/josephmiller/Downloads/ladder-with-breaker.sgf"
# PATH = "/Users/josephmiller/Downloads/connection_test_realgame2_B.sgf"
# PATH = "/Users/josephmiller/Downloads/connection_test_realgame2_A.sgf"

def board_state_to_2d_tensors(board):
    board_size = board.side
    b_tensor = t.zeros((board_size, board_size), dtype=t.float32)
    w_tensor = t.zeros((board_size, board_size), dtype=t.float32)
    for new_color, (row, col) in board.list_occupied_points():
        if new_color == 'b':
            b_tensor[row, col] = 1
        else:
            w_tensor[row, col] = 1
    return b_tensor, w_tensor


def get_board_and_leela_input(game):
    board, plays = sgf_moves.get_setup_and_moves(game)
    game_len = len(plays)

    sample_len = random.randint(1, game_len)
    b_tensors, w_tensors = [], []
    color = None
    try:
        for i, (new_color, pos) in enumerate(plays[:sample_len]):
            if pos is None:
                continue
            row, column = pos 
            color = new_color
            board.play(row, column, color)
            if i >= sample_len - 8:
                b_tensor, w_tensor = board_state_to_2d_tensors(board)
                b_tensors.append(b_tensor)
                w_tensors.append(w_tensor)
    except Exception as e:
        print("ERROR", e)
        print("plays", plays)

    b_tensors.reverse(), w_tensors.reverse()
    b_tensors += [t.zeros_like(b_tensors[0])] * (8 - len(b_tensors))
    w_tensors += [t.zeros_like(w_tensors[0])] * (8 - len(w_tensors))
    move_tensors = w_tensors + b_tensors if color == 'b' else b_tensors + w_tensors
    b_turn_tensors = [t.ones_like(move_tensors[0]), t.zeros_like(move_tensors[0])]
    w_turn_tensors = [t.zeros_like(move_tensors[0]), t.ones_like(move_tensors[0])]
    turn_tensors = w_turn_tensors if color == 'b' else b_turn_tensors
    leela_input = t.stack(move_tensors + turn_tensors)
    return board, leela_input

# print(ascii_boards.render_board(board))
# display_tensor_grid(leela_input, bord_const=0.1)

def liberties_and_cycles(board):
    stones = {(r, c): col for col, (r, c) in board.list_occupied_points()}
    for color, (row, col) in board.list_occupied_points():
        stones[(row, col)] = color
    
    def add_stone_to_group(group, stone):
        row, col = stone
        neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        group.add(stone)
        stones.pop(stone)
        for neighbor in neighbors:
            if neighbor in stones.keys() and stones[neighbor] == color:
                add_stone_to_group(group, neighbor)
    
    groups = {}
    while stones:
        # Get any stone withour removing
        pos, color = list(stones.items())[0]
        group = set()
        add_stone_to_group(group, pos)
        groups[pos] = group
    
    stones_2 = {(r, c): col for col, (r, c) in board.list_occupied_points()}
    liberties = {}
    for pos, group in groups.items():
        liberties[pos] = set()
        for row, col in group:
            neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            for (r, c) in neighbors:
                if r >= 0 and r < board.side and c >= 0 and c < board.side:
                    if (r, c) not in stones_2.keys():
                        liberties[pos].add((r, c))
    
    edges = set([(0, col) for col in range(board.side)] + \
            [(board.side-1, col) for col in range(board.side)] + \
            [(row, 0) for row in range(board.side)] + \
            [(row, board.side-1) for row in range(board.side)])

    cyclic_groups = {}
    for pos, group in groups.items():
        reached = [pos for pos in edges if pos not in stones_2.keys()]
        unused = reached.copy()

        while unused:
            (row, col) = unused.pop()
            neighbors = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            neighbors += [(row-1, col-1), (row+1, col+1), (row-1, col+1), (row+1, col-1)]
            for (r, c) in neighbors:
                if r >= 0 and r < board.side and c >= 0 and c < board.side:
                    if (r, c) not in group and (r, c) not in reached:
                        reached.append((r, c))
                        unused.append((r, c))
        
        cyclic = len(reached) + len(group) < board.side**2
        cyclic_groups[pos] = cyclic
    
    return groups, liberties, cyclic_groups

def file_to_data(path):
    games = []
    with open(path, "r") as f:
        for line_str in f:
            games.append(sgf.Sgf_game.from_string(line_str))

    boards, leela_inputs = [], []
    for game in games:
        board, leela_input = get_board_and_leela_input(game)
        boards.append(board)
        leela_inputs.append(leela_input)
    
    group_list, liberties_list, cyclic_list = [], [], []
    for board in boards:
        groups, liberties, cyclic_groups = liberties_and_cycles(board)
        group_list.append(groups)
        liberties_list.append(liberties)
        cyclic_list.append(cyclic_groups)


    liberties_tensors = []
    for groups, liberties in zip(group_list, liberties_list):
        liberties_tensor = t.zeros((board.side, board.side), dtype=t.int32)
        for pos, group in groups.items():
            for row, col in group:
                liberties_tensor[row, col] = len(liberties[pos])
        liberties_tensors.append(liberties_tensor)

    cyclic_tensors = []
    for groups, cyclic_groups in zip(group_list, cyclic_list):
        cyclic_tensor = t.zeros((board.side, board.side), dtype=t.int32)
        for pos, group in groups.items():
            for row, col in group:
                cyclic_tensor[row, col] = cyclic_groups[pos]
        cyclic_tensors.append(cyclic_tensor)
    
    # px.imshow(liberties_tensors[0], origin="lower").show()
    # px.imshow(cyclic_tensors[0], origin="lower").show()
    return leela_inputs, liberties_tensors, cyclic_tensors

end_time = time.time()

def process_file(file_path):
    inputs, liberties, cycles = file_to_data(file_path)
    if file_path.endswith(".sgf") or file_path.endswith(".sgfs"):
        return (file_path, [inputs, liberties, cycles])
    else:
        return None, None

def main(input_dir, output_dir):
    pool = Pool(cpu_count())  # Create a multiprocessing Pool
    file_paths = [os.path.join(root, file) for root, _, files in os.walk(input_dir) for file in files]
    print("File paths len", len(file_paths))
    print("File paths", file_paths[:10])
    results = pool.map(process_file, file_paths)
    
    tensor_dict = {file_path: tensors for file_path, tensors in results if tensors is not None}
    
    print("Tensor dict len", len(tensor_dict))
    print("tensor dict keys", tensor_dict.keys())
    # Write tensor_dict to file in output_dir
    t.save(tensor_dict, os.path.join(output_dir, 'tensor_dict.pt'))

# input_dir = "/Users/josephmiller/Downloads/"
# output_dir = "/Users/josephmiller/Documents/go-interp/parsed_data"
# main(input_dir, output_dir)

if __name__ == "__main__":
    input_dir = "/Users/josephmiller/Documents/FAR/cp505-vs-cp103/sgfs"
    output_dir = "/Users/josephmiller/Documents/go-interp/parsed_data"
    main(input_dir, output_dir)

# print("Game len", game_len, "Sample len", sample_len)
# print("Time taken: {:.2f}s".format(end_time - start_time))

#%%

class ProbeDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def get_data_loader(type, batch_size=64):
    parsed_data = t.load("/Users/josephmiller/Documents/go-interp/parsed_data/tensor_dict.pt")
    print(len(parsed_data))

    combined_leela_inputs = []
    combined_liberties = []
    combined_cycles = []

    for file_path, (leela_inputs, liberties, cycles) in parsed_data.items():
        combined_leela_inputs.extend(leela_inputs)
        combined_liberties.extend(liberties)
        combined_cycles.extend(cycles)

    if type == "liberties":
        dataset = ProbeDataset(combined_leela_inputs, combined_liberties)

    else:
        dataset = ProbeDataset(combined_leela_inputs, combined_cycles)

    train_data, test_data = random_split(dataset, [0.9, 0.1])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
