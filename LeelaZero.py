# Based on https://github.com/yukw777/leela-zero-pytorch
import gzip
import torch
import torch.nn as nn
import torch.nn.functional as F

from io import TextIOWrapper
from typing import List


class ConvBlock(nn.Module):
    """
    A convolutional block with a convolution layer, batchnorm (with beta) and
    an optional relu

    Note on the bias for the convolutional layer:
    Leela Zero actually uses the bias for the convolutional layer to represent
    the learnable parameters (gamma and beta) of the following batch norm layer.
    This was done so that the format of the weights file, which only has one line
    for the layer weights and another for the bias, didn't have to change when
    batch norm layers were added.

    Currently, Leela Zero only uses the beta term of batch norm, and sets gamma to 1.
    Then, how do you actually use the convolutional bias to produce the same results
    as applying the learnable parameters in batch norm? Let's first take
    a look at the equation for batch norm:

    y = gamma * (x - mean)/sqrt(var - eps) + beta

    Since Leela Zero sets gamma to 1, the equation becomes:

    y = (x - mean)/sqrt(var - eps) + beta

    Now, let `x_conv` be the output of a convolutional layer without the bias.
    Then, we want to add some bias to `x_conv`, so that when you run it through
    batch norm without `beta`, the result is the same as running `x_conv`
    through the batch norm equation with only beta mentioned above. In an equation form:

    (x_conv + bias - mean)/sqrt(var - eps) = (x_conv - mean)/sqrt(var - eps) + beta
    x_conv + bias - mean = x_conv - mean + beta * sqrt(var - eps)
    bias = beta * sqrt(var - eps)

    So if we set the convolutional bias to `beta * sqrt(var - eps)`, we get the desired
    output, and this is what LeelaZero does.

    In Tensorflow, you can tell the batch norm layer to ignore just the gamma term
    by calling `tf.layers.batch_normalization(scale=False)` and be done with it.
    Unfortunately, in PyTorch you can't set batch normalization layers to ignore only
    `gamma`; you can only ignore both `gamma` and `beta` by setting the affine
    parameter to False: `BatchNorm2d(out_channels, affine=False)`. So, ConvBlock sets
    batch normalization to ignore both, then simply adds a tensor after, which
    represents `beta`.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, relu: bool = True
    ):
        super().__init__()
        # we only support the kernel sizes of 1 and 3
        assert kernel_size in (1, 3)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=1 if kernel_size == 3 else 0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.beta = nn.Parameter(torch.zeros(out_channels))  # type: ignore
        self.relu = relu

        # initializations
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, input, leaky_relu: bool = False):
        a = self.conv(input)
        b = self.bn(a)
        c = b + self.beta.view(1, self.bn.num_features, 1, 1).expand_as(b)
        if leaky_relu:
            return F.leaky_relu(c) if self.relu else c
        else:
            return F.relu(c) if self.relu else c


# *Residualified* residual block
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 3)
        self.conv2 = ConvBlock(out_channels, out_channels, 3, relu=False)

    def forward(self, x, leaky_relu: bool = False):
        identity = x
        mid = self.conv1(x, leaky_relu=leaky_relu)
        out = self.conv2(mid)
        if leaky_relu:
            return torch.where(identity + out > 0, out, -identity * 0.99), mid
        else:
            return torch.where(identity + out > 0, out, -identity), mid

class Network(nn.Module):
    def __init__(
        self,
        board_size: int,
        in_channels: int,
        residual_channels: int,
        residual_layers: int,
    ):
        super().__init__()
        self.conv_input = ConvBlock(in_channels, residual_channels, 3)
        self.residual_tower = nn.Sequential(
            *[
                ResBlock(residual_channels, residual_channels)
                for _ in range(residual_layers)
            ]
        )
        self.policy_conv = ConvBlock(residual_channels, 2, 1)
        self.policy_fc = nn.Linear(
            2 * board_size * board_size, board_size * board_size + 1
        )
        self.value_conv = ConvBlock(residual_channels, 1, 1)
        self.value_fc_1 = nn.Linear(board_size * board_size, 256)
        self.value_fc_2 = nn.Linear(256, 1)

    def forward(self, planes, leaky_relu: bool = False):
        resids = []
        block_activations = []

        # first conv layer
        x = self.conv_input(planes, leaky_relu=leaky_relu)
        resids.append(x.detach().clone())

        # residual tower
        for layer in self.residual_tower:
            layer_output, intermediate = layer(x, leaky_relu=leaky_relu)
            # block_activations.append(intermediate.detach().clone())
            block_activations.append(layer_output.detach().clone())
            x = x + layer_output
            resids.append(x)

        # policy head
        pol = self.policy_conv(x, leaky_relu=leaky_relu)
        pol = self.policy_fc(torch.flatten(pol, start_dim=1))

        # value head
        val = self.value_conv(x, leaky_relu=leaky_relu)
        if leaky_relu:
            val = F.leaky_relu(self.value_fc_1(torch.flatten(val, start_dim=1)), inplace=True)
        else:
            val = F.relu(self.value_fc_1(torch.flatten(val, start_dim=1)), inplace=True)
        val = torch.tanh(self.value_fc_2(val))

        return pol, val, torch.stack(resids), torch.stack(block_activations)

    def to_leela_weights(self, filename: str):
        """
        Save the current weights in the Leela Zero format to the given file name.

        The Leela Zero weights format:

        The residual tower is first, followed by the policy head, and then the value
        head. All convolution filters are 3x3 except for the ones at the start of
        the policy and value head, which are 1x1 (as in the paper).

        Convolutional layers have 2 weight rows:
            1. convolution weights w/ shape [output, input, filter_size, filter_size]
            2. channel biases
        Batchnorm layers have 2 weight rows:
            1. batchnorm means
            2. batchnorm variances
        Innerproduct (fully connected) layers have 2 weight rows:
            1. layer weights w/ shape [output, input]
            2. output biases

        Therefore, the equation for the number of layers is

        n_layers = 1 (version number) +
                   2 (input convolution) +
                   2 (input batch norm) +

                   n_res (number of residual blocks) *
                   8 (first conv + first batch norm + second conv + second batch norm) +
                   
                   2 (policy head convolution) +
                   2 (policy head batch norm) +
                   2 (policy head linear) +
                   
                   2 (value head convolution) +
                   2 (value head batch norm) +
                   2 (value head first linear) +
                   
                   2 (value head second linear)
        """
        with gzip.open(filename, "wb") as f:
            # version tag
            f.write(b"1\n")
            for child in self.children():
                # newline unless last line (single bias)
                if isinstance(child, ConvBlock):
                    f.write(self.conv_block_to_leela_weights(child).encode("utf-8"))
                elif isinstance(child, nn.Linear):
                    f.write(self.tensor_to_leela_weights(child.weight).encode("utf-8"))
                    f.write(self.tensor_to_leela_weights(child.bias).encode("utf-8"))
                elif isinstance(child, nn.Sequential):
                    # residual tower
                    for grand_child in child.children():
                        if isinstance(grand_child, ResBlock):
                            f.write(self.conv_block_to_leela_weights(grand_child.conv1).encode("utf-8"))
                            f.write(self.conv_block_to_leela_weights(grand_child.conv2).encode("utf-8"))
                        else:
                            err = (
                                "Sequential should only have ResBlocks, but found "
                                + str(type(grand_child))
                            )
                            raise ValueError(err)
                elif isinstance(child, pl.metrics.Accuracy):
                    continue
                else:
                    raise ValueError("Unknown layer type" + str(type(child)))
                
    def from_leela_weights(self, filename: str):
        state_dict = {}
        
        if filename.endswith(".gz") or filename.endswith("best-network"):
            with gzip.open(filename, "rt", encoding='utf-8') as f:
                # calculate n_res from the number of lines
                num_lines = sum(1 for _ in f)
                print("num_lines", num_lines)
                n_res = (num_lines - 19) // 8
                assert num_lines == 19 + 8 * n_res

            with gzip.open(filename, "rt", encoding='utf-8') as f:
                version = f.readline().strip()
                assert version == "1"
                # input conv block
                state_dict.update(self.leela_weights_to_conv_block(f, "conv_input", self.conv_input))
                # residual tower
                for i in range(n_res):
                    state_dict.update(self.leela_weights_to_conv_block(f, f"residual_tower.{i}.conv1", self.residual_tower[i].conv1))
                    state_dict.update(self.leela_weights_to_conv_block(f, f"residual_tower.{i}.conv2", self.residual_tower[i].conv2))
                # policy head
                state_dict.update(self.leela_weights_to_conv_block(f, "policy_conv", self.policy_conv))
                state_dict.update(self.leela_weights_to_linear_layer(f, "policy_fc", self.policy_fc))
                # value head
                state_dict.update(self.leela_weights_to_conv_block(f, "value_conv", self.value_conv))
                state_dict.update(self.leela_weights_to_linear_layer(f, "value_fc_1", self.value_fc_1))
                state_dict.update(self.leela_weights_to_linear_layer(f, "value_fc_2", self.value_fc_2))
        else:
            with open(filename, "rt", encoding='utf-8') as f:
                # calculate n_res from the number of lines
                num_lines = sum(1 for _ in f)
                n_res = (num_lines - 19) // 8
                assert num_lines == 19 + 8 * n_res
            
            with open(filename, "rt", encoding='utf-8') as f:
                version = f.readline().strip()
                assert version == "1"
                # input conv block
                state_dict.update(self.leela_weights_to_conv_block(f, "conv_input", self.conv_input))
                # residual tower
                for i in range(n_res):
                    state_dict.update(self.leela_weights_to_conv_block(f, f"residual_tower.{i}.conv1", self.residual_tower[i].conv1))
                    state_dict.update(self.leela_weights_to_conv_block(f, f"residual_tower.{i}.conv2", self.residual_tower[i].conv2))
                # policy head
                state_dict.update(self.leela_weights_to_conv_block(f, "policy_conv", self.policy_conv))
                state_dict.update(self.leela_weights_to_linear_layer(f, "policy_fc", self.policy_fc))
                # value head
                state_dict.update(self.leela_weights_to_conv_block(f, "value_conv", self.value_conv))
                state_dict.update(self.leela_weights_to_linear_layer(f, "value_fc_1", self.value_fc_1))
                state_dict.update(self.leela_weights_to_linear_layer(f, "value_fc_2", self.value_fc_2))
        self.load_state_dict(state_dict)
        

    @staticmethod
    def conv_block_to_leela_weights(conv_block: ConvBlock) -> str:
        weights = []
        weights.append(Network.tensor_to_leela_weights(conv_block.conv.weight))
        # calculate beta * sqrt(var - eps)
        bias = conv_block.beta * torch.sqrt(
            conv_block.bn.running_var - conv_block.bn.eps  # type: ignore
        )
        weights.append(Network.tensor_to_leela_weights(bias))
        weights.append(
            Network.tensor_to_leela_weights(conv_block.bn.running_mean)  # type: ignore
        )
        weights.append(
            Network.tensor_to_leela_weights(conv_block.bn.running_var)  # type: ignore
        )
        return "".join(weights)

    @staticmethod
    def leela_weights_to_conv_block(f: TextIOWrapper, conv_name: str, conv_block: ConvBlock, correct_beta=False) -> dict:
        state_dict = {}

        weight = Network.leela_weights_to_tensor(f.readline()) 
        beta = Network.leela_weights_to_tensor(f.readline())
        running_mean = Network.leela_weights_to_tensor(f.readline())
        running_var = Network.leela_weights_to_tensor(f.readline())
        # beta = beta / torch.sqrt(running_var - conv_block.bn.eps)

        # Subtract the biases from the means
        running_mean -= beta
        # Set the biases to zero
        beta.zero_()
        
        state_dict[conv_name + ".conv.weight"] = weight.reshape(conv_block.conv.weight.shape)
        state_dict[conv_name + ".beta"] = beta.reshape(conv_block.beta.shape)
        state_dict[conv_name + ".bn.running_var"] = running_var.reshape(conv_block.bn.running_var.shape)
        state_dict[conv_name + ".bn.running_mean"] = running_mean.reshape(conv_block.bn.running_mean.shape)

        return state_dict
    
    @staticmethod
    def leela_weights_to_linear_layer(f: TextIOWrapper, linear_name: str, linear: nn.Linear):
        state_dict = {}

        weight = Network.leela_weights_to_tensor(f.readline())
        bias = Network.leela_weights_to_tensor(f.readline())

        print("linear_name", linear_name, "weight", weight.shape, weight)

        state_dict[linear_name + ".weight"] = weight.reshape(linear.weight.shape)
        state_dict[linear_name + ".bias"] = bias.reshape(linear.bias.shape)
        return state_dict

    @staticmethod
    def tensor_to_leela_weights(t: torch.Tensor) -> str:
        return " ".join([str(w) for w in t.detach().numpy().ravel()]) + "\n"

    @staticmethod
    def leela_weights_to_tensor(s: str) -> torch.Tensor:
        return torch.tensor([float(w) for w in s.split()])
