
import numpy as np
import torch
import torch.nn as nn

def load_darknet_weights(model, weight_path, num_conv_layers=20):

    with open(weight_path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=3)
        seen = np.fromfile(f, dtype=np.int64, count=1)
        weights = np.fromfile(f, dtype=np.float32)

    conv_layers = (
        [model.conv1[0], model.conv2[0]]
        + [block[0] for block in model.conv3]
        + [block[0] for block in model.conv4]
        + [block[0] for block in model.conv5]
        + [block[0] for block in model.conv6]
    )

    bn_layers = (
        [model.conv1[1], model.conv2[1]]
        + [block[1] for block in model.conv3]
        + [block[1] for block in model.conv4]
        + [block[1] for block in model.conv5]
        + [block[1] for block in model.conv6]
    )

    ptr = 0

    assert num_conv_layers <= len(conv_layers)

    for i in range(num_conv_layers):

        conv = conv_layers[i]
        bn = bn_layers[i]

        num_bn_bias = bn.bias.numel()
        bn_bias = weights[ptr:ptr + num_bn_bias]
        ptr += num_bn_bias

        num_bn_weight = bn.weight.numel()
        bn_weight = weights[ptr:ptr + num_bn_weight]
        ptr += num_bn_weight

        num_running_mean = bn.running_mean.numel()
        bn_running_mean = weights[ptr:ptr + num_running_mean]
        ptr += num_running_mean

        num_running_var = bn.running_var.numel()
        bn_running_var = weights[ptr:ptr + num_running_var]
        ptr += num_running_var

        num_conv_weight = conv.weight.numel()
        conv_weight = weights[ptr:ptr + num_conv_weight]
        ptr += num_conv_weight

        with torch.no_grad():

            bn.bias.copy_(
                torch.from_numpy(
                    bn_bias.copy()
                ).view_as(bn.bias).to(bn.bias.device)
            )

            bn.weight.copy_(
                torch.from_numpy(
                    bn_weight.copy()
                ).view_as(bn.weight).to(bn.weight.device)
            )

            bn.running_mean.copy_(
                torch.from_numpy(
                    bn_running_mean.copy()
                ).view_as(bn.running_mean).to(bn.running_mean.device)
            )

            bn.running_var.copy_(
                torch.from_numpy(
                    bn_running_var.copy()
                ).view_as(bn.running_var).to(bn.running_var.device)
            )

            conv.weight.copy_(
                torch.from_numpy(
                    conv_weight.copy()
                ).view_as(conv.weight).to(conv.weight.device)
            )

    return model