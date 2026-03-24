"""
model_builder.py (training_service)
Builds a nn.Sequential from a list of layer config tuples/dicts.
No Gradio. Receives layer_configs as a parameter.
"""

import torch.nn as nn
from layers import layer_map, build_layer_config_from_dict


def build_model(layer_configs: list) -> nn.Sequential:
    """
    Build a nn.Sequential model from a list of layer config tuples or dicts.

    Each element should be:
        (desc, layer_type, in_dim, out_dim, kernel, padding, stride, bias)
    OR a dict with those same keys (sent from the client as JSON).
    """
    if layer_configs and isinstance(layer_configs[0], dict):
        layer_configs = [build_layer_config_from_dict(d) for d in layer_configs]

    layers = []
    for config in layer_configs:
        _, layer_type, in_dim, out_dim, kernel, padding, stride, bias = config

        if layer_type == "Conv2d":
            layers.append(nn.Conv2d(
                int(in_dim), int(out_dim),
                kernel_size=kernel,
                padding=padding,
                stride=stride,
                bias=bool(bias)
            ))
        elif layer_type == "Linear":
            layers.append(nn.Linear(int(in_dim), int(out_dim)))
        elif layer_type == "GELU":
            layers.append(nn.GELU())
        elif layer_type == "LeakyReLU":
            slope = in_dim if in_dim is not None else 0.01
            layers.append(nn.LeakyReLU(negative_slope=float(slope)))
        elif layer_type == "ELU":
            alpha = in_dim if in_dim is not None else 1.0
            layers.append(nn.ELU(alpha=float(alpha)))
        elif layer_type == "Dropout":
            p = in_dim if in_dim is not None else 0.5
            layers.append(nn.Dropout(float(p)))
        elif layer_type == "Softmax":
            layers.append(nn.Softmax(dim=1))
        elif layer_type == "MaxPool2d":
            layers.append(nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding))
        elif layer_type == "AvgPool2d":
            layers.append(nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding))
        else:
            # ReLU, Tanh, Sigmoid, Flatten
            layers.append(layer_map[layer_type]())

    return nn.Sequential(*layers)
