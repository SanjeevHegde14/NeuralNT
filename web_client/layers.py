<<<<<<< HEAD:web_client/layers.py
import gradio as gr

def parse_int_or_tuple(val):
    try:
        return tuple(map(int, str(val).split(','))) if ',' in str(val) else int(val)
    except Exception:
        gr.Warning(f"Invalid numeric input: '{val}'. Please enter an integer or comma-separated pair.")
=======
import torch.nn as nn
from utils import parse_int_or_tuple
>>>>>>> 4c2a4c3fb9f0f55ad3b27e947998d05a20696829:layers.py

# The dictionary containing a map of all layers.
layer_map = {
    "Linear": None, "Conv2d": None, "MaxPool2d": None, "AvgPool2d": None,  
    "Dropout": None, "ReLU": None, "Tanh": None, "Sigmoid": None,
    "Flatten": None, "Softmax": None, "LeakyReLU": None, "GELU": None,
    "ELU": None
}

layer_configs = []

def validate_layer_inputs(layer_type, **kwargs):
    try:
        if layer_type == "Linear":
            in_dim_int = int(kwargs.get("in_dim") or 0)
            out_dim_int = int(kwargs.get("out_dim") or 0)
            if in_dim_int <= 0 or out_dim_int <= 0:
                return False, f"{layer_type} dimensions must be positive integers"
        elif layer_type == "Conv2d":
            in_dim_int = int(kwargs.get("in_dim") or 0)
            out_dim_int = int(kwargs.get("out_dim") or 0)
            if in_dim_int <= 0 or out_dim_int <= 0:
                return False, f"{layer_type} in/out dims must be positive integers"
        return True, None
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def add_layer(
    layer_type, in_dim, out_dim,
    kernel_size=3, padding=1, stride=1, bias=1,
    pool_kernel="2", pool_stride="2", pool_padding="0",
    avgpool_kernel=None, avgpool_stride=None, avgpool_padding=None,
    leaky_slope = "0.01", elu_alpha = "1.0"
):
    is_valid, err_msg = validate_layer_inputs(layer_type=layer_type, in_dim=in_dim, out_dim=out_dim)
    if not is_valid: return err_msg
    
    desc = f"{layer_type}({in_dim}, {out_dim})"
    config = (desc, layer_type, in_dim, out_dim, kernel_size, padding, stride, bias)
    layer_configs.append(config)
    return update_architecture_text()

def update_layer(index, layer_type, in_dim, out_dim, *args, **kwargs):
    index = int(index)
    if 0 <= index < len(layer_configs):
        desc = f"{layer_type}({in_dim}, {out_dim})"
        layer_configs[index] = (desc, layer_type, in_dim, out_dim, None, None, None, None)
    return update_architecture_text()

def delete_layer(index):
    index = int(index)
    if 0 <= index < len(layer_configs):
        layer_configs.pop(index)
    return update_architecture_text()

def reset_layers():
    layer_configs.clear()

def update_architecture_text(highlight_index=None):
    lines = []
    for i, config in enumerate(layer_configs):
        prefix = f"{i}: "
        desc = config[0]
        if i == highlight_index:
            desc = f"⚠️ {desc}"
        lines.append(prefix + desc)
    return "\n".join(lines)
