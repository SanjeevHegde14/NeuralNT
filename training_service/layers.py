"""
layers.py (training_service)
Stateless layer helpers — no Gradio, no global state.
Layer configs are passed in as plain Python dicts from the client.
"""

import torch.nn as nn
from utils import parse_int_or_tuple


# ---------------------------------------------------------------------------
# Layer registry
# ---------------------------------------------------------------------------
layer_map = {
    "Linear":    lambda in_dim, out_dim: nn.Linear(int(in_dim), int(out_dim)),
    "Conv2d":    lambda in_dim, out_dim: nn.Conv2d(int(in_dim), int(out_dim), kernel_size=3, padding=1),
    "MaxPool2d": lambda *_: nn.MaxPool2d(kernel_size=2),
    "AvgPool2d": lambda *_: nn.AvgPool2d(kernel_size=2),
    "Dropout":   lambda p=0.5, *_: nn.Dropout(float(p)),
    "ReLU":      lambda *_: nn.ReLU(),
    "Tanh":      lambda *_: nn.Tanh(),
    "Sigmoid":   lambda *_: nn.Sigmoid(),
    "Flatten":   lambda *_: nn.Flatten(),
    "Softmax":   lambda *_: nn.Softmax(dim=1),
    "LeakyReLU": lambda slope=0.01, *_: nn.LeakyReLU(negative_slope=float(slope)),
    "GELU":      lambda *_: nn.GELU(),
    "ELU":       lambda alpha=1.0, *_: nn.ELU(alpha=float(alpha)),
}


# ---------------------------------------------------------------------------
# Validation — returns (bool, error_message | None)
# ---------------------------------------------------------------------------
def validate_layer_inputs(layer_type: str, **kwargs):
    try:
        if layer_type == "Linear":
            in_dim_int = int(kwargs.get("in_dim"))
            out_dim_int = int(kwargs.get("out_dim"))
            if in_dim_int <= 0 or out_dim_int <= 0:
                return False, f"{layer_type} dimensions must be positive integers"

        elif layer_type == "Conv2d":
            in_dim_int = int(kwargs.get("in_dim"))
            out_dim_int = int(kwargs.get("out_dim"))
            kernel_dim = parse_int_or_tuple(kwargs.get("kernel_size", 3))
            padding_dim = parse_int_or_tuple(kwargs.get("padding", 1))
            stride = parse_int_or_tuple(kwargs.get("stride", 1))
            for val in [kernel_dim, padding_dim, stride]:
                if isinstance(val, tuple):
                    if any(v < 0 for v in val):
                        return False, f"{layer_type} tuple values must be non-negative"
                else:
                    if val < 0:
                        return False, f"{layer_type} values must be non-negative"
            if in_dim_int <= 0 or out_dim_int <= 0:
                return False, f"{layer_type} in/out dims must be positive integers"

        elif layer_type == "Dropout":
            p = float(kwargs.get("in_dim"))
            if not (0 <= p <= 1):
                return False, "Dropout probability must be between 0 and 1"

        elif layer_type == "MaxPool2d":
            for k in ["pool_kernel", "pool_stride", "pool_padding"]:
                val = parse_int_or_tuple(kwargs.get(k, 2 if "kernel" in k else (2 if "stride" in k else 0)))
                if isinstance(val, tuple):
                    if any(v < 0 for v in val):
                        return False, f"{layer_type} tuple values must be non-negative"
                else:
                    if val < 0:
                        return False, f"{layer_type} values must be non-negative"

        elif layer_type == "AvgPool2d":
            for k in ["avgpool_kernel", "avgpool_stride", "avgpool_padding"]:
                val = parse_int_or_tuple(kwargs.get(k, 2 if "kernel" in k else (2 if "stride" in k else 0)))
                if isinstance(val, tuple):
                    if any(v < 0 for v in val):
                        return False, f"{layer_type} tuple values must be non-negative"
                else:
                    if val < 0:
                        return False, f"{layer_type} values must be non-negative"

        elif layer_type == "LeakyReLU":
            slope = float(kwargs.get("leaky_slope", 0.01))
            if slope < 0:
                return False, "LeakyReLU negative_slope must be >= 0"

        elif layer_type == "ELU":
            alpha = float(kwargs.get("elu_alpha", 1.0))
            if alpha < 0:
                return False, "ELU alpha must be >= 0"

        return True, None
    except Exception as e:
        return False, f"Validation error in {layer_type}: {str(e)}"


# ---------------------------------------------------------------------------
# Build a single layer config tuple from a dict sent by the client.
# Dict shape mirrors what the Gradio front-end already tracks in layer_configs:
#   {"type": str, "in_dim": ..., "out_dim": ..., "kernel": ...,
#    "padding": ..., "stride": ..., "bias": ..., "desc": str}
# Returns: (desc, layer_type, in_dim, out_dim, kernel, padding, stride, bias)
# ---------------------------------------------------------------------------
def build_layer_config_from_dict(d: dict):
    layer_type = d.get("type") or d.get("layer_type")
    in_dim     = d.get("in_dim")
    out_dim    = d.get("out_dim")
    kernel     = d.get("kernel")
    padding    = d.get("padding")
    stride     = d.get("stride")
    bias       = d.get("bias", True)
    desc       = d.get("desc", layer_type)
    return (desc, layer_type, in_dim, out_dim, kernel, padding, stride, bias)
