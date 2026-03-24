"""
validation.py (training_service)
Same logic as the original but with Gradio removed.
Raises ValueError / returns (bool, msg, idx) instead of gr.Warning/gr.Info.
"""

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_model_forward_pass(model, data_type, image_size=28, num_features=None, num_channels=3):
    """
    Run a dummy forward pass to check for shape mismatches.
    Returns: (is_valid: bool, error_msg: str|None, bad_layer_idx: int|None)
    """
    try:
        if data_type == "tabular":
            assert num_features is not None, "Missing number of input features for tabular data"
            dummy_input = torch.randn(1, num_features).to(device)
        else:
            dummy_input = torch.randn(1, num_channels, image_size, image_size).to(device)

        x = dummy_input
        for idx, layer in enumerate(model):
            try:
                x = layer(x)
            except Exception as e:
                return False, f"Shape mismatch at layer {idx}: {layer.__class__.__name__} — {str(e)}", idx
        return True, None, None
    except Exception as e:
        return False, f"Unexpected validation error: {str(e)}", None
