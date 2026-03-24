"""
training.py (training_service)
Core training loop — no Gradio. Yields dicts for SSE streaming:
  {"type": "log",    "data": "Epoch 1/10 — Loss: 0.1234"}
  {"type": "result", "data": {"loss_plot": <base64>, "animation": <base64>|None, "model": <base64>}}
  {"type": "error",  "data": "error message"}
"""

import gc
import os
import shutil
import subprocess
import logging
import base64
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model_builder import build_model
from data_loader import load_data
from validation import validate_model_forward_pass
from visualization import get_flat_weights, generate_3d_animation_pca, generate_loss_plot

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _b64(path: str) -> str | None:
    """Read a file and return its base64-encoded string, or None if missing."""
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None


def create_dummy_video(output_path: str):
    command = [
        "ffmpeg", "-f", "lavfi",
        "-i", "color=c=black:s=1280x720:d=5",
        "-c:v", "libx264", "-t", "5",
        "-pix_fmt", "yuv420p", "-y", output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        logger.warning(f"Could not create dummy video: {e}")


def get_device_status() -> dict:
    if not torch.cuda.is_available():
        return {"device": "cpu", "message": "No CUDA device found — running on CPU."}
    try:
        name = torch.cuda.get_device_name(0)
        mem_alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
        mem_res   = torch.cuda.memory_reserved(0) / (1024 ** 2)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        return {
            "device": "cuda",
            "name": name,
            "memory_allocated_mb": round(mem_alloc, 2),
            "memory_reserved_mb":  round(mem_res, 2),
            "memory_total_mb":     round(total_mem, 2),
        }
    except Exception as e:
        return {"device": "cuda_error", "message": str(e)}


# ---------------------------------------------------------------------------
# Main training generator
# ---------------------------------------------------------------------------

def train_model(
    layer_configs: list,
    loss_name: str,
    opt_name: str,
    lr: str,
    batch_size: str = "32",
    image_size: str = "28",
    file_path: str = None,
    custom_path: str = None,
    epochs: str = "100",
    num_channels: int = 3,
    generate_animation: bool = False,
    target_frames: str = "300",
    frame_rate: str = "10",
):
    """
    Generator that yields SSE-compatible dicts.
    """
    work_dir  = tempfile.mkdtemp()
    anim_path = None
    plot_path = None
    model_path = None

    try:
        # ── 1. Parse scalars ─────────────────────────────────────────────
        try:
            target_frames_int = int(target_frames)
            if target_frames_int <= 0:
                raise ValueError("Target frames must be a positive integer.")
        except (TypeError, ValueError) as e:
            yield {"type": "error", "data": f"❌ {e}"}
            return

        try:
            frame_rate_int = int(frame_rate)
            if frame_rate_int <= 0:
                raise ValueError("Frame rate must be a positive integer.")
        except (TypeError, ValueError) as e:
            yield {"type": "error", "data": f"❌ {e}"}
            return

        try:
            channels = int(num_channels)
            if channels not in [1, 3]:
                raise ValueError("Channels must be 1 or 3.")
        except (TypeError, ValueError) as e:
            yield {"type": "error", "data": f"❌ {e}"}
            return

        try:
            max_epochs = int(epochs)
            if max_epochs <= 0:
                raise ValueError("Epochs must be a positive integer.")
        except (TypeError, ValueError) as e:
            yield {"type": "error", "data": f"❌ {e}"}
            return

        try:
            batch_size_int = int(batch_size)
            if batch_size_int <= 0:
                raise ValueError("Batch size must be a positive integer.")
        except (TypeError, ValueError) as e:
            yield {"type": "error", "data": f"❌ {e}"}
            return

        try:
            image_size_int = int(image_size)
            if image_size_int <= 0:
                raise ValueError("Image size must be a positive integer.")
        except (TypeError, ValueError) as e:
            yield {"type": "error", "data": f"❌ {e}"}
            return

        try:
            lr_float = float(lr)
        except (TypeError, ValueError):
            yield {"type": "error", "data": "❌ Learning rate must be a valid number."}
            return

        # ── 2. Validate inputs ───────────────────────────────────────────
        if not layer_configs:
            yield {"type": "error", "data": "❌ No model configured! Please add at least one trainable layer."}
            return

        if not file_path or not os.path.exists(file_path):
            yield {"type": "error", "data": f"❌ Dataset file not found: {file_path}"}
            return

        if not (file_path.endswith('.csv') or file_path.endswith('.zip')):
            yield {"type": "error", "data": "❌ Invalid file type. Please provide a .csv or .zip file."}
            return

        # ── 3. Build model ───────────────────────────────────────────────
        import pandas as pd
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        model = build_model(layer_configs).to(device)

        # ── 4. Validate forward pass ─────────────────────────────────────
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            if 'y' not in df.columns:
                yield {"type": "error", "data": "❌ CSV missing 'y' column."}
                return
            num_features = df.shape[1] - 1
            is_valid, err_msg, bad_layer_idx = validate_model_forward_pass(
                model, "tabular", num_features=num_features)
        else:
            is_valid, err_msg, bad_layer_idx = validate_model_forward_pass(
                model, "image", image_size=image_size_int, num_channels=channels)

        if not is_valid:
            yield {"type": "error", "data": f"❌ {err_msg} (at layer index {bad_layer_idx})"}
            return

        # ── 5. Check trainable params ────────────────────────────────────
        if not any(p.requires_grad for p in model.parameters()):
            yield {"type": "error", "data": "⚠️ Model has no trainable parameters. Add a Linear or Conv2d layer."}
            return

        # ── 6. Load data + optimizer + loss ──────────────────────────────
        loss_fn   = nn.MSELoss() if loss_name == 'MSELoss' else nn.CrossEntropyLoss()
        data      = load_data(file_path, custom_path, batch_size=batch_size_int,
                              image_size=image_size_int, num_channels=channels, loss_fn=loss_fn)
        optimizer = (optim.SGD(model.parameters(), lr=lr_float)
                     if opt_name == 'SGD'
                     else optim.Adam(model.parameters(), lr=lr_float))

        loss_history = []
        weight_snapshots = []
        status_lines = []

        # ── 7. Training loop ─────────────────────────────────────────────
        if data["type"] == "tabular":
            X_train, y_train = data["train"]
            X_train = torch.tensor(X_train, dtype=torch.float32)
            if isinstance(loss_fn, nn.MSELoss):
                y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            else:
                y_train = torch.tensor(y_train, dtype=torch.long).view(-1)

            train_loader = DataLoader(TensorDataset(X_train, y_train),
                                      batch_size=batch_size_int, shuffle=True)

            total_batches = len(train_loader) * max_epochs
            snapshot_interval = max(1, total_batches // target_frames_int)
            global_step = 0

            for epoch in range(1, max_epochs + 1):
                epoch_loss, num_batches = 0.0, 0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    if isinstance(loss_fn, nn.MSELoss):
                        y_input = y_batch.float().view(-1, 1)
                    else:
                        y_input = y_batch.long().view(-1)
                    out  = model(X_batch)
                    loss = loss_fn(out, y_input)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    if generate_animation and (global_step % snapshot_interval == 0):
                        weight_snapshots.append(get_flat_weights(model).cpu().numpy())
                    global_step += 1

                avg_loss = epoch_loss / num_batches
                loss_history.append(avg_loss)
                line = f"Epoch {epoch}/{max_epochs} — Loss: {avg_loss:.4f}"
                status_lines.append(line)
                yield {"type": "log", "data": line}

        else:  # image
            train_loader = data["train"]
            
            total_batches = len(train_loader) * max_epochs
            snapshot_interval = max(1, total_batches // target_frames_int)
            global_step = 0

            for epoch in range(1, max_epochs + 1):
                epoch_loss, num_batches = 0.0, 0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    if isinstance(loss_fn, nn.MSELoss):
                        num_classes = len(torch.unique(y_batch))
                        y_input = torch.nn.functional.one_hot(y_batch, num_classes=num_classes).float().to(device)
                    else:
                        y_input = y_batch
                    out = model(X_batch)
                    if isinstance(loss_fn, nn.MSELoss):
                        out = torch.softmax(out, dim=1)
                    loss = loss_fn(out, y_input)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    if generate_animation and (global_step % snapshot_interval == 0):
                        weight_snapshots.append(get_flat_weights(model).cpu().numpy())
                    global_step += 1

                avg_loss = epoch_loss / num_batches
                loss_history.append(avg_loss)
                line = f"Epoch {epoch}/{max_epochs} — Loss: {avg_loss:.4f}"
                status_lines.append(line)
                yield {"type": "log", "data": line}

            # Clean up extracted ZIP
            if data.get("path"):
                try:
                    shutil.rmtree(data["path"])
                except Exception as e:
                    logger.warning(f"Could not delete extracted folder: {e}")

        # ── 8. Save artefacts ─────────────────────────────────────────────
        plot_path  = os.path.join(work_dir, "loss_plot.png")
        anim_path  = os.path.join(work_dir, "animation.mp4")
        model_path = os.path.join(work_dir, "trained_model.pt")

        generate_loss_plot(loss_history, plot_path)

        if generate_animation:
            generate_3d_animation_pca(
                np.array(weight_snapshots), loss_history, anim_path,
                target_frames=target_frames_int, frame_rate=frame_rate_int
            )
        else:
            create_dummy_video(anim_path)

        torch.save(model, model_path)

        # ── 9. Encode + yield result ──────────────────────────────────────
        yield {
            "type": "result",
            "data": {
                "loss_plot_b64":  _b64(plot_path),
                "animation_b64":  _b64(anim_path),
                "model_b64":      _b64(model_path),
                "model_filename": "trained_model.pt",
                "logs":           "\n".join(status_lines),
            }
        }

    except Exception as e:
        logger.exception("Unexpected training error")
        yield {"type": "error", "data": f"❌ Unexpected error: {str(e)}"}

    finally:
        try:
            del model
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # clean up work dir
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass
