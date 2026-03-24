"""
app_training_client.py
HTTP client that calls the remote NeuralNT training microservice and streams
log events back as a Gradio-compatible generator.

The remote service URL is configured by the user in the Gradio "Settings" tab.
"""

import base64
import json
import os
import tempfile

import requests
from layers import layer_configs


# ---------------------------------------------------------------------------
# Serialise local layer_configs → list of JSON-serialisable dicts
# ---------------------------------------------------------------------------

def _serialise_layer_configs() -> list:
    """
    Convert the in-memory layer_configs list of tuples to JSON-serialisable dicts.
    Tuple format: (desc, layer_type, in_dim, out_dim, kernel, padding, stride, bias)
    """
    result = []
    for config in layer_configs:
        desc, layer_type, in_dim, out_dim, kernel, padding, stride, bias = config
        result.append({
            "desc":       desc,
            "type":       layer_type,
            "layer_type": layer_type,
            "in_dim":     in_dim,
            "out_dim":    out_dim,
            "kernel":     kernel,
            "padding":    padding,
            "stride":     stride,
            "bias":       bias,
        })
    return result


# ---------------------------------------------------------------------------
# Save base64-encoded content to a temp file; return the path
# ---------------------------------------------------------------------------

def _b64_to_tempfile(b64_str: str, suffix: str) -> str | None:
    if not b64_str:
        return None
    data = base64.b64decode(b64_str)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Main streaming generator — called by Gradio's train button
# ---------------------------------------------------------------------------

def remote_train(
    loss_name: str,
    opt_name: str,
    lr: str,
    batch_size: str,
    image_size: str,
    file_path: str,
    custom_path: str,
    epochs: str,
    num_channels,
    generate_animation: bool,
    target_frames: str,
    frame_rate: str,
):
    """
    Generator that mirrors the signature of the local train_model_with_default_path.
    Yields: (loss_curve_path, animation_path, model_file_path, arch_text, log_text)

    If hf_service_url is empty or unreachable, yields an error log immediately.
    """
    from layers import update_architecture_text  # local import to avoid circular

    url = "https://neuralnt-neuralnt.hf.space"

    # Validate file locally before uploading
    if not file_path or not os.path.exists(file_path):
        yield None, None, None, update_architecture_text(), f"❌ File not found: {file_path}"
        return

    if not custom_path or custom_path.strip() == "":
        custom_path = None

    cfg = {
        "loss":               loss_name,
        "optimizer":          opt_name,
        "lr":                 lr,
        "batch_size":         batch_size,
        "image_size":         image_size,
        "custom_path":        custom_path,
        "epochs":             epochs,
        "num_channels":       int(num_channels),
        "generate_animation": bool(generate_animation),
        "target_frames":      target_frames,
        "frame_rate":         frame_rate,
        "layer_configs":      _serialise_layer_configs(),
    }

    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                f"{url}/train",
                data={"config": json.dumps(cfg)},
                files={"dataset": (os.path.basename(file_path), f)},
                stream=True,
                timeout=3600,  # 1 hour timeout for long training runs
            )
            resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        yield None, None, None, update_architecture_text(), f"❌ Could not connect to training service at {url}. Is it running?"
        return
    except requests.exceptions.HTTPError as e:
        yield None, None, None, update_architecture_text(), f"❌ Training service returned an error: {e}"
        return
    except Exception as e:
        yield None, None, None, update_architecture_text(), f"❌ Unexpected error contacting service: {e}"
        return

    log_lines = []
    import time
    start_time = time.time()

    # ── Parse SSE stream ──────────────────────────────────────────────────
    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue

        etype = event.get("type")
        edata = event.get("data")
        
        elapsed = int(time.time() - start_time)
        mins, secs = divmod(elapsed, 60)
        time_str = f"{mins:02d}:{secs:02d}"

        if etype == "log":
            log_lines.append(f"[⏱ {time_str}] {edata}")
            yield None, None, None, update_architecture_text(), "\n\n".join(log_lines)

        elif etype == "error":
            yield None, None, None, update_architecture_text(), f"❌ [⏱ {time_str}] {edata}"
            return

        elif etype == "result":
            loss_path  = _b64_to_tempfile(edata.get("loss_plot_b64"),  ".png")
            anim_path  = _b64_to_tempfile(edata.get("animation_b64"),  ".mp4")
            
            # Save model to a persistent device directory!
            os.makedirs("trained_models", exist_ok=True)
            model_b64 = edata.get("model_b64")
            if model_b64:
                model_path = os.path.join("trained_models", f"trained_model_{int(time.time())}.pt")
                with open(model_path, "wb") as f:
                    f.write(base64.b64decode(model_b64))
                log_lines.append(f"\n✅ Model permanently saved to device at: {model_path}")
            else:
                model_path = None
                
            final_logs = edata.get("logs", "\n".join(log_lines))
            yield loss_path, anim_path, model_path, update_architecture_text(), final_logs
            return

    # If stream ends without a result event
    yield None, None, None, update_architecture_text(), "⚠️ Training stream ended without a result."

# ---------------------------------------------------------------------------
# Remote Prediction / Inference
# ---------------------------------------------------------------------------

def remote_predict(model_file_path: str, image_file_path: str, tabular_data: str, image_size: str, num_channels, class_names: str = ""):
    """
    Submits a trained model and a new image/data to the remote prediction endpoint.
    """
    url = "https://neuralnt-neuralnt.hf.space"

    if not model_file_path or not os.path.exists(model_file_path):
        return "❌ Please upload a trained model file (.pt)."

    data = {
        "tabular_data": tabular_data or "",
        "image_size": int(image_size) if image_size and image_size.isdigit() else 28,
        "num_channels": int(num_channels) if num_channels else 3
    }

    files = {}
    try:
        with open(model_file_path, "rb") as mf:
            files["model_file"] = (os.path.basename(model_file_path), mf.read())
            
        if image_file_path and os.path.exists(image_file_path):
            with open(image_file_path, "rb") as imgf:
                files["image_file"] = (os.path.basename(image_file_path), imgf.read())

        resp = requests.post(f"{url}/predict", data=data, files=files, timeout=60)
        resp.raise_for_status()
        
        result = resp.json()
        if result.get("status") == "success":
            if "predicted_class" in result:
                probs = result.get('probabilities', [])
                
                pred_idx = result['predicted_class']
                display_name = str(pred_idx)
                names = []
                
                if class_names and class_names.strip():
                    names = [n.strip() for n in class_names.split(",")]
                    if 0 <= pred_idx < len(names):
                        display_name = f"{names[pred_idx]} (Class {pred_idx})"
                
                md = f"### ✅ Prediction Success!\n\n**Predicted:** {display_name}\n\n**Confidence Matrix:**\n\n"
                for i, p in enumerate(probs):
                    c_name = names[i] if names and i < len(names) else f"Class {i}"
                    pct = round(p * 100, 2)
                    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                    md += f"- **{c_name}**: `{bar}` {pct}%\n"
                
                return md
            else:
                return f"✅ Prediction Success!\n\nPrediction Value: {result.get('prediction_value')}"
        else:
            return f"❌ Prediction error: {result.get('message')}"
            
    except requests.exceptions.ConnectionError:
        return f"❌ Could not connect to prediction service at {url}. Is it running?"
    except Exception as e:
        return f"❌ Unexpected error contacting prediction service: {e}"
