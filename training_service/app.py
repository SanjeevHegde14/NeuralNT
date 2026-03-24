"""
app.py (training_service)
FastAPI microservice — the HuggingFace Space entry point.

Endpoints:
  GET  /health            → {"status": "ok", "device": ...}
  POST /train             → SSE stream of training log/result events
"""

import json
import logging
import os
import tempfile

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

import torch
import torchvision.transforms as transforms
from PIL import Image
import io

from training import train_model, get_device_status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NeuralNT Training Microservice",
    description="GPU-accelerated neural network training backend for NeuralNT. "
                "Deploy this on a HuggingFace Space with a GPU runtime.",
    version="1.0.0",
)

# Allow cross-origin requests from the Gradio front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "device_info": get_device_status()}


# ---------------------------------------------------------------------------
# /train  — SSE streaming endpoint
# ---------------------------------------------------------------------------

@app.post("/train")
async def train_endpoint(
    config: str = Form(...,
        description="JSON string containing all training hyperparameters and layer_configs."),
    dataset: UploadFile = File(...,
        description="The dataset file (.csv or .zip)."),
):
    """
    Accepts multipart form data:
      - config  : JSON string with keys:
            loss, optimizer, lr, batch_size, image_size, epochs,
            num_channels, generate_animation, target_frames, frame_rate,
            layer_configs   ← list of layer dicts from the Gradio app
      - dataset : the CSV or ZIP file

    Returns a Server-Sent Events (SSE) stream. Each event is a JSON object:
      {"type": "log",    "data": "Epoch 1/10 — Loss: 0.1234"}
      {"type": "result", "data": {"loss_plot_b64": ..., "animation_b64": ..., "model_b64": ...}}
      {"type": "error",  "data": "error message"}
    """
    # Parse config
    try:
        cfg = json.loads(config)
    except Exception as e:
        async def error_stream():
            yield {"data": json.dumps({"type": "error", "data": f"Invalid config JSON: {e}"})}
        return EventSourceResponse(error_stream())

    # Save uploaded dataset to a temp file
    suffix = os.path.splitext(dataset.filename)[1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await dataset.read())
        tmp_path = tmp.name

    layer_configs = cfg.get("layer_configs", [])

    async def event_stream():
        try:
            for event in train_model(
                layer_configs     = layer_configs,
                loss_name         = cfg.get("loss", "CrossEntropyLoss"),
                opt_name          = cfg.get("optimizer", "Adam"),
                lr                = str(cfg.get("lr", "0.01")),
                batch_size        = str(cfg.get("batch_size", "32")),
                image_size        = str(cfg.get("image_size", "28")),
                file_path         = tmp_path,
                custom_path       = cfg.get("custom_path"),
                epochs            = str(cfg.get("epochs", "100")),
                num_channels      = int(cfg.get("num_channels", 3)),
                generate_animation= bool(cfg.get("generate_animation", False)),
                target_frames     = str(cfg.get("target_frames", "300")),
                frame_rate        = str(cfg.get("frame_rate", "10")),
            ):
                yield {"data": json.dumps(event)}
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return EventSourceResponse(event_stream())

# ---------------------------------------------------------------------------
# /predict - Inference endpoint
# ---------------------------------------------------------------------------

@app.post("/predict")
async def predict_endpoint(
    model_file: UploadFile = File(..., description="The trained .pt model file"),
    image_file: UploadFile = File(default=None, description="Image to predict"),
    tabular_data: str = Form(default="", description="Comma-separated values for tabular inference"),
    image_size: int = Form(default=28),
    num_channels: int = Form(default=3)
):
    try:
        model_bytes = await model_file.read()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model architecture and weights
        model = torch.load(io.BytesIO(model_bytes), map_location=device, weights_only=False)
        model.eval()
        
        if image_file and image_file.filename:
            img_bytes = await image_file.read()
            img = Image.open(io.BytesIO(img_bytes))
            if num_channels == 1:
                img = img.convert("L")
            else:
                img = img.convert("RGB")
                
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])
            tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = model(tensor)
                
            if out.shape[1] > 1:
                probs = torch.nn.functional.softmax(out, dim=1).squeeze().tolist()
                probs = probs if isinstance(probs, list) else [probs]
                pred_class = int(torch.argmax(out, dim=1).item())
                return {"status": "success", "predicted_class": pred_class, "probabilities": probs}
            else:
                return {"status": "success", "prediction_value": out.item()}
                
        elif tabular_data:
            vals = [float(x.strip()) for x in tabular_data.split(",") if x.strip()]
            tensor = torch.tensor(vals, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
            
            if out.shape[1] > 1:
                probs = torch.nn.functional.softmax(out, dim=1).squeeze().tolist()
                probs = probs if isinstance(probs, list) else [probs]
                pred_class = int(torch.argmax(out, dim=1).item())
                return {"status": "success", "predicted_class": pred_class, "probabilities": probs}
            else:
                return {"status": "success", "prediction_value": out.item()}
        else:
            return {"status": "error", "message": "Please upload an image_file or provide tabular_data."}

    except Exception as e:
        logger.exception("Prediction failed")
        return {"status": "error", "message": f"Inference error: {str(e)}"}
