---
title: NeuralNT Training Service
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# NeuralNT Training Service — HuggingFace Deployment Guide

This folder contains the **FastAPI training microservice** for NeuralNT.  
It is designed to run on a **HuggingFace Space with a GPU** so that training happens in the cloud, not on your local machine.

---

## 1. Create a HuggingFace Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **New Space**.
2. Choose a name, e.g. `neuralnt-training-service`.
3. **SDK**: Select **Docker**.
4. **Hardware**: Select **T4-small** (free GPU tier) or any GPU tier.

---

## 2. Upload this folder as the Space repository

You can either:
- **Use the HF web editor**: drag and drop all files in this folder.
- **Use Git**:
  ```bash
  git clone https://huggingface.co/spaces/YOUR_USERNAME/neuralnt-training-service
  cp -r training_service/* neuralnt-training-service/
  cd neuralnt-training-service
  git add .
  git commit -m "Add NeuralNT training microservice"
  git push
  ```

---

## 3. Add a Dockerfile

HuggingFace Docker Spaces require a `Dockerfile`. Create one in the root of the Space:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

> **Tip**: If you want PyTorch with CUDA support, use a CUDA base image instead:
> ```dockerfile
> FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
> ```

---

## 4. Get your Space URL

Once the Space builds successfully, your public URL will be:
```
https://YOUR_USERNAME-neuralnt-training-service.hf.space
```

---

## 5. Connect the Gradio app

1. Run the main Gradio app locally: `python app.py`
2. Go to the **Settings** tab.
3. Paste your HuggingFace Space URL.
4. Click **Refresh Remote Device Status** to verify the connection.
5. Now click **🚀 Start Cloud Training** in the **Train** tab — all training will run on the HuggingFace GPU.

---

## Local Testing (no HuggingFace)

To test without deploying to HuggingFace, run the service locally:

```powershell
cd training_service
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7862 --reload
```

In the Gradio Settings tab, set the URL to `http://localhost:7862`.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns service status and GPU info |
| `/train`  | POST  | Accepts `config` (JSON form field) + `dataset` (file upload). Returns SSE stream of training events. |

### SSE Event Format

```json
{"type": "log",    "data": "Epoch 1/10 — Loss: 0.1234"}
{"type": "result", "data": {"loss_plot_b64": "...", "animation_b64": "...", "model_b64": "..."}}
{"type": "error",  "data": "error message"}
```
