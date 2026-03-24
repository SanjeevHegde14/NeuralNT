# 🧠 NeuralNT

NeuralNT is a high-performance cross-platform system that completely decouples Machine Learning configuration from processing hardware. By leveraging a Flutter mobile application and a cloud-based Python deployment, it allows anyone to build, train, configure, and monitor deep neural networks natively on their phone—offloading all intensive computing to a remote GPU environment (e.g., Hugging Face Spaces).

## 🚀 Key Features

*   **Cloud GPU Training via Mobile**: You don't need a massive rig or local CUDA environment to train PyTorch architectures. Simply interact natively with the Flutter Android/iOS app, and your tasks are streamed via SSE networks directly to the backend GPU payload.
*   **Fully Asynchronous UI (`IndexedStack`)**: Toggle between Training and Predict tabs seamlessly. Your live server connections and real-time training progress bars remain strictly active processing in the background.
*   **Live Log Streaming & Visual Timers**: View exact batch logs, elapsed model-fitting timestamps in real-time (`Timer.periodic`), and live gradient descent outputs cleanly compiled by the backend microservice.
*   **Native Inference Histories**: All previously predicted outcomes and compiled `.pt` Base64 binaries are explicitly stored locally and dynamically cached via `shared_preferences` for instantaneous loading.

---

## 📂 Architecture

NeuralNT is divided into three standalone environments inside this monorepo:

### 1. `neuralnt_mobile/` ✨ (Core)
The highly-polished, Dark/Light Mode compatible native Flutter application.
*   **Framework**: Dart / Flutter
*   **UI Features**: Custom premium NeuralNT AI app icon (`flutter_launcher_icons`), dynamic memory routing, and hardware cancellation hooks via `StreamSubscription()`.
*   **How to Build**:
    ```bash
    cd neuralnt_mobile
    flutter clean
    flutter pub get
    flutter build apk --release
    ```
    This outputs the signed Android APK directly to `build/app/outputs/flutter-apk/app-release.apk`.

### 2. `training_service/` ☁️ (Cloud Microservice)
The blazing fast FastAPI payload explicitly configured to deploy instantly as a Docker instance.
*   **Framework**: Python, FastAPI, PyTorch
*   **Capabilities**: Parses custom user network layers, synthesizes dynamic `torch.nn.Sequential` load graphs, dispatches SSE chunking streams, and encodes compiled models back into raw `Base64` files for instant mobile extraction.
*   **How to Deploy**:
    Navigate to your Hugging Face space, select the "Docker" framework template, and upload the entire raw contents of this folder. The native `Dockerfile` explicitly invokes `uvicorn`.

### 3. `web_client/` 💻 (Browser Testing UI)
A native web testing environment utilizing Gradio components for manipulating training hyperparameter configurations cleanly across generic Desktop browsers.
*   **Framework**: Python, Gradio
*   **Execution**:
    ```bash
    cd web_client
    python app.py
    ```

---

## 🛠 Integration Pipeline

1. Deploy the `training_service/` folder securely as a standard Docker cluster natively inside Hugging Face Spaces.
2. The remote backend instances automatically spin up REST API endpoints at `/health`, `/train`, and `/predict`.
3. Build the `neuralnt_mobile/` Flutter payload entirely across local device clusters.
4. Upload custom `.zip` or `.csv` dataset schemas natively through the mobile front-end to trigger, compile, tune, and query deeply sophisticated Neural Network outcomes seamlessly!

<p align="center"><i>Beautiful, scalable, robust deep-learning architectures deployed explicitly onto native user hardware.</i></p>

---

## ⚙️ Prerequisites

Before you begin, please ensure you have the following installed on your machine:
- **Flutter SDK** (v3.19+ recommended) for `neuralnt_mobile/` compilation and testing.
- **Python 3.11+** with `pip` for local `training_service/` API execution.
- **Docker** (Optional, recommended for mimicking the production Hugging Face environment).
- A valid **Hugging Face** account to orchestrate the backend GPU endpoints via Spaces.

---

## 🏁 Quickstart Guide

### 1. Launching the Backend Service
1. Navigate to the backend directory:
   ```bash
   cd training_service/
   ```
2. Install the necessary Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Boot up the local FastAPI hardware stream:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 7860 --reload
   ```

### 2. Bootstrapping the Mobile Client
1. Navigate to the Flutter core library:
   ```bash
   cd neuralnt_mobile/
   ```
2. Fetch the required Dart configurations:
   ```bash
   flutter pub get
   ```
3. Test locally on an emulator or a tethered hardware device:
   ```bash
   flutter run
   ```

---

## 📜 License

This repository is licensed under the MIT License. See the native configurations for detailed distribution rights. Open-source deep-learning architectures deployed elegantly across native UI systems.
