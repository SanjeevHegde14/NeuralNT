# NeuralNT (Flutter + PyTorch Edition)

Neural Network Training App made to train and test custom AI models without any coding required. Built with a **Flutter** mobile frontend and a **FastAPI/PyTorch** backend.

## 🚀 Setup Instructions

### 1. Prerequisites
- **Python 3.9+**
- **Flutter SDK**: Installed at `C:\src\flutter` (or updated in your PATH).
- **Android Studio**: With Android SDK 34 and 36 installed via SDK Manager.

### 2. Start the Backend (The Engine)
Open a terminal in the project root and run:
```bash
# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn python-multipart pandas numpy scikit-learn matplotlib Pillow py-cpuinfo

# Start the server
python backend_api.py
```
*The server will run on `http://localhost:8000`. Leave this terminal open.*

### 3. Start the Frontend (The App)
Open a **new** terminal tab, navigate to the `frontend` folder, and run:
```bash
cd frontend

# Fix Path if 'flutter' is not recognized
$env:Path += ";C:\src\flutter\bin"

# Sync and Launch
flutter clean
flutter pub get
flutter run --android-skip-build-dependency-validation
```

## 🛠 Features & How to Use
1.  **Build**: Use the side menu to add layers (Conv2d, Linear, etc.). It talks to the backend to verify dimensions.
2.  **Train**: Switch to the Train tab. Upload a `.zip` dataset (like CIFAR-10). Hit **Start Training**. 
    - *Note: Training happens on your PC. The phone shows progress.*
3.  **Test**: Switch to the Test tab. Upload a single image (e.g., a cat) to see the model's prediction and confidence score.

## 📡 Mobile + PC local Wi-Fi (recommended)
This is the stable working mode:
- Backend server on PC (`uvicorn ... --host 0.0.0.0 --port 8000`)
- Flutter app on device connects to `http://<PC_IP>:8000`

### 1) Start backend
```powershell
cd D:\projeks\internship\P2\NNT2\NeuralNT
python -m uvicorn backend_api:app --host 0.0.0.0 --port 8000 --reload
```

### 2) Confirm backend is reachable on PC
```powershell
curl http://127.0.0.1:8000/health
# expected: {"status":"ok" ...}
```

### 3) Configure firewall (Windows)
```powershell
netsh advfirewall firewall add rule name="NeuralNT 8000" dir=in action=allow protocol=tcp localport=8000
```

### 4) Get PC IP
`ipconfig` -> e.g. `192.168.1.42`

### 5) In mobile app, set backend target
- On **Build** tab use fields:
  - Host: `192.168.1.42`
  - Port: `8000`
- Click **Apply & Connect**
- You should see architecture and API readable state.

### 6) Test in mobile browser
`http://192.168.1.42:8000/health` should show `{"status":"ok"}`.

## 📱 APK build and install
From `frontend`:
```powershell
flutter clean
flutter pub get
flutter build apk --release
adb install -r build\app\outputs\flutter-apk\app-release.apk
```

## 🐞 Debug tips
- If mobile says API unavailable:
  - ensure same Wi-Fi
  - ensure backend is running and reachable from PC browser
  - ensure firewall rule is active
  - try `http://<PC_IP>:8000/docs` from mobile browser
- If app is still using `localhost`, use the UI server fields in Build tab.

## ⚠️ Known Fixes (Troubleshooting)
- **SDK Errors**: This project is configured for **compileSdk 36**. If missing, install it via Android Studio > SDK Manager.
- **Java/Kotlin Errors**: The project uses **Java 17**. Ensure your environment supports it.
- **Connection**: On Emulator, the app uses `10.0.2.2` to find your PC. For a real phone, update `baseUrl` in `main.dart` to your PC's IPv4 address.

---
**NeuralNT - Building the future of no-code AI.**
