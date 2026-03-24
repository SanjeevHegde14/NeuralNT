# 📱 NeuralNT Mobile

NeuralNT Mobile is a highly polished, robust frontend written entirely in Dart/Flutter. It acts as the core interface bridging mobile clients seamlessly with our remote Hugging Face GPU hardware clusters via secure backend streaming.

## ✨ Features

- **Dark/Light Mode**: Premium, fluid aesthetic configurations scaling cleanly across native devices.
- **Real-Time Timers (`Timer.periodic`)**: Asynchronous training timers tracking elapsed logic precisely independent of server roundtrips.
- **Live SSE Gradient Streams**: Consumes chunked backend data asynchronously directly mapping logs to beautifully rendered user graphs.
- **Stateful Navigation (`IndexedStack`)**: Keeps complex multi-screen background training fully persisted while toggling over tabs intuitively.
- **Hardware-Level Abortion (`StreamSubscription`)**: Direct kill-switch mechanisms for stopping intensive training endpoints prematurely through local button bindings.

## 🛠 Compilation Instructions

To build a fresh natively signed Android APK:
```bash
flutter clean
flutter pub get
flutter build apk --release
```
The resulting executable binary payload will successfully stream to `build/app/outputs/flutter-apk/app-release.apk`.

## 🌐 Connected Dependencies 

NeuralNT Mobile expects the `training_service/` FastAPI application to be securely accessible. Modify the internal endpoint variable cleanly in `lib/services/api_service.dart` if testing locally on a `localhost` subnet interface!
