import 'dart:convert';
import 'dart:io';
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:path_provider/path_provider.dart';
import '../services/api_service.dart';
import '../theme/theme_provider.dart';

class TrainScreen extends StatefulWidget {
  const TrainScreen({super.key});

  @override
  State<TrainScreen> createState() => _TrainScreenState();
}

class _TrainScreenState extends State<TrainScreen> {
  final TextEditingController _epochsController = TextEditingController(text: '10');
  final TextEditingController _lrController = TextEditingController(text: '0.01');
  
  String _datasetPath = '';
  List<String> _liveLogs = [];
  bool _isTraining = false;
  
  int _elapsedSeconds = 0;
  Timer? _timer;
  StreamSubscription? _streamSub;

  @override
  void dispose() {
    _timer?.cancel();
    _streamSub?.cancel();
    super.dispose();
  }

  void _startTimer() {
    _elapsedSeconds = 0;
    _timer?.cancel();
    _timer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if(mounted && _isTraining) {
        setState(() => _elapsedSeconds++);
      }
    });
  }

  void _stopTraining() {
    _streamSub?.cancel();
    _timer?.cancel();
    setState(() {
      _isTraining = false;
      _liveLogs.insert(0, "🛑 Local Training Stream interrupted by user!");
    });
  }

  Future<void> _pickDataset() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles();
    if (result != null && result.files.single.path != null) {
      setState(() => _datasetPath = result.files.single.path!);
    }
  }

  Future<void> _startTraining() async {
    if (_datasetPath.isEmpty) {
      if(mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Please select a Dataset (.csv/.zip).')));
      }
      return;
    }
    
    setState(() {
      _isTraining = true;
      _liveLogs = ["🚀 Training started... Connecting to Cloud GPU..."];
    });

    final configStr = '''{
      "loss": "CrossEntropyLoss",
      "optimizer": "Adam",
      "lr": "${_lrController.text}",
      "batch_size": "32",
      "image_size": "32",
      "epochs": "${_epochsController.text}",
      "num_channels": 3,
      "layer_configs": [
        {"desc": "Conv2d(3, 16)", "layer_type": "Conv2d", "in_dim": 3, "out_dim": 16, "kernel": 3, "padding": 1, "stride": 1, "bias": null},
        {"desc": "ReLU()", "layer_type": "ReLU", "in_dim": null, "out_dim": null, "kernel": null, "padding": null, "stride": null, "bias": null},
        {"desc": "MaxPool2d(2)", "layer_type": "MaxPool2d", "in_dim": null, "out_dim": null, "kernel": 2, "padding": 0, "stride": 2, "bias": null},
        {"desc": "Conv2d(16, 32)", "layer_type": "Conv2d", "in_dim": 16, "out_dim": 32, "kernel": 3, "padding": 1, "stride": 1, "bias": null},
        {"desc": "ReLU()", "layer_type": "ReLU", "in_dim": null, "out_dim": null, "kernel": null, "padding": null, "stride": null, "bias": null},
        {"desc": "MaxPool2d(2)", "layer_type": "MaxPool2d", "in_dim": null, "out_dim": null, "kernel": 2, "padding": 0, "stride": 2, "bias": null},
        {"desc": "Flatten()", "layer_type": "Flatten", "in_dim": null, "out_dim": null, "kernel": null, "padding": null, "stride": null, "bias": null},
        {"desc": "Linear", "layer_type": "Linear", "in_dim": 2048, "out_dim": 128, "kernel": null, "padding": null, "stride": null, "bias": null},
        {"desc": "ReLU()", "layer_type": "ReLU", "in_dim": null, "out_dim": null, "kernel": null, "padding": null, "stride": null, "bias": null},
        {"desc": "Linear", "layer_type": "Linear", "in_dim": 128, "out_dim": 10, "kernel": null, "padding": null, "stride": null, "bias": null}
      ]
    }''';

    _startTimer();

    final res = await ApiService.train(
      dataFilePath: _datasetPath,
      configJson: configStr,
    );

    if (res != null && res.statusCode == 200) {
      _streamSub = res.stream.transform(utf8.decoder).transform(const LineSplitter()).listen((line) async {
        if (line.isNotEmpty) {
          try {
            final parsed = jsonDecode(line);
            if (parsed['type'] == 'log') {
              setState(() => _liveLogs.insert(0, parsed['data']));
            } else if (parsed['type'] == 'result') {
              setState(() => _liveLogs.insert(0, "✅ Training Complete! Processing files..."));
              
              // Base64 Decode the .pt model file and save to local path
              final modelB64 = parsed['data']['model_b64'];
              if (modelB64 != null) {
                try {
                  final bytes = base64Decode(modelB64);
                  final dir = await getApplicationDocumentsDirectory();
                  final newFile = File('\${dir.path}/trained_model_\${DateTime.now().millisecondsSinceEpoch}.pt');
                  await newFile.writeAsBytes(bytes);
                  
                  final prefs = await SharedPreferences.getInstance();
                  List<String> history = prefs.getStringList('model_history') ?? [];
                  history.insert(0, newFile.path);
                  if (history.length > 5) history = history.sublist(0, 5);
                  await prefs.setStringList('model_history', history);
                  
                  setState(() => _liveLogs.insert(0, "💾 Model Auto-Saved to device! Available instantly in Predict tab."));
                } catch(e) {
                  setState(() => _liveLogs.insert(0, "⚠️ Failed to save model: \$e"));
                }
              }
              _timer?.cancel();
              setState(() => _isTraining = false);
            } else if (parsed['type'] == 'error') {
              _timer?.cancel();
              setState(() {
                _liveLogs.insert(0, "❌ Error: \${parsed['data']}");
                _isTraining = false;
              });
            }
          } catch (e) {
            // Ignored, streaming fragments
          }
        }
      }, onDone: () {
        _timer?.cancel();
        if(_isTraining) {
            setState(() {
                _isTraining = false;
                _liveLogs.insert(0, "✅ Stream Closed.");
            });
        }
      }, onError: (err) {
        _timer?.cancel();
        setState(() {
          _isTraining = false;
          _liveLogs.insert(0, "❌ Stream Error: \$err");
        });
      });
    } else {
      _timer?.cancel();
      setState(() {
        _isTraining = false;
        _liveLogs.insert(0, "❌ Training connection failed. Status: \${res?.statusCode}");
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context, listen: false);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Train Model ☁️'),
        actions: [
          IconButton(
            icon: Icon(themeProvider.isDarkMode ? Icons.light_mode : Icons.dark_mode),
            onPressed: () => themeProvider.toggleTheme(),
            tooltip: 'Toggle Theme',
          )
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    const Text('Upload Dataset & Start Live Training', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                    const SizedBox(height: 12),
                    ElevatedButton.icon(
                      icon: const Icon(Icons.folder),
                      label: Text(_datasetPath.isEmpty ? 'Select Dataset (.csv/.zip)' : 'Dataset Selected ✓'),
                      onPressed: _isTraining ? null : _pickDataset,
                      style: ElevatedButton.styleFrom(minimumSize: const Size.fromHeight(50)),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(child: TextField(controller: _epochsController, decoration: const InputDecoration(labelText: 'Epochs'))),
                const SizedBox(width: 8),
                Expanded(child: TextField(controller: _lrController, decoration: const InputDecoration(labelText: 'Learning Rate'))),
              ],
            ),
            const SizedBox(height: 24),
            Row(
              children: [
                if (_isTraining) Expanded(
                  child: OutlinedButton.icon(
                    icon: const Icon(Icons.stop, color: Colors.red),
                    label: Text('⏹ Stop [${(_elapsedSeconds ~/ 60).toString().padLeft(2, "0")}:${(_elapsedSeconds % 60).toString().padLeft(2, "0")}]', style: const TextStyle(color: Colors.red)),
                    onPressed: _stopTraining,
                    style: OutlinedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 16)),
                  ),
                ),
                if (_isTraining) const SizedBox(width: 8),
                Expanded(
                  flex: _isTraining ? 1 : 2,
                  child: ElevatedButton(
                    onPressed: _isTraining ? null : _startTraining,
                    style: ElevatedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 16)),
                    child: _isTraining
                        ? const SizedBox(height: 20, width: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                        : const Text('🚀 Start Cloud Training', style: TextStyle(fontSize: 16)),
                  ),
                ),
              ]
            ),
            const SizedBox(height: 24),
            const Text('Live Training Logs', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Container(
              height: 200,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(color: Theme.of(context).cardTheme.color, borderRadius: BorderRadius.circular(8)),
              child: ListView.builder(
                itemCount: _liveLogs.length,
                itemBuilder: (context, index) {
                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 4.0),
                    child: Text(_liveLogs[index], style: const TextStyle(fontFamily: 'monospace', fontSize: 13)),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
