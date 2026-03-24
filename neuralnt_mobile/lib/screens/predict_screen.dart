import 'dart:io';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/api_service.dart';
import '../theme/theme_provider.dart';

class PredictScreen extends StatefulWidget {
  const PredictScreen({super.key});

  @override
  State<PredictScreen> createState() => _PredictScreenState();
}

class _PredictScreenState extends State<PredictScreen> {
  final TextEditingController _imageSizeController = TextEditingController(text: '32');
  final TextEditingController _channelsController = TextEditingController(text: '3');
  final TextEditingController _tabularController = TextEditingController();
  
  // Default to CIFAR-10 classes to instantly map integer 0-9 cleanly for the user!
  final TextEditingController _classMapController = TextEditingController(
    text: 'airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck'
  );
  
  String _modelFilePath = '';
  String _imageFilePath = '';
  List<String> _modelHistory = [];
  
  bool _isLoading = false;
  Map<String, dynamic>? _predictionResult;
  String _errorMessage = '';

  @override
  void initState() {
    super.initState();
    _loadSavedData();
  }

  Future<void> _loadSavedData() async {
    final prefs = await SharedPreferences.getInstance();
    final history = prefs.getStringList('model_history') ?? [];
    setState(() {
      _modelHistory = history;
      if (_modelHistory.isNotEmpty && _modelFilePath.isEmpty) {
        _modelFilePath = _modelHistory.first;
      }
    });

    final savedClassMap = prefs.getString('class_map');
    if (savedClassMap != null && savedClassMap.isNotEmpty) {
      _classMapController.text = savedClassMap;
    }
  }

  Future<void> _saveModelToHistory(String path) async {
    if (path.isEmpty) return;
    final prefs = await SharedPreferences.getInstance();
    List<String> history = prefs.getStringList('model_history') ?? [];
    if (!history.contains(path)) {
      history.insert(0, path);
      if (history.length > 5) history = history.sublist(0, 5); // Keep up to 5
      await prefs.setStringList('model_history', history);
      setState(() => _modelHistory = history);
    }
  }

  Future<void> _pickModel() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles();
    if (result != null && result.files.single.path != null) {
      final path = result.files.single.path!;
      setState(() => _modelFilePath = path);
      _saveModelToHistory(path);
    }
  }

  Future<void> _pickImage() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(type: FileType.image);
    if (result != null && result.files.single.path != null) {
      setState(() => _imageFilePath = result.files.single.path!);
    }
  }

  Future<void> _runPrediction() async {
    if (_modelFilePath.isEmpty) {
      setState(() {
        _errorMessage = "Please select a Model file.";
        _predictionResult = null;
      });
      return;
    }
    
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('class_map', _classMapController.text);
    
    setState(() {
      _isLoading = true;
      _errorMessage = '';
      _predictionResult = null;
    });

    final res = await ApiService.predict(
      modelFilePath: _modelFilePath,
      imageFilePath: _imageFilePath,
      imageSize: _imageSizeController.text,
      numChannels: _channelsController.text,
      tabularData: _tabularController.text,
    );

    setState(() {
      _isLoading = false;
      if (res['status'] == 'success') {
        _predictionResult = res;
      } else {
        _errorMessage = res['message'] ?? 'Unknown Error';
      }
    });
  }

  Widget _buildPredictionUI() {
    if (_errorMessage.isNotEmpty) {
      return Text('❌ Error: $_errorMessage', style: const TextStyle(color: Colors.redAccent, fontSize: 16));
    }
    if (_predictionResult == null) {
      return const Text('Run prediction to see results.', style: TextStyle(fontSize: 16));
    }

    final res = _predictionResult!;
    if (res.containsKey('predicted_class')) {
      int predIdx = res['predicted_class'];
      String displayClass = "Class $predIdx";

      String classMapRaw = _classMapController.text.trim();
      if (classMapRaw.isNotEmpty) {
        List<String> map = classMapRaw.split(',').map((e) => e.trim()).toList();
        if (predIdx >= 0 && predIdx < map.length) {
          displayClass = "${map[predIdx]} ($displayClass)";
        }
      }

      List<dynamic> probs = res['probabilities'] ?? [];
      
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.check_circle, color: Colors.green),
              const SizedBox(width: 8),
              Expanded(child: Text('Prediction: $displayClass', style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold))),
            ],
          ),
          const SizedBox(height: 16),
          const Text('Probability Confidence Matrix:', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          ...List.generate(probs.length, (index) {
            double prob = (probs[index] as num).toDouble();
            String name = "Class $index";
            if (classMapRaw.isNotEmpty) {
              List<String> map = classMapRaw.split(',').map((e) => e.trim()).toList();
              if (index < map.length) name = map[index];
            }
            return Padding(
              padding: const EdgeInsets.only(bottom: 8.0),
              child: Row(
                children: [
                  SizedBox(width: 80, child: Text(name, style: const TextStyle(fontSize: 13), overflow: TextOverflow.ellipsis)),
                  Expanded(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: LinearProgressIndicator(
                        value: prob,
                        backgroundColor: Colors.grey.withOpacity(0.2),
                        color: prob == 1.0 ? Colors.green : Colors.blueAccent,
                        minHeight: 12,
                      ),
                    ),
                  ),
                  SizedBox(width: 50, child: Text('${(prob * 100).toStringAsFixed(1)}%', textAlign: TextAlign.right, style: const TextStyle(fontSize: 12))),
                ],
              ),
            );
          }),
        ],
      );
    } else {
      return Text("✅ Prediction Value: ${res['prediction_value']}", style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold));
    }
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context, listen: false);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Predict 🔮'),
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
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    const Text('1. Select Trained Model (.pt)', style: TextStyle(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 12),
                    if (_modelHistory.isNotEmpty) ...[
                      DropdownButton<String>(
                        isExpanded: true,
                        value: _modelHistory.contains(_modelFilePath) ? _modelFilePath : null,
                        hint: const Text('Select Model from History'),
                        items: _modelHistory.map((String path) {
                          String filename = path.split(Platform.pathSeparator).last;
                          return DropdownMenuItem<String>(
                            value: path,
                            child: Text(filename, overflow: TextOverflow.ellipsis),
                          );
                        }).toList(),
                        onChanged: (val) {
                          if (val != null) {
                            setState(() {
                              _modelFilePath = val;
                              _predictionResult = null;
                            });
                          }
                        },
                      ),
                      const SizedBox(height: 8),
                    ],
                    OutlinedButton.icon(
                      icon: const Icon(Icons.folder_open),
                      label: Text(_modelFilePath.isEmpty ? 'Browse Device for .pt Model' : 'Selected: ...${_modelFilePath.split(Platform.pathSeparator).last}'),
                      onPressed: _pickModel,
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    const Text('2. Provide New Data', style: TextStyle(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 12),
                    if (_imageFilePath.isNotEmpty) ...[
                      ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: Image.file(
                          File(_imageFilePath),
                          height: 150,
                          fit: BoxFit.cover,
                        ),
                      ),
                      const SizedBox(height: 12),
                    ],
                    OutlinedButton.icon(
                      icon: const Icon(Icons.image),
                      label: Text(_imageFilePath.isEmpty ? 'Upload Image' : 'Change Selected Image 📸'),
                      onPressed: _pickImage,
                      style: OutlinedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 12)),
                    ),
                    const SizedBox(height: 8),
                    const Text('OR', textAlign: TextAlign.center, style: TextStyle(fontWeight: FontWeight.bold)),
                    TextField(
                      controller: _tabularController,
                      decoration: const InputDecoration(labelText: 'Tabular Data (comma separated)'),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _classMapController,
              decoration: const InputDecoration(labelText: 'Class Names Mapping', hintText: 'airplane, automobile, bird...'),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(child: TextField(controller: _imageSizeController, decoration: const InputDecoration(labelText: 'Image Size (px)'))),
                const SizedBox(width: 8),
                Expanded(child: TextField(controller: _channelsController, decoration: const InputDecoration(labelText: 'Channels (1 or 3)'))),
              ],
            ),
            const SizedBox(height: 28),
            ElevatedButton(
              onPressed: _isLoading ? null : _runPrediction,
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                backgroundColor: Theme.of(context).primaryColor,
                foregroundColor: Colors.white,
              ),
              child: _isLoading ? const SizedBox(height:20, width: 20, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2)) : const Text('🔮 Run Native Prediction', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            ),
            const SizedBox(height: 24),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(color: Theme.of(context).cardTheme.color, borderRadius: BorderRadius.circular(12)),
              child: _buildPredictionUI(),
            )
          ],
        ),
      ),
    );
  }
}
