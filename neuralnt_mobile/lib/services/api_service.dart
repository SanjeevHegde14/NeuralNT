import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String _hfUrl = 'https://neuralnt-neuralnt.hf.space';

  static Future<Map<String, dynamic>> predict({
    required String modelFilePath,
    required String imageFilePath,
    required String imageSize,
    required String numChannels,
    required String tabularData,
  }) async {
    try {
      final uri = Uri.parse('$_hfUrl/predict');
      var request = http.MultipartRequest('POST', uri);

      request.fields['image_size'] = imageSize.isNotEmpty ? imageSize : '28';
      request.fields['num_channels'] = numChannels.isNotEmpty ? numChannels : '3';
      request.fields['tabular_data'] = tabularData;

      if (modelFilePath.isNotEmpty) {
        request.files.add(await http.MultipartFile.fromPath('model_file', modelFilePath));
      }
      if (imageFilePath.isNotEmpty) {
        request.files.add(await http.MultipartFile.fromPath('image_file', imageFilePath));
      }

      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);
      
      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        return {"status": "error", "message": "Server returned ${response.statusCode}: ${response.body}"};
      }
    } catch (e) {
      return {"status": "error", "message": "Exception: ${e.toString()}"};
    }
  }

  static Future<http.StreamedResponse?> train({
    required String dataFilePath,
    required String configJson,
  }) async {
    try {
      final uri = Uri.parse('$_hfUrl/train');
      var request = http.MultipartRequest('POST', uri);

      request.fields['config'] = configJson;
      if (dataFilePath.isNotEmpty) {
        request.files.add(await http.MultipartFile.fromPath('dataset', dataFilePath));
      }
      return await request.send();
    } catch (e) {
      return null;
    }
  }
}
