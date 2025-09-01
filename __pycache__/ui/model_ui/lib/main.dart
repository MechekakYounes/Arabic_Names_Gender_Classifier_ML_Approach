import 'dart:typed_data';
import 'package:flutter/foundation.dart'; // for kIsWeb
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'package:universal_html/html.dart' as html; // only used on web


void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: FileUploadScreen(), // starting screen
    );
  }
}

class FileUploadScreen extends StatefulWidget {
  @override
  _FileUploadScreenState createState() => _FileUploadScreenState();
}

class _FileUploadScreenState extends State<FileUploadScreen> {
  String _status = "No file selected";

  Future<void> uploadFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['xlsx'],
    );

    if (result != null) {
      var request = http.MultipartRequest(
        "POST",
        Uri.parse("http://127.0.0.1:8000/prediction"),
      );

      if (kIsWeb) {
        // 🔹 Web: use bytes
        request.files.add(
          http.MultipartFile.fromBytes(
            "file",
            result.files.single.bytes!,
            filename: result.files.single.name,
          ),
        );
      } else {
        // 🔹 Mobile/Desktop: use file path
        request.files.add(
          await http.MultipartFile.fromPath(
            "file",
            result.files.single.path!,
          ),
        );
      }

      var response = await request.send();

      if (response.statusCode == 200) {
        Uint8List bytes = await response.stream.toBytes();

        if (kIsWeb) {
          // 🔹 Trigger browser download on web
          final blob = html.Blob([bytes]);
          final url = html.Url.createObjectUrlFromBlob(blob);
          final anchor = html.AnchorElement(href: url)
            ..setAttribute("download", "result.xlsx")
            ..click();
          html.Url.revokeObjectUrl(url);
        } else {
          // 🔹 Save to downloads folder (optional for mobile/desktop)
          // TODO: implement saving with path_provider if needed
        }

        setState(() {
          _status = "✅ File processed! Download started.";
        });
      } else {
        setState(() {
          _status = "❌ Error: ${response.statusCode}";
        });
      }
    } else {
      setState(() {
        _status = "No file selected";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Gender Predictor (Excel)")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: uploadFile,
              child: Text("Upload Excel File"),
            ),
            SizedBox(height: 20),
            Text(_status, textAlign: TextAlign.center),
          ],
        ),
      ),
    );
  }
}
