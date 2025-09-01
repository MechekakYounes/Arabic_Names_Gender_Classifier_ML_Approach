import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'package:universal_html/html.dart' as html;

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Gender Predictor File',
      home: FileUploadScreen(),
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
   try {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['xlsx'],
    );

    if (result != null) {
      Uint8List? fileBytes = result.files.first.bytes; 
      String fileName = result.files.first.name;

      if (fileBytes == null) {
        setState(() {
          _status = "Failed to read file bytes.";
        });
        return;
      }

      var request = http.MultipartRequest(
        "POST",
        Uri.parse("http://127.0.0.1:8000/prediction"),
      );

    
      request.files.add(
        http.MultipartFile.fromBytes("file", fileBytes, filename: fileName),
      );

      var response = await request.send();

      if (response.statusCode == 200) {
        Uint8List bytes = await response.stream.toBytes();

        // Trigger browser download (Web only)
        final blob = html.Blob([bytes]);
        final url = html.Url.createObjectUrlFromBlob(blob);
        final anchor = html.AnchorElement(href: url)
          ..setAttribute("download", "result.xlsx")
          ..click();
        html.Url.revokeObjectUrl(url);

        setState(() {
          _status = "File processed! Download started.";
        });
      } else {
        setState(() {
          _status = "Error: ${response.statusCode}";
        });
      }
    } else {
      setState(() {
        _status = "No file selected";
      });
    } 

   }catch (e) {
     if(e is SocketException) {
      setState(() {
        _status = "Could not connect to server. Please ensure the backend is running.";
      });
      return;
     }
     if(e is http.ClientException) {
      setState(() {
        _status = "Network error occurred. Please check your connection.";
      });
      return;
     }
     if (e is FormatException) {
      setState(() {
        _status = "Response format error. Please try again.";
      });
      return;
     }
     
    setState(() {
      _status = "An error occurred: $e";
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
