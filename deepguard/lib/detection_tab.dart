import 'package:flutter/material.dart';
import 'package:getwidget/getwidget.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http; // 1. http 패키지 임포트
import 'dart:convert'; // 2. json 및 utf8 임포트
import 'http_client.dart'; // 3. 공유 클라이언트 (세션 유지용) 임포트

class DetectionTab extends StatefulWidget {
  final bool isLoggedIn;
  final String? sharedUrl;

  const DetectionTab({super.key, required this.isLoggedIn, this.sharedUrl});

  @override
  State<DetectionTab> createState() => _DetectionTabState();
}

class _DetectionTabState extends State<DetectionTab> {
  final TextEditingController _urlController = TextEditingController();
  String? _selectedFilePath;
  bool _isAnalyzing = false; // 분석 진행 상태

  @override
  void initState() {
    super.initState();
    if (widget.sharedUrl != null) {
      _urlController.text = widget.sharedUrl!;
    }
  }

  @override
  void didUpdateWidget(covariant DetectionTab oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.sharedUrl != oldWidget.sharedUrl && widget.sharedUrl != null) {
      setState(() {
        _urlController.text = widget.sharedUrl!;
        _selectedFilePath = null;
      });
    }
  }

  Future<void> _pickFile() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.video,
      );

      if (result != null) {
        setState(() {
          _selectedFilePath = result.files.single.path;
          _urlController.clear();
        });
      }
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('파일 선택 중 오류 발생: $e')));
    }
  }

  // [수정됨] 분석 시작 함수 (새로운 백엔드 API 연동 완료)
  void _startAnalysis() async {
    if (_isAnalyzing) return; // 중복 실행 방지

    // API가 세션을 확인하므로 로그인 상태를 먼저 체크합니다.
    if (!widget.isLoggedIn) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('로그인이 필요합니다.'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    final url = _urlController.text;
    final filePath = _selectedFilePath;

    if (url.isEmpty && filePath == null) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('URL을 입력하거나 파일을 선택해주세요.')));
      return;
    }

    setState(() {
      _isAnalyzing = true; // 분석 시작 상태
    });

    // [중요] 테스트용 Callback URL
    // 서버의 SSRF 방지 로직 [cite: 40-51] 때문에 localhost, 192.168... 등은 차단됩니다.
    // https://webhook.site/ 에서 본인의 고유 주소를 발급받아 사용하세요.
    const String tempCallbackUrl =
        "https://webhook.site/8d6449dc-bffb-4265-89e1-6a2f6eff554f";

    try {
      // --- 1. URL 전송 로직 (/link API 호출) ---
      if (url.isNotEmpty) {
        print('URL 전송 시작: $url');

        // 1. [수정] 쿠키 헤더 불러오기
        final headers = await getAuthHeaders();

        // [수정] 백엔드가 @RequestParam [cite: 21]으로 받으므로, JSON이 아닌 Map(form-urlencoded)으로 전송
        final response = await httpClient.post(
          Uri.parse('$baseUrl/api/v1/detection/link'),
          headers: headers,
          body: {'videoUrl': url, 'callbackUrl': tempCallbackUrl},
        );

        final responseBody = json.decode(utf8.decode(response.bodyBytes));

        if (response.statusCode == 202) {
          // 202 Accepted
          // [수정] 응답이 requestId만 포함
          final String requestId = responseBody['requestId'].toString();
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('URL 분석 요청 성공! (ID: $requestId)'),
              backgroundColor: Colors.green,
            ),
          );
        } else if (response.statusCode == 401) {
          // [신규] 세션 만료 처리 [cite: 22-25]
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                responseBody['message'] ?? '로그인이 만료되었습니다. 다시 로그인해주세요.',
              ),
              backgroundColor: Colors.red,
            ),
          );
        } else {
          // 400 (유효하지 않은 URL) 또는 500 (서버 에러) 등
          final String errorMessage = responseBody['message'] ?? '알 수 없는 오류';
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('URL 요청 실패: $errorMessage'),
              backgroundColor: Colors.red,
            ),
          );
        }

        // --- 2. 파일 전송 로직 (/upload API 호출) ---
      } else if (filePath != null) {
        print('파일 전송 시작: $filePath');

        // [유지] 파일이 포함된 요청은 MultipartRequest를 사용합니다.
        final request = http.MultipartRequest(
          'POST',
          Uri.parse('$baseUrl/api/v1/detection/upload'),
        );

        // API가 요구하는 두 파라미터
        request.fields['callbackUrl'] = tempCallbackUrl;
        request.files.add(
          await http.MultipartFile.fromPath(
            'file', // @RequestParam("file") [cite: 5]
            filePath,
          ),
        );

        // 3. [수정] 쿠키 헤더 불러오기
        final headers = await getAuthHeaders();
        // 4. [수정] MultipartRequest에 헤더 추가
        request.headers.addAll(headers);

        // [중요] 세션 유지를 위해 공유 httpClient로 전송
        final streamedResponse = await httpClient.send(request);
        final response = await http.Response.fromStream(streamedResponse);
        final responseBody = json.decode(utf8.decode(response.bodyBytes));

        if (response.statusCode == 202) {
          // 202 Accepted
          // [수정] 응답이 requestId만 포함
          final String requestId = responseBody['requestId'].toString();
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('파일 분석 요청 성공! (ID: $requestId)'),
              backgroundColor: Colors.green,
            ),
          );
        } else if (response.statusCode == 401) {
          // [신규] 세션 만료 처리 [cite: 6-9]
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                responseBody['message'] ?? '로그인이 만료되었습니다. 다시 로그인해주세요.',
              ),
              backgroundColor: Colors.red,
            ),
          );
        } else {
          // 400 (유효하지 않은 파일) 또는 500 (서버 에러) 등
          final String errorMessage = responseBody['message'] ?? '알 수 없는 오류';
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('파일 업로드 실패: $errorMessage'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    } catch (e) {
      print('분석 요청 오류: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('서버 통신 오류: $e'), backgroundColor: Colors.red),
      );
    } finally {
      setState(() {
        _isAnalyzing = false; // 분석 종료 상태
        _selectedFilePath = null; // 분석 후 선택 초기화
        _urlController.clear(); // 분석 후 URL 초기화
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.isLoggedIn) {
      // (기존 코드와 동일 - 로그인 안된 경우)
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.lock_outline, size: 80, color: Colors.grey),
            const SizedBox(height: 20),
            const Text(
              '딥페이크 탐지 기능을 사용하려면\n로그인이 필요합니다.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 16, color: Colors.grey),
            ),
            const SizedBox(height: 30),
            GFButton(
              onPressed: () {
                Navigator.of(context).pushReplacementNamed('/login');
              },
              text: "로그인 하러 가기",
              icon: const Icon(Icons.login, color: Colors.white),
              color: GFColors.PRIMARY,
            ),
          ],
        ),
      );
    } else {
      // (기존 코드와 동일 - 로그인 된 경우)
      return SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Center(
              child: GFAvatar(
                size: GFSize.LARGE * 1.5,
                backgroundColor: GFColors.LIGHT,
                child: Icon(
                  Icons.face_retouching_natural,
                  size: GFSize.LARGE,
                  color: GFColors.PRIMARY,
                ),
              ),
            ),
            const SizedBox(height: 15),
            const Text(
              'Deepfake 변조 영상을 탐지합니다.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 40),

            // URL 입력
            const Text(
              'URL로 탐지하기',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),
            GFTextField(
              controller: _urlController,
              decoration: InputDecoration(
                hintText: '영상 URL을 입력하세요 (예: https://...)',
                border: const OutlineInputBorder(),
                suffixIcon: IconButton(
                  icon: const Icon(Icons.clear),
                  onPressed: () {
                    _urlController.clear();
                    setState(() {
                      _selectedFilePath = null;
                    });
                  },
                ),
              ),
              keyboardType: TextInputType.url,
              onChanged: (value) {
                if (value.isNotEmpty) {
                  setState(() {
                    _selectedFilePath = null;
                  });
                }
              },
            ),
            const SizedBox(height: 30),

            // 또는 파일 업로드
            const Text(
              '파일로 탐지하기',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),
            GFButton(
              onPressed: _pickFile,
              text: _selectedFilePath == null ? "동영상 파일 선택" : "다른 파일 선택",
              icon: const Icon(Icons.upload_file, color: Colors.white),
              type: GFButtonType.outline2x,
              blockButton: true,
            ),
            if (_selectedFilePath != null)
              Padding(
                padding: const EdgeInsets.only(top: 8.0),
                child: Text(
                  '선택됨: ${Uri.file(_selectedFilePath!).pathSegments.last}',
                  style: const TextStyle(color: Colors.green),
                  overflow: TextOverflow.ellipsis,
                  textAlign: TextAlign.center,
                ),
              ),
            const SizedBox(height: 50),

            // 탐지 시작 버튼
            _isAnalyzing
                ? const Center(child: CircularProgressIndicator())
                : GFButton(
                    onPressed: _startAnalysis,
                    text: "탐지 시작",
                    size: GFSize.LARGE,
                    icon: const Icon(Icons.play_arrow, color: Colors.white),
                    blockButton: true,
                  ),
          ],
        ),
      );
    }
  }
}
