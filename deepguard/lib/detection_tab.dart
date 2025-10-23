import 'package:flutter/material.dart';
import 'package:getwidget/getwidget.dart';
import 'package:file_picker/file_picker.dart'; // 파일 피커 임포트

class DetectionTab extends StatefulWidget {
  final bool isLoggedIn; // 로그인 상태를 받아옴

  const DetectionTab({super.key, required this.isLoggedIn});

  @override
  State<DetectionTab> createState() => _DetectionTabState();
}

class _DetectionTabState extends State<DetectionTab> {
  final TextEditingController _urlController = TextEditingController();
  String? _selectedFilePath;
  bool _isAnalyzing = false; // 분석 진행 상태

  // 파일 선택 함수
  Future<void> _pickFile() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.video, // 비디오 파일만 선택하도록 제한
      );

      if (result != null) {
        setState(() {
          _selectedFilePath = result.files.single.path;
          _urlController.clear(); // 파일 선택 시 URL 입력 초기화
        });
        print('선택된 파일: $_selectedFilePath');
      } else {
        // 사용자가 선택 취소
        print('파일 선택이 취소되었습니다.');
      }
    } catch (e) {
      print('파일 선택 오류: $e');
      // 오류 메시지 표시 (예: SnackBar)
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('파일 선택 중 오류 발생: $e')));
    }
  }

  // 분석 시작 함수 (서버 전송 로직 필요)
  void _startAnalysis() async {
    if (_isAnalyzing) return; // 이미 분석 중이면 중복 실행 방지

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

    try {
      // --- 서버 전송 로직 ---
      if (url.isNotEmpty) {
        print('URL 전송 시작: $url');
        // TODO: http 패키지 등을 사용하여 서버로 URL 전송 및 결과 대기
        await Future.delayed(const Duration(seconds: 3)); // 임시 지연
      } else if (filePath != null) {
        print('파일 전송 시작: $filePath');
        // TODO: http 패키지 등을 사용하여 서버로 파일 전송 및 결과 대기
        await Future.delayed(const Duration(seconds: 5)); // 임시 지연
      }
      // --- 분석 완료 ---
      print('분석 완료!');
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('분석이 완료되었습니다.')));
      // TODO: 분석 결과 화면으로 이동 또는 결과 표시 로직
    } catch (e) {
      print('분석 요청 오류: $e');
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('분석 요청 중 오류 발생: $e')));
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
      // 로그인 안된 경우
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
      // 로그인 된 경우
      return SingleChildScrollView(
        // 키보드 올라올 때 오버플로우 방지
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // 참고 이미지와 유사한 상단 디자인 (선택 사항)
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
                  // 입력 내용 지우기 버튼
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
              // URL 입력 시 파일 선택 해제
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
              blockButton: true, // 버튼 너비 최대로
            ),
            if (_selectedFilePath != null) // 선택된 파일 경로 표시
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
                ? const Center(child: CircularProgressIndicator()) // 분석 중 로딩 표시
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
