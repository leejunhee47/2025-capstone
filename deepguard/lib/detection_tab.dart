import 'package:flutter/material.dart';
import 'package:getwidget/getwidget.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'http_client.dart';
import 'dart:async';
import 'result_detail_page.dart';

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

  // 상태 변수
  bool _isAnalyzing = false;
  String _analysisStatusText = "";

  // 결과 데이터 저장 변수
  String? _finalResultMsg;
  String? _finalResultId;
  double _realProb = 0.0;
  double _fakeProb = 0.0;
  String _videoName = ""; // 결과 카드에 표시할 영상 이름
  bool _isFakeResult = false;
  bool _showResult = false; // 결과를 보여줄지 여부

  Timer? _pollingTimer;

  @override
  void initState() {
    super.initState();
    if (widget.sharedUrl != null) {
      _urlController.text = widget.sharedUrl!;
      _videoName = widget.sharedUrl!;
    }
  }

  @override
  void didUpdateWidget(covariant DetectionTab oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.sharedUrl != oldWidget.sharedUrl && widget.sharedUrl != null) {
      setState(() {
        _urlController.text = widget.sharedUrl!;
        _selectedFilePath = null;
        _showResult = false; // 새 URL이 오면 결과 숨김
        _videoName = widget.sharedUrl!;
      });
    }
  }

  @override
  void dispose() {
    _pollingTimer?.cancel();
    super.dispose();
  }

  Future<void> _pickFile() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.video,
      );
      if (result != null) {
        setState(() {
          _selectedFilePath = result.files.single.path;
          _videoName = result.files.single.name; // 파일명 저장
          _urlController.clear();
          _showResult = false; // 새 파일 선택 시 결과 숨김
        });
      }
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('파일 선택 오류: $e')));
    }
  }

  void _startAnalysis() async {
    if (_isAnalyzing) return;

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

    // [분석 시작] 로딩 ON, 기존 결과 OFF
    setState(() {
      _isAnalyzing = true;
      _analysisStatusText = "서버에 요청을 전송 중입니다...";
      _showResult = false;
      // 영상 이름 업데이트 (URL인 경우)
      if (url.isNotEmpty) _videoName = url;
    });

    const String tempCallbackUrl = "https://webhook.site/test-callback";

    try {
      String? requestId;

      // --- URL 전송 ---
      if (url.isNotEmpty) {
        final headers = await getAuthHeaders();
        final response = await httpClient.post(
          Uri.parse('$baseUrl/api/v1/detection/link'),
          headers: headers,
          body: {'videoUrl': url, 'callbackUrl': tempCallbackUrl},
        );
        requestId = _parseRequestId(response);

        // --- 파일 전송 ---
      } else if (filePath != null) {
        final request = http.MultipartRequest(
          'POST',
          Uri.parse('$baseUrl/api/v1/detection/upload'),
        );

        request.fields['callbackUrl'] = tempCallbackUrl;
        request.files.add(await http.MultipartFile.fromPath('file', filePath));

        final headers = await getAuthHeaders();
        request.headers.addAll(headers);

        final streamedResponse = await httpClient.send(request);
        final response = await http.Response.fromStream(streamedResponse);
        requestId = _parseRequestId(response);
      }

      // 요청 성공 -> 폴링 시작
      if (requestId != null) {
        setState(() {
          _analysisStatusText = "분석 진행 중... (ID: $requestId)\n잠시만 기다려주세요.";
        });
        _startPolling(requestId);
      }
    } catch (e) {
      print('오류 발생: $e');
      _finishAnalysis(false, "오류 발생: $e");
    }
  }

  String? _parseRequestId(http.Response response) {
    try {
      final body = json.decode(utf8.decode(response.bodyBytes));
      if (response.statusCode == 202) {
        return body['requestId'].toString();
      } else {
        throw Exception(body['message'] ?? '요청 실패 (${response.statusCode})');
      }
    } catch (e) {
      throw Exception('응답 파싱 실패: $e');
    }
  }

  void _startPolling(String requestId) {
    int pollCount = 0;
    const int maxPollCount = 60; // 3초 * 60회 = 3분 대기

    _pollingTimer?.cancel();
    _pollingTimer = Timer.periodic(const Duration(seconds: 3), (timer) async {
      pollCount++;

      if (pollCount > maxPollCount) {
        timer.cancel();
        _finishAnalysis(false, "시간 초과: 서버 응답이 없습니다.");
        return;
      }

      try {
        final headers = await getAuthHeaders();
        // [수정] 폴링 API 주소는 그대로 유지 (body 파싱 방식이 변경됨)
        final response = await httpClient.post(
          Uri.parse('$baseUrl/api/v1/detection/polling'),
          headers: headers,
          body: {'requestId': requestId},
        );

        final rawBody = utf8.decode(response.bodyBytes);
        print("Polling 응답 ($pollCount회차): $rawBody");

        if (response.statusCode == 200) {
          // [수정] 200 OK가 떨어지면 분석 완료로 간주하고 데이터 파싱
          // 문서에 따르면 성공 시 바로 결과 JSON이 옴
          final body = json.decode(rawBody);

          // 데이터 파싱 (변경된 API 구조 반영)
          // 예시: {"resultId": 123, "durationSec": 15.5, "probabilityReal": 0.12, "probabilityFake": 0.88, ...}

          timer.cancel();

          double rProb = 0.0;
          double fProb = 0.0;
          double duration = 0.0;

          if (body is Map) {
            // [중요] 기존 probabilities['real'] 방식에서 -> probabilityReal (Flat 필드)로 변경
            rProb = (body['probabilityReal'] ?? 0.0).toDouble();
            fProb = (body['probabilityFake'] ?? 0.0).toDouble();

            if (body.containsKey('durationSec')) {
              duration = (body['durationSec'] as num).toDouble();
            }
          }

          _finishAnalysisSuccess(rProb, fProb, duration, requestId);
        } else if (response.statusCode == 202) {
          // 202 Accepted: 아직 처리 중
          print("서버 처리 중... ($pollCount회차)");
        } else if (response.statusCode == 401) {
          timer.cancel();
          _finishAnalysis(false, "세션 만료: 다시 로그인해주세요.");
        } else {
          // 실패 혹은 에러
          // 만약 status: FAILED 같은 응답이 온다면 여기서 처리
          timer.cancel();
          _finishAnalysis(false, "분석 실패 또는 오류 (${response.statusCode})");
        }
      } catch (e) {
        print("폴링 통신 오류: $e");
      }
    });
  }

  // 성공 시 처리
  void _finishAnalysisSuccess(
    double real,
    double fake,
    double duration,
    String requestId,
  ) {
    setState(() {
      _isAnalyzing = false;
      _analysisStatusText = "";
      _showResult = true; // 결과 표시 ON
      _realProb = real;
      _fakeProb = fake;
      _finalResultId = requestId;
      _isFakeResult = fake > 0.5;
    });

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text("분석이 완료되었습니다."),
        backgroundColor: Colors.green,
      ),
    );
  }

  // 실패 시 처리
  void _finishAnalysis(bool success, String message) {
    setState(() {
      _isAnalyzing = false;
      _analysisStatusText = "";
      _showResult = false;
    });
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.isLoggedIn) {
      return Center(
        child: GFButton(
          onPressed: () => Navigator.of(context).pushReplacementNamed('/login'),
          text: "로그인 필요",
          icon: const Icon(Icons.login, color: Colors.white),
        ),
      );
    }

    return SingleChildScrollView(
      padding: const EdgeInsets.all(20.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const SizedBox(height: 20),

          // [상단 로고] - 결과가 없을 때만 표시
          if (!_showResult)
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

          const SizedBox(height: 30),

          // [입력 폼] - 언제나 표시 (다음 영상 바로 분석 가능하게)
          const Text(
            'URL로 탐지하기',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 10),
          GFTextField(
            controller: _urlController,
            readOnly: _isAnalyzing,
            decoration: InputDecoration(
              hintText: '영상 URL 입력',
              border: const OutlineInputBorder(),
              suffixIcon: IconButton(
                icon: const Icon(Icons.clear),
                onPressed: _isAnalyzing
                    ? null
                    : () {
                        _urlController.clear();
                        setState(() {
                          _selectedFilePath = null;
                          _showResult = false; // 입력 변경 시 결과 숨김
                        });
                      },
              ),
            ),
            onChanged: (val) {
              if (val.isNotEmpty) {
                setState(() {
                  _selectedFilePath = null;
                  _showResult = false;
                });
              }
            },
          ),
          const SizedBox(height: 30),

          const Text(
            '파일로 탐지하기',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 10),
          GFButton(
            onPressed: _isAnalyzing ? null : _pickFile,
            text: _selectedFilePath == null ? "동영상 파일 선택" : "다른 파일 선택",
            type: GFButtonType.outline2x,
            blockButton: true,
            icon: const Icon(Icons.upload_file, color: Colors.white),
          ),
          if (_selectedFilePath != null && !_showResult)
            Padding(
              padding: const EdgeInsets.only(top: 8.0),
              child: Text(
                '선택됨: ${Uri.file(_selectedFilePath!).pathSegments.last}',
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.green),
              ),
            ),

          const SizedBox(height: 50),

          // [액션 버튼] 또는 [로딩]
          if (_isAnalyzing)
            Column(
              children: [
                const CircularProgressIndicator(),
                const SizedBox(height: 20),
                Text(
                  _analysisStatusText,
                  textAlign: TextAlign.center,
                  style: const TextStyle(color: Colors.grey, fontSize: 16),
                ),
              ],
            )
          else
            GFButton(
              onPressed: _startAnalysis,
              text: "탐지 시작",
              size: GFSize.LARGE,
              icon: const Icon(Icons.play_arrow, color: Colors.white),
              blockButton: true,
            ),

          // [결과 카드] - 버튼 바로 아래에 표시
          if (_showResult) ...[const SizedBox(height: 30), _buildResultCard()],
        ],
      ),
    );
  }

  // 결과 표시 위젯 (게이지 바 + 상세 버튼)
  Widget _buildResultCard() {
    int realPercent = (_realProb * 100).round();
    int fakePercent = (_fakeProb * 100).round();

    // 게이지 바 비율 계산 (최소 10%는 보이게 보정)
    int flexReal = realPercent < 10 ? 10 : realPercent;
    int flexFake = fakePercent < 10 ? 10 : fakePercent;
    if (realPercent == 0 && fakePercent == 0) {
      flexReal = 1;
      flexFake = 1;
    }

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border.all(color: Colors.grey.shade300),
        borderRadius: BorderRadius.circular(15),
        boxShadow: [
          BoxShadow(color: Colors.black12, blurRadius: 10, spreadRadius: 2),
        ],
      ),
      child: Column(
        children: [
          // 1. 썸네일 & 제목
          Icon(Icons.video_library_rounded, size: 50, color: Colors.blueGrey),
          const SizedBox(height: 10),
          Text(
            _videoName,
            style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
            textAlign: TextAlign.center,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 25),

          // 2. 게이지 바 (REAL vs FAKE)
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              // FAKE 쪽
              Column(
                children: [
                  Icon(
                    Icons.warning_amber_rounded,
                    color: Colors.red,
                    size: 28,
                  ),
                  Text(
                    "FAKE\n$fakePercent%",
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: Colors.red,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                    ),
                  ),
                ],
              ),

              // 바
              Expanded(
                child: Container(
                  height: 15,
                  margin: const EdgeInsets.symmetric(horizontal: 10),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: Row(
                      children: [
                        Expanded(
                          flex: flexFake,
                          child: Container(color: Colors.red),
                        ),
                        Expanded(
                          flex: flexReal,
                          child: Container(color: Colors.green),
                        ),
                      ],
                    ),
                  ),
                ),
              ),

              // REAL 쪽
              Column(
                children: [
                  Icon(
                    Icons.check_circle_outline,
                    color: Colors.green,
                    size: 28,
                  ),
                  Text(
                    "REAL\n$realPercent%",
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: Colors.green,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 20),

          // 3. 판정 결과
          Text(
            _isFakeResult ? "딥페이크 의심 영상입니다." : "정상 영상입니다.",
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: _isFakeResult ? Colors.red : Colors.green,
            ),
          ),

          const SizedBox(height: 20),

          // 4. 자세히 보기 버튼
          SizedBox(
            width: double.infinity,
            child: GFButton(
              onPressed: () {
                if (_finalResultId != null) {
                  Navigator.of(context).push(
                    MaterialPageRoute(
                      builder: (context) =>
                          ResultDetailPage(requestId: _finalResultId!),
                    ),
                  );
                }
              },
              text: "자세히 보기",
              type: GFButtonType.outline,
              size: GFSize.MEDIUM,
              blockButton: true,
            ),
          ),
        ],
      ),
    );
  }
}
