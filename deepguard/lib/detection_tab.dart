import 'package:flutter/material.dart';
import 'package:getwidget/getwidget.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'http_client.dart';
import 'dart:async';
import 'dart:typed_data'; // [추가] 이미지 바이너리 처리를 위해 필요
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
  String? _finalResultId; // [중요] 상세 페이지 조회를 위한 Result ID
  String? _currentRequestId; // [추가] 썸네일 조회를 위한 Request ID
  double _realProb = 0.0;
  double _fakeProb = 0.0;
  String _videoName = "";
  bool _isFakeResult = false;
  bool _showResult = false;

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
        _showResult = false;
        _videoName = widget.sharedUrl!;
      });
    }
  }

  @override
  void dispose() {
    _pollingTimer?.cancel();
    super.dispose();
  }

  // [추가] 썸네일 이미지를 POST 요청으로 받아오는 함수
  Future<Uint8List?> _fetchThumbnailBytes(String reqId) async {
    try {
      final headers = await getAuthHeaders();
      // 서버의 DetectionController: passDetectionThumbnail (@PostMapping)
      final response = await httpClient.post(
        Uri.parse('$baseUrl/api/v1/detection/thumbnail/polling'),
        headers: headers,
        body: {'requestId': reqId},
      );

      if (response.statusCode == 200) {
        return response.bodyBytes;
      }
    } catch (e) {
      print("썸네일 로드 실패: $e");
    }
    return null;
  }

  Future<void> _pickFile() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.video,
      );
      if (result != null) {
        setState(() {
          _selectedFilePath = result.files.single.path;
          _videoName = result.files.single.name;
          _urlController.clear();
          _showResult = false;
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

    setState(() {
      _isAnalyzing = true;
      _analysisStatusText = "서버에 요청을 전송 중입니다...";
      _showResult = false;
      _currentRequestId = null; // 초기화
      if (url.isNotEmpty) _videoName = url;
    });

    const String tempCallbackUrl = "https://webhook.site/test-callback";

    try {
      String? requestId;

      if (url.isNotEmpty) {
        final headers = await getAuthHeaders();
        final response = await httpClient.post(
          Uri.parse('$baseUrl/api/v1/detection/link'),
          headers: headers,
          body: {'videoUrl': url, 'callbackUrl': tempCallbackUrl},
        );
        requestId = _parseRequestId(response);
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

      if (requestId != null) {
        setState(() {
          _currentRequestId = requestId; // 썸네일 조회를 위해 ID 저장
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
        // [수정] 결과 폴링 API 주소 변경
        final response = await httpClient.post(
          Uri.parse('$baseUrl/api/v1/detection/result/polling'),
          headers: headers,
          body: {'requestId': requestId},
        );

        // 202: 처리 중, 200: 완료
        if (response.statusCode == 200) {
          final rawBody = utf8.decode(response.bodyBytes);
          final body = json.decode(rawBody);

          timer.cancel();

          // 서버의 DetectionBriefResponse 파싱
          String? resultIdStr;
          if (body.containsKey('resultId')) {
            resultIdStr = body['resultId'].toString();
          }

          // 확률 정보 매핑 (BriefResponse 필드명 확인 필요, 여기선 일반적인 키값 사용)
          double rProb = (body['probabilityReal'] ?? 0.0).toDouble();
          double fProb = (body['probabilityFake'] ?? 0.0).toDouble();
          double duration = 0.0;
          if (body.containsKey('durationSec')) {
            duration = (body['durationSec'] as num).toDouble();
          }

          _finishAnalysisSuccess(rProb, fProb, duration, resultIdStr);
        } else if (response.statusCode == 202) {
          print("서버 처리 중... ($pollCount회차)");
        } else if (response.statusCode == 401) {
          timer.cancel();
          _finishAnalysis(false, "세션 만료: 다시 로그인해주세요.");
        } else {
          timer.cancel();
          _finishAnalysis(false, "분석 실패 또는 오류 (${response.statusCode})");
        }
      } catch (e) {
        print("폴링 통신 오류: $e");
        // 에러가 나도 타이머를 바로 끄진 않고 다음 틱을 기다리거나, 심각한 에러면 종료
      }
    });
  }

  void _finishAnalysisSuccess(
    double real,
    double fake,
    double duration,
    String? resultId,
  ) {
    setState(() {
      _isAnalyzing = false;
      _analysisStatusText = "";
      _showResult = true;
      _realProb = real;
      _fakeProb = fake;
      _finalResultId = resultId; // [중요] 상세페이지용 ResultID 저장
      _isFakeResult = fake > 0.5;
    });

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text("분석이 완료되었습니다."),
        backgroundColor: Colors.green,
      ),
    );
  }

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
                          _showResult = false;
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

  // 결과 표시 위젯
  Widget _buildResultCard() {
    int realPercent = (_realProb * 100).round();
    int fakePercent = (_fakeProb * 100).round();

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
        boxShadow: const [
          BoxShadow(color: Colors.black12, blurRadius: 10, spreadRadius: 2),
        ],
      ),
      child: Column(
        children: [
          // 1. [수정] 썸네일 표시 (FutureBuilder 사용)
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: SizedBox(
              height: 200,
              width: double.infinity,
              child: _currentRequestId != null
                  ? FutureBuilder<Uint8List?>(
                      // RequestID로 썸네일 요청
                      future: _fetchThumbnailBytes(_currentRequestId!),
                      builder: (context, snapshot) {
                        if (snapshot.connectionState ==
                            ConnectionState.waiting) {
                          return Container(
                            color: Colors.grey.shade100,
                            child: const Center(
                              child: CircularProgressIndicator(),
                            ),
                          );
                        }
                        if (snapshot.hasData && snapshot.data != null) {
                          return Image.memory(
                            snapshot.data!,
                            fit: BoxFit.cover,
                          );
                        }
                        // 썸네일 로드 실패 시 기본 아이콘
                        return Container(
                          color: Colors.grey.shade200,
                          child: const Icon(
                            Icons.broken_image,
                            size: 50,
                            color: Colors.grey,
                          ),
                        );
                      },
                    )
                  : Container(
                      color: Colors.grey.shade200,
                      child: const Icon(
                        Icons.video_library_rounded,
                        size: 50,
                        color: Colors.blueGrey,
                      ),
                    ),
            ),
          ),

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
              // FAKE
              Column(
                children: [
                  const Icon(
                    Icons.warning_amber_rounded,
                    color: Colors.red,
                    size: 28,
                  ),
                  Text(
                    "FAKE\n$fakePercent%",
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      color: Colors.red,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
              // Bar
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
              // REAL
              Column(
                children: [
                  const Icon(
                    Icons.check_circle_outline,
                    color: Colors.green,
                    size: 28,
                  ),
                  Text(
                    "REAL\n$realPercent%",
                    textAlign: TextAlign.center,
                    style: const TextStyle(
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
                      // [중요] ResultDetailPage는 이제 resultId를 받음
                      builder: (context) =>
                          ResultDetailPage(resultId: _finalResultId!),
                    ),
                  );
                } else {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text("상세 정보를 확인할 수 없습니다.")),
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
