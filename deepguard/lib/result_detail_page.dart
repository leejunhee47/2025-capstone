import 'package:flutter/material.dart';
import 'package:getwidget/getwidget.dart';
import 'http_client.dart';
import 'dart:convert';

class ResultDetailPage extends StatefulWidget {
  final String requestId; // API 문서상 resultId와 동일하게 취급

  const ResultDetailPage({super.key, required this.requestId});

  @override
  State<ResultDetailPage> createState() => _ResultDetailPageState();
}

class _ResultDetailPageState extends State<ResultDetailPage> {
  bool _isLoading = true;
  bool _isReportLoading = true;

  Map<String, dynamic>? _resultData;
  List<dynamic> _reportImages = []; // 보고서 이미지 리스트

  @override
  void initState() {
    super.initState();
    _fetchDetail();
    _fetchReport(); // 보고서 이미지 가져오기
  }

  // 1. 상세 정보 가져오기 (API 경로 및 파싱 수정)
  Future<void> _fetchDetail() async {
    try {
      final headers = await getAuthHeaders();
      // [수정] API 경로 변경: /api/v1/detection/record -> /api/v1/record
      final response = await httpClient.get(
        Uri.parse('$baseUrl/api/v1/detection/record/${widget.requestId}'),
        headers: headers,
      );

      if (response.statusCode == 200) {
        setState(() {
          _resultData = json.decode(utf8.decode(response.bodyBytes));
          _isLoading = false;
        });
      } else {
        setState(() => _isLoading = false);
        print('상세 조회 실패: ${response.statusCode}');
      }
    } catch (e) {
      print('상세 조회 에러: $e');
      setState(() => _isLoading = false);
    }
  }

  // 2. [추가] 보고서 이미지 가져오기
  Future<void> _fetchReport() async {
    try {
      final headers = await getAuthHeaders();
      // [추가] 보고서 API 호출
      final response = await httpClient.get(
        Uri.parse(
          '$baseUrl/api/v1/detection/record/${widget.requestId}/report',
        ),
        headers: headers,
      );

      if (response.statusCode == 200) {
        final body = json.decode(utf8.decode(response.bodyBytes));
        if (body['images'] != null) {
          setState(() {
            _reportImages = body['images']; // [{sequence: 1, url: ...}, ...]
            // sequence 순으로 정렬
            _reportImages.sort(
              (a, b) => (a['sequence'] ?? 0).compareTo(b['sequence'] ?? 0),
            );
            _isReportLoading = false;
          });
        }
      } else {
        setState(() => _isReportLoading = false);
      }
    } catch (e) {
      print('보고서 조회 에러: $e');
      setState(() => _isReportLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    if (_resultData == null) {
      return const Scaffold(body: Center(child: Text("결과를 불러올 수 없습니다.")));
    }

    // 데이터 매핑 (API 문서 참고)
    // verdict: "real" or "fake"
    final String verdict = _resultData!['verdict'] ?? "unknown";
    final bool isFake = verdict.toLowerCase() == "fake";

    // 확률
    final double probReal = (_resultData!['probabilityReal'] ?? 0.0).toDouble();
    final double probFake = (_resultData!['probabilityFake'] ?? 0.0).toDouble();
    final double confidence = (_resultData!['confidence'] ?? 0.0).toDouble();

    // 요약 정보
    final String summaryTitle = _resultData!['summaryTitle'] ?? "분석 완료";
    final String summaryReason = _resultData!['summaryPrimaryReason'] ?? "-";
    final String summaryDetail =
        _resultData!['summaryDetailedExplanation'] ?? "";
    final String riskLevel = _resultData!['summaryRiskLevel'] ?? "unknown";

    // 프레임 정보
    final int suspiciousCount = _resultData!['suspiciousFrameCount'] ?? 0;
    final double suspiciousRatio = (_resultData!['suspiciousFrameRatio'] ?? 0.0)
        .toDouble();

    return Scaffold(
      appBar: AppBar(title: const Text("상세 분석 결과")),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // --- 1. 메인 판정 카드 ---
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: isFake ? Colors.red.shade50 : Colors.green.shade50,
                borderRadius: BorderRadius.circular(15),
                border: Border.all(
                  color: isFake ? Colors.red : Colors.green,
                  width: 2,
                ),
              ),
              child: Column(
                children: [
                  Icon(
                    isFake
                        ? Icons.warning_amber_rounded
                        : Icons.check_circle_outline,
                    size: 60,
                    color: isFake ? Colors.red : Colors.green,
                  ),
                  const SizedBox(height: 10),
                  Text(
                    summaryTitle,
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: isFake ? Colors.red : Colors.green,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 5),
                  Text(
                    "신뢰도 ${(confidence * 100).toStringAsFixed(1)}% / 위험도 $riskLevel",
                    style: const TextStyle(fontSize: 14, color: Colors.black54),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 25),

            // --- 2. 상세 수치 정보 ---
            const Text(
              "분석 수치",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),
            _buildInfoRow("Real 확률", "${(probReal * 100).toStringAsFixed(1)}%"),
            _buildInfoRow("Fake 확률", "${(probFake * 100).toStringAsFixed(1)}%"),
            _buildInfoRow(
              "의심 프레임",
              "$suspiciousCount장 (전체의 ${suspiciousRatio.toStringAsFixed(1)}%)",
            ),
            _buildInfoRow("주요 원인", summaryReason),

            const SizedBox(height: 25),

            // --- 3. 분석 상세 내용 ---
            const Text(
              "AI 분석 요약",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(15),
              decoration: BoxDecoration(
                color: Colors.grey.shade100,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Text(
                summaryDetail,
                style: const TextStyle(fontSize: 15, height: 1.5),
              ),
            ),

            const SizedBox(height: 30),

            // --- 4. 상세 보고서 (이미지) ---
            const Text(
              "상세 보고서",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),
            if (_isReportLoading)
              const Center(child: CircularProgressIndicator())
            else if (_reportImages.isEmpty)
              const Text(
                "생성된 보고서 이미지가 없습니다.",
                style: TextStyle(color: Colors.grey),
              )
            else
              Column(
                children: _reportImages.map((img) {
                  return Container(
                    margin: const EdgeInsets.only(bottom: 15),
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.grey.shade300),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      // 이미지 로드 (네트워크 이미지)
                      child: Image.network(
                        img['url'],
                        width: double.infinity,
                        fit: BoxFit.cover,
                        loadingBuilder: (context, child, loadingProgress) {
                          if (loadingProgress == null) return child;
                          return Container(
                            height: 200,
                            color: Colors.grey.shade100,
                            child: const Center(
                              child: CircularProgressIndicator(),
                            ),
                          );
                        },
                        errorBuilder: (context, error, stackTrace) {
                          return Container(
                            height: 100,
                            alignment: Alignment.center,
                            child: const Text("이미지를 불러올 수 없습니다."),
                          );
                        },
                      ),
                    ),
                  );
                }).toList(),
              ),
            const SizedBox(height: 30),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 100,
            child: Text(
              label,
              style: const TextStyle(
                color: Colors.grey,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(fontWeight: FontWeight.w500),
            ),
          ),
        ],
      ),
    );
  }
}
