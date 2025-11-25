import 'package:flutter/material.dart';
import 'package:getwidget/getwidget.dart';
import 'http_client.dart';
import 'dart:convert';
import 'result_detail_page.dart';
import 'package:intl/intl.dart';

class HistoryTab extends StatefulWidget {
  final bool isLoggedIn;
  const HistoryTab({super.key, required this.isLoggedIn});

  @override
  State<HistoryTab> createState() => _HistoryTabState();
}

class _HistoryTabState extends State<HistoryTab> {
  List<dynamic> _historyList = [];
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    if (widget.isLoggedIn) _fetchHistory();
  }

  Future<void> _fetchHistory() async {
    setState(() => _isLoading = true);
    try {
      final headers = await getAuthHeaders();
      // [수정] limit 등의 파라미터가 필요하다면 쿼리 스트링 추가
      final response = await httpClient.get(
        Uri.parse('$baseUrl/api/v1/record?limit=20'),
        headers: headers,
      );

      if (response.statusCode == 200) {
        // [수정] 서버 응답 구조: { "items": [...], "pageInfo": {...} }
        final body = json.decode(utf8.decode(response.bodyBytes));
        setState(() {
          // items 리스트만 추출
          _historyList = body['items'] ?? [];
        });
      } else {
        print("히스토리 로드 실패: ${response.statusCode}");
      }
    } catch (e) {
      print("히스토리 로드 에러: $e");
    } finally {
      setState(() => _isLoading = false);
    }
  }

  // 날짜 포맷 헬퍼
  String _formatDate(String? isoDate) {
    if (isoDate == null) return "-";
    try {
      final dt = DateTime.parse(isoDate).toLocal();
      return DateFormat('yyyy-MM-dd HH:mm').format(dt);
    } catch (e) {
      return isoDate;
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.isLoggedIn) {
      // (기존 로그인 유도 UI 유지)
      return const Center(child: Text("로그인이 필요합니다."));
    }

    if (_isLoading) return const Center(child: CircularProgressIndicator());

    if (_historyList.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text("탐지 기록이 없습니다."),
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: _fetchHistory,
            ),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _fetchHistory,
      child: ListView.separated(
        padding: const EdgeInsets.all(10),
        itemCount: _historyList.length,
        separatorBuilder: (context, index) => const SizedBox(height: 10),
        itemBuilder: (context, index) {
          final item = _historyList[index];

          // [수정] 서버 필드명 매핑 (DetectionBriefResponse 참고)
          // resultId, verdict, confidence, createdAt, summaryTitle 등
          final String title =
              item['summaryTitle'] ?? '분석 결과 #${item['resultId']}';
          final String date = _formatDate(item['createdAt']);
          final String verdict = item['verdict'] ?? 'unknown';
          final bool isFake = verdict.toLowerCase() == 'fake';
          final double confidence = (item['confidence'] ?? 0.0).toDouble();

          return GFListTile(
            color: Colors.white,
            shadow: const BoxShadow(color: Colors.black12, blurRadius: 2),
            avatar: GFAvatar(
              shape: GFAvatarShape.standard,
              backgroundColor: isFake
                  ? Colors.red.shade100
                  : Colors.green.shade100,
              child: Icon(
                isFake ? Icons.warning : Icons.check_circle,
                color: isFake ? Colors.red : Colors.green,
              ),
            ),
            titleText: title,
            subTitleText: date,
            description: Text(
              isFake
                  ? "FAKE (${(confidence * 100).toStringAsFixed(1)}%)"
                  : "REAL (${(confidence * 100).toStringAsFixed(1)}%)",
              style: TextStyle(
                fontWeight: FontWeight.bold,
                color: isFake ? Colors.red : Colors.green,
              ),
            ),
            icon: const Icon(
              Icons.arrow_forward_ios,
              size: 16,
              color: Colors.grey,
            ),
            onTap: () {
              Navigator.of(context).push(
                MaterialPageRoute(
                  builder: (context) =>
                      ResultDetailPage(resultId: item['resultId'].toString()),
                ),
              );
            },
          );
        },
      ),
    );
  }
}
