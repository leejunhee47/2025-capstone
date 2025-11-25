import 'package:flutter/material.dart';
import 'package:getwidget/getwidget.dart';
import 'http_client.dart';
import 'dart:convert';
import 'result_detail_page.dart'; // [필수] 상세 페이지 임포트

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
      final response = await httpClient.get(
        Uri.parse('$baseUrl/api/v1/detection/history'),
        headers: headers,
      );

      if (response.statusCode == 200) {
        setState(() {
          _historyList = json.decode(utf8.decode(response.bodyBytes));
        });
      }
    } catch (e) {
      print("히스토리 로드 실패: $e");
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.isLoggedIn) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.history_toggle_off, size: 80, color: Colors.grey),
            const SizedBox(height: 20),
            const Text(
              '탐지 기록을 보려면\n로그인이 필요합니다.',
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
      child: ListView.builder(
        itemCount: _historyList.length,
        itemBuilder: (context, index) {
          final item = _historyList[index];
          // item: {requestId, date, status, result, title, thumbnail...}

          // 상태에 따른 아이콘/색상 결정
          bool isCompleted = item['status'] == 'COMPLETED';
          // (서버가 아직 result를 "안전" 등으로 주지 않고 status만 줄 경우 대비)
          String resultText = item['result'] ?? item['status'];

          return GFListTile(
            // [수정됨] icon -> child
            avatar: const GFAvatar(
              shape: GFAvatarShape.square,
              backgroundColor: GFColors.LIGHT,
              child: Icon(
                Icons.play_circle_fill,
                color: Colors.white,
              ), // 배경색 추가
            ),
            titleText: item['title'] ?? '영상 #${item['requestId']}',
            subTitleText: item['date'],
            icon: Icon(
              isCompleted ? Icons.check_circle : Icons.hourglass_empty,
              color: isCompleted ? Colors.green : Colors.grey,
            ),
            description: Text(
              resultText,
              style: TextStyle(
                fontWeight: FontWeight.bold,
                color: resultText == "위험" ? Colors.red : Colors.black54,
              ),
            ),
            onTap: () {
              // 클릭 시 상세 페이지 이동
              Navigator.of(context).push(
                MaterialPageRoute(
                  builder: (context) =>
                      ResultDetailPage(requestId: item['requestId'].toString()),
                ),
              );
            },
          );
        },
      ),
    );
  }
}
