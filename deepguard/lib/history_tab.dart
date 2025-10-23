import 'package:flutter/material.dart';
import 'package:getwidget/getwidget.dart';

class HistoryTab extends StatelessWidget {
  final bool isLoggedIn; // 로그인 상태

  const HistoryTab({super.key, required this.isLoggedIn});

  @override
  Widget build(BuildContext context) {
    if (!isLoggedIn) {
      // 로그인 안된 경우
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
    } else {
      // 로그인 된 경우 (예시 데이터 사용)
      // TODO: 실제 앱에서는 서버에서 받아온 분석 기록 데이터를 사용
      final List<Map<String, dynamic>> historyData = [
        {
          'title': '분석 영상 1.mp4',
          'date': '2025-10-22',
          'result': '위험',
          'thumbnail': 'https://via.placeholder.com/150/f60',
        },
        {
          'title': 'https://...',
          'date': '2025-10-21',
          'result': '주의',
          'thumbnail': 'https://via.placeholder.com/150/fc0',
        },
        {
          'title': '일상 브이로그.avi',
          'date': '2025-10-20',
          'result': '안전',
          'thumbnail': 'https://via.placeholder.com/150/0c0',
        },
        {
          'title': '뉴스 클립.mov',
          'date': '2025-10-19',
          'result': '위험',
          'thumbnail': 'https://via.placeholder.com/150/f30',
        },
      ];

      if (historyData.isEmpty) {
        return const Center(
          child: Text('아직 탐지 기록이 없습니다.', style: TextStyle(color: Colors.grey)),
        );
      }

      // GetWidget 리스트 타일 사용
      return ListView.builder(
        itemCount: historyData.length,
        itemBuilder: (context, index) {
          final item = historyData[index];
          Color resultColor;
          IconData resultIcon;
          switch (item['result']) {
            case '위험':
              resultColor = GFColors.DANGER;
              resultIcon = Icons.error_outline;
              break;
            case '주의':
              resultColor = GFColors.WARNING;
              resultIcon = Icons.warning_amber_rounded;
              break;
            default: // 안전
              resultColor = GFColors.SUCCESS;
              resultIcon = Icons.check_circle_outline;
          }

          return GFListTile(
            avatar: GFAvatar(
              backgroundImage: NetworkImage(item['thumbnail']),
              shape: GFAvatarShape.square, // 썸네일은 사각형으로
            ),
            titleText: item['title'],
            subTitleText: '분석일: ${item['date']}',
            icon: Icon(resultIcon, color: resultColor), // 결과 아이콘
            // onTap: () {
            //   // TODO: 상세 결과 화면으로 이동
            //   print('${item['title']} 상세 보기');
            // },
          );
        },
      );
    }
  }
}
