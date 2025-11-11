import 'package:flutter/material.dart';
import 'home_page.dart'; // 메인 화면
import 'login_page.dart'; // 로그인 화면 (별도 생성 필요)
import 'signup_page.dart';
import 'http_client.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() async {
  // 4. Flutter 바인딩 초기화
  WidgetsFlutterBinding.ensureInitialized();

  // 5. 앱 시작 시 저장된 상태 불러오기
  await loadAppState();

  runApp(const MyApp());
}

Future<void> loadAppState() async {
  final prefs = await SharedPreferences.getInstance();

  // 1. 저장된 로그인 상태를 전역 변수에 복원
  globalIsLoggedIn = prefs.getBool('isLoggedIn') ?? false;
  globalUserNickname = prefs.getString('userNickname') ?? "";
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Deepfake Detector',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      // 시작 화면을 HomePage로 변경
      home: const HomePage(),
      // 페이지 이동을 위한 라우트 설정
      routes: {
        '/home': (context) => const HomePage(),
        '/login': (context) => const LoginPage(),
        '/signup': (context) => const SignupPage(),
      },
    );
  }
}
