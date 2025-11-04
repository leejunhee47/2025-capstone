import 'package:flutter/material.dart';
import 'home_page.dart'; // 메인 화면
import 'login_page.dart'; // 로그인 화면 (별도 생성 필요)
import 'signup_page.dart';

void main() {
  // TODO: 앱 시작 시 SharedPreferences 등에서 로그인 정보 불러오기
  runApp(const MyApp());
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
