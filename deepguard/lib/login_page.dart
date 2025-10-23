import 'package:flutter/material.dart';
import 'package:getwidget/getwidget.dart';
import 'home_page.dart'; // 상태 변수 사용을 위해 임포트

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final TextEditingController _idController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();

  void _login() {
    // TODO: 실제 서버 로그인 로직 구현 (http 패키지 사용)
    String id = _idController.text;
    String password = _passwordController.text;

    // --- 임시 로그인 성공 처리 ---
    if (id.isNotEmpty && password.isNotEmpty) {
      setState(() {
        globalIsLoggedIn = true;
        globalUserNickname = id; // 아이디를 닉네임으로 임시 사용
      });
      // 로그인 성공 후 홈 화면으로 돌아가기 (이전 화면 스택 제거)
      Navigator.of(
        context,
      ).pushNamedAndRemoveUntil('/home', (Route<dynamic> route) => false);
    } else {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('아이디와 비밀번호를 입력하세요.')));
    }
    // --- 임시 처리 끝 ---
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('로그인')),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            GFTextField(
              controller: _idController,
              decoration: const InputDecoration(
                labelText: '아이디',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 15),
            GFTextField(
              controller: _passwordController,
              obscureText: true, // 비밀번호 가리기
              decoration: const InputDecoration(
                labelText: '비밀번호',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 30),
            GFButton(
              onPressed: _login,
              text: "로그인",
              blockButton: true,
              size: GFSize.LARGE,
            ),
            const SizedBox(height: 15),
            TextButton(
              onPressed: () {
                // TODO: 회원가입 페이지로 이동하는 로직
                // Navigator.of(context).pushNamed('/signup');
                ScaffoldMessenger.of(
                  context,
                ).showSnackBar(const SnackBar(content: Text('회원가입 기능 구현 필요')));
              },
              child: const Text('회원가입'),
            ),
          ],
        ),
      ),
    );
  }
}
