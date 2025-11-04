import 'package:flutter/material.dart';
import 'package:getwidget/getwidget.dart';
import 'home_page.dart'; // 상태 변수 사용을 위해 임포트
import 'http_client.dart'; // <-- 1단계에서 만든 공유 클라이언트
import 'dart:convert'; // JSON 인코딩/디코딩

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final TextEditingController _idController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  bool _isLoading = false; // 로딩 상태 변수 추가

  Future<void> _login() async {
    if (_isLoading) return; // 중복 요청 방지

    setState(() {
      _isLoading = true;
    });

    String id = _idController.text;
    String password = _passwordController.text;

    if (id.isEmpty || password.isEmpty) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('아이디와 비밀번호를 입력하세요.')));
      setState(() {
        _isLoading = false;
      });
      return;
    }

    final String url = '$baseUrl/api/v1/auth/login'; // 로그인 API 주소
    final Map<String, String> requestBody = {
      'username': id,
      'password': password,
    };

    try {
      // [중요] 세션 유지를 위해 1단계에서 만든 httpClient 사용
      final response = await httpClient.post(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json; charset=UTF-8'},
        body: json.encode(requestBody),
      );

      final responseBody = json.decode(utf8.decode(response.bodyBytes));

      if (response.statusCode == 200) {
        // 로그인 성공
        final String nickname = responseBody['name']; // 서버에서 'name' 필드

        // [중요] 세션 쿠키가 httpClient에 자동으로 저장되었습니다.

        setState(() {
          globalIsLoggedIn = true;
          globalUserNickname = nickname; // 백엔드에서 받은 이름(닉네임)
        });

        // 로그인 성공 후 홈 화면으로 돌아가기 (이전 화면 스택 제거)
        Navigator.of(
          context,
        ).pushNamedAndRemoveUntil('/home', (Route<dynamic> route) => false);
      } else {
        // 로그인 실패 (아이디/비번 오류 등)
        final String errorMessage = responseBody['message'];
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('로그인 실패: $errorMessage'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } catch (e) {
      // 네트워크 오류 또는 서버 접속 불가
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('서버 연결 오류: $e'), backgroundColor: Colors.red),
      );
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
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
              obscuringCharacter: '*',
              decoration: const InputDecoration(
                labelText: '비밀번호',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 30),
            _isLoading
                ? const CircularProgressIndicator()
                : GFButton(
                    onPressed: _login,
                    text: "로그인",
                    blockButton: true,
                    size: GFSize.LARGE,
                  ),
            const SizedBox(height: 15),
            TextButton(
              onPressed: () {
                // 회원가입 페이지로 이동
                Navigator.of(context).pushNamed('/signup');
              },
              child: const Text('회원가입'),
            ),
          ],
        ),
      ),
    );
  }
}
