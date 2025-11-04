import 'package:flutter/material.dart';
import 'package:getwidget/getwidget.dart';
import 'http_client.dart'; // <-- 1단계에서 만든 공유 클라이언트
import 'dart:convert'; // JSON 인코딩/디코딩

class SignupPage extends StatefulWidget {
  const SignupPage({super.key});

  @override
  State<SignupPage> createState() => _SignupPageState();
}

class _SignupPageState extends State<SignupPage> {
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _idController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _passwordConfirmController =
      TextEditingController();
  bool _isLoading = false;

  Future<void> _signUp() async {
    if (_isLoading) return;

    setState(() {
      _isLoading = true;
    });

    // 백엔드 SignUpRequest DTO 형식에 맞게 body 구성
    final Map<String, String> requestBody = {
      'name': _nameController.text,
      'username': _idController.text,
      'password': _passwordController.text,
      'passwordConfirm': _passwordConfirmController.text,
    };

    final String url = '$baseUrl/api/v1/users'; // 회원가입 API 주소

    try {
      final response = await httpClient.post(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json; charset=UTF-8'},
        body: json.encode(requestBody),
      );

      // 한글 깨짐 방지를 위해 utf8.decode 사용
      final responseBody = json.decode(utf8.decode(response.bodyBytes));
      final String message = responseBody['message'];

      if (response.statusCode == 201) {
        // 회원가입 성공
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text(message)));
        // 성공 시 로그인 페이지로 복귀
        Navigator.of(context).pop();
      } else {
        // 회원가입 실패 (유효성 검사, ID 중복, 비번 불일치 등)
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('회원가입 실패: $message'),
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
      appBar: AppBar(title: const Text('회원가입')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            GFTextField(
              controller: _nameController,
              decoration: const InputDecoration(
                labelText: '이름 (예: 홍길동)',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 15),
            GFTextField(
              controller: _idController,
              decoration: const InputDecoration(
                labelText: '아이디 (영문 소문자+숫자, 4~20자)',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 15),
            GFTextField(
              controller: _passwordController,
              obscureText: true,
              obscuringCharacter: '*',
              decoration: const InputDecoration(
                labelText: '비밀번호 (영문+숫자+특수문자, 8~16자)',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 15),
            GFTextField(
              controller: _passwordConfirmController,
              obscureText: true,
              obscuringCharacter: '*',
              decoration: const InputDecoration(
                labelText: '비밀번호 확인',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 30),
            _isLoading
                ? const CircularProgressIndicator()
                : GFButton(
                    onPressed: _signUp,
                    text: "가입하기",
                    blockButton: true,
                    size: GFSize.LARGE,
                  ),
          ],
        ),
      ),
    );
  }
}
