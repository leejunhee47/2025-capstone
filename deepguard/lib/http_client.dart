import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

// 앱 전역에서 세션 쿠키를 공유하기 위한 단일 클라이언트 인스턴스
final http.Client httpClient = http.Client();

// 백엔드 서버 주소 (본인의 로컬 환경에 맞게 수정)
const String baseUrl = "http://121.164.108.44:8080";
//const String baseUrl = "https://untauntingly-impuissant-amada.ngrok-free.dev";

// 2. [신규] 쿠키가 포함된 헤더를 반환하는 헬퍼 함수
Future<Map<String, String>> getAuthHeaders({bool isJson = false}) async {
  final prefs = await SharedPreferences.getInstance();
  final String? cookie = prefs.getString('sessionCookie');

  final headers = <String, String>{};

  // 3. JSON 요청인지 여부에 따라 Content-Type 설정
  if (isJson) {
    headers['Content-Type'] = 'application/json; charset=UTF-8';
  }

  // 4. 저장된 쿠키가 있으면 헤더에 추가
  if (cookie != null) {
    headers['Cookie'] = cookie;
  }
  return headers;
}
