import 'package:flutter/material.dart';
import 'package:getwidget/getwidget.dart';
import 'detection_tab.dart'; // 딥페이크 탐지 탭
import 'history_tab.dart'; // 탐지 기록 탭
import 'package:flutter/services.dart';
// import 'login_page.dart'; // 로그인 페이지 (실제 구현 시 필요)

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

// TODO: 실제 앱에서는 Provider, Riverpod, GetX 등으로 상태 관리 필요
// 임시 로그인 상태 변수 (앱 재시작 시 초기화됨)
bool globalIsLoggedIn = false;
String globalUserNickname = "";

class _HomePageState extends State<HomePage> with TickerProviderStateMixin {
  late TabController tabController;

  @override
  void initState() {
    super.initState();
    tabController = TabController(length: 2, vsync: this);
    // TODO: 앱 시작 시 실제 로그인 상태 확인 로직 (예: SharedPreferences)
    // 예시: _loadLoginStatus();
  }

  @override
  void dispose() {
    tabController.dispose();
    super.dispose();
  }

  // 로그아웃 함수 (예시)
  void _logout() {
    setState(() {
      globalIsLoggedIn = false;
      globalUserNickname = "";
    });
    // TODO: 실제 로그아웃 처리 (예: SharedPreferences 토큰 삭제)
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('로그아웃 되었습니다.')));
  }

  // --- 개발용 임시 로그인 토글 함수 ---
  void _toggleLoginForDev() {
    setState(() {
      globalIsLoggedIn = !globalIsLoggedIn;
      if (globalIsLoggedIn) {
        globalUserNickname = "TestUser"; // 임시 닉네임
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text('임시 로그인 완료 (TestUser)')));
      } else {
        globalUserNickname = "";
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text('임시 로그아웃 완료')));
      }
    });
  }
  // --- 개발용 함수 끝 ---

  @override
  Widget build(BuildContext context) {
    // build 메서드 내에서 상태를 다시 읽어옴 (로그인/로그아웃 후 UI 갱신 위해)
    bool isLoggedIn = globalIsLoggedIn;
    String userNickname = globalUserNickname;

    const SystemUiOverlayStyle customSystemOverlayStyle = SystemUiOverlayStyle(
      // 상태바 배경색
      statusBarColor: Color.fromARGB(255, 22, 101, 175),

      // 상태바 아이콘 색상 (Android)
      // 배경이 밝은 회색이므로 아이콘은 어둡게 (dark)
      statusBarIconBrightness: Brightness.dark,

      // 상태바 아이콘 색상 (iOS)
      // 배경이 밝으므로 아이콘은 어둡게 (light가 어두운 아이콘임)
      statusBarBrightness: Brightness.light,
    );

    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: customSystemOverlayStyle,
      child: Scaffold(
        appBar: GFAppBar(
          // --- AppBar 디자인 수정 ---
          backgroundColor: GFColors.WHITE, // AppBar 배경 흰색
          iconTheme: const IconThemeData(color: Colors.black54), // 아이콘 색상
          // titleTextStyle 삭제
          elevation: 1.0, // 약간의 그림자 효과
          // --- AppBar 디자인 수정 끝 ---
          leading: Padding(
            padding: const EdgeInsets.only(top: 8.0), // 상단 여백 추가
            child: SizedBox(
              height: kToolbarHeight + 8.0,
              child: Center(
                child: IconButton(
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(),
                  icon: Icon(
                    isLoggedIn
                        ? Icons.account_circle
                        : Icons.account_circle_outlined,
                    color: isLoggedIn ? GFColors.PRIMARY : Colors.black54,
                    size: 35,
                  ),
                  tooltip: isLoggedIn ? '내 정보 / 로그아웃' : '로그인 / 회원가입',
                  onPressed: () {
                    if (isLoggedIn) {
                      // 로그인 상태: 내 정보 팝업 또는 로그아웃 확인
                      showDialog(
                        context: context,
                        builder: (BuildContext context) {
                          return AlertDialog(
                            title: Text('$userNickname님'),
                            content: const Text('로그아웃 하시겠습니까?'),
                            actions: <Widget>[
                              TextButton(
                                child: const Text('취소'),
                                onPressed: () {
                                  Navigator.of(context).pop();
                                },
                              ),
                              TextButton(
                                child: const Text(
                                  '로그아웃',
                                  style: TextStyle(color: Colors.red),
                                ),
                                onPressed: () {
                                  Navigator.of(context).pop();
                                  _logout(); // 로그아웃 함수 호출
                                },
                              ),
                            ],
                          );
                        },
                      );
                    } else {
                      // 로그아웃 상태: 로그인 페이지로 이동
                      Navigator.of(context).pushNamed('/login').then((_) {
                        setState(() {});
                      });
                    }
                  },
                ),
              ),
            ),
          ),

          // title 에 직접 스타일 적용
          // --- [수정 2] title을 Padding으로 감싸기 ---
          title: Padding(
            padding: const EdgeInsets.only(top: 8.0), // 상단 여백 추가
            child: Container(
              // 1. leading과 동일한 높이 지정
              height: kToolbarHeight + 8.0,
              // 2. 텍스트를 (수직)중앙, (수평)좌측 정렬
              alignment: Alignment.center,
              child: const Text(
                'DeepGuard',
                style: TextStyle(
                  color: Colors.black87,
                  fontSize: 21,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ), // 앱 이름 고정
          // --- [수정 3] actions 안의 Padding 수정 ---
          actions: <Widget>[
            Padding(
              padding: const EdgeInsets.only(top: 4.0), // 상단 여백 추가
              child: Container(
                // 1. leading/title과 동일한 높이 지정
                height: kToolbarHeight + 8.0,
                // 2. 버튼을 (수직)중앙 정렬
                alignment: Alignment.center,
                child: Padding(
                  // 3. 좌우 여백만 유지
                  padding: const EdgeInsets.symmetric(horizontal: 8.0),
                  child: TextButton(
                    onPressed: _toggleLoginForDev,
                    style: TextButton.styleFrom(
                      padding: EdgeInsets.zero,
                      minimumSize: const Size(50, 30),
                      tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                      alignment: Alignment.center,
                    ),
                    child: Text(
                      isLoggedIn ? '로그아웃' : '로그인',
                      style: TextStyle(
                        color: isLoggedIn
                            ? Colors.deepOrange
                            : Colors.blueAccent,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
              ),
            ),
            // --- 개발용 버튼 끝 ---
          ], // --- actions 끝 ---
          // 탭 바 구성
          bottom: PreferredSize(
            // <-- [수정] bottom도 Center 밖, GFAppBar의 속성입니다.
            // PreferredSize로 감싸기
            preferredSize: const Size.fromHeight(
              kToolbarHeight + 25.0,
            ), // 탭 바 높이 지정
            child: Padding(
              padding: const EdgeInsets.only(top: 1.0), // 상단 여백 증가
              child: Container(
                // 탭 바 아래에 구분선 추가
                decoration: BoxDecoration(
                  border: Border(
                    bottom: BorderSide(color: Colors.grey.shade300, width: 1.0),
                  ),
                ),
                child: GFTabBar(
                  length: 2,
                  controller: tabController,
                  // backgroundColor 삭제
                  tabBarColor: GFColors.WHITE, // 탭 바 배경 흰색
                  labelColor: GFColors.PRIMARY, // 활성 탭 텍스트 색상 (Primary)
                  unselectedLabelColor: Colors.black54, // 비활성 탭 텍스트 색상 (회색)
                  indicatorColor: GFColors.PRIMARY, // 활성 탭 밑줄 색상 (Primary)
                  indicatorWeight: 3.0, // 밑줄 두께
                  tabs: const <Widget>[
                    Tab(
                      icon: Icon(Icons.security_outlined), // 아이콘 변경 (선택)
                      text: "딥페이크 탐지",
                    ),
                    Tab(
                      icon: Icon(Icons.history_outlined), // 아이콘 변경 (선택)
                      text: "탐지 기록",
                    ),
                  ],
                ),
              ),
            ),
          ),
        ), // <-- GFAppBar 닫기
        // 탭 뷰 구성 (isLoggedIn 상태 전달)
        body: GFTabBarView(
          // <-- body는 Scaffold의 속성이므로 이 위치가 맞습니다.
          controller: tabController,
          // 사용자가 스와이프로 탭 전환하는 것을 막으려면 아래 주석 해제
          // physics: NeverScrollableScrollPhysics(),
          children: <Widget>[
            DetectionTab(isLoggedIn: isLoggedIn), // 로그인 상태 전달
            HistoryTab(isLoggedIn: isLoggedIn), // 로그인 상태 전달
          ],
        ),
      ),
    );
  }
}
