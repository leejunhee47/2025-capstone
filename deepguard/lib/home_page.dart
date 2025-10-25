import 'package:flutter/material.dart';
import 'package:getwidget/getwidget.dart';
import 'detection_tab.dart'; // 딥페이크 탐지 탭
import 'history_tab.dart'; // 탐지 기록 탭
import 'package:flutter/services.dart';
// import 'login_page.dart'; // 로그인 페이지 (실제 구현 시 필요)

import 'dart:async'; // StreamSubscription을 위해 임포트
import 'package:receive_sharing_intent/receive_sharing_intent.dart';

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

  StreamSubscription? _intentMediaSub;
  String? _sharedText;

  @override
  void initState() {
    super.initState();
    tabController = TabController(length: 2, vsync: this);

    _intentMediaSub = ReceiveSharingIntent.instance.getMediaStream().listen((
      List<SharedMediaFile> value,
    ) {
      if (value.isNotEmpty) {
        final String sharedUrl = value.first.path;
        setState(() {
          _sharedText = sharedUrl;
        });
        tabController.animateTo(0);
      }
    });

    ReceiveSharingIntent.instance.getInitialMedia().then((
      List<SharedMediaFile> value,
    ) {
      if (value.isNotEmpty) {
        final String sharedUrl = value.first.path;
        setState(() {
          _sharedText = sharedUrl;
        });
        Future.microtask(() => tabController.animateTo(0));
        ReceiveSharingIntent.instance.reset();
      }
    });
  }

  @override
  void dispose() {
    tabController.dispose();
    _intentMediaSub?.cancel();
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
      statusBarIconBrightness: Brightness.dark,

      // 상태바 아이콘 색상 (iOS)
      statusBarBrightness: Brightness.light,
    );

    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: customSystemOverlayStyle,
      child: Scaffold(
        appBar: GFAppBar(
          backgroundColor: GFColors.WHITE,
          iconTheme: const IconThemeData(color: Colors.black54),
          elevation: 1.0,
          leading: Padding(
            padding: const EdgeInsets.only(top: 8.0),
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
                                  _logout();
                                },
                              ),
                            ],
                          );
                        },
                      );
                    } else {
                      Navigator.of(context).pushNamed('/login').then((_) {
                        setState(() {});
                      });
                    }
                  },
                ),
              ),
            ),
          ),

          title: Padding(
            padding: const EdgeInsets.only(top: 8.0),
            child: Container(
              height: kToolbarHeight + 8.0,
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
          ),
          actions: <Widget>[
            Padding(
              padding: const EdgeInsets.only(top: 4.0),
              child: Container(
                height: kToolbarHeight + 8.0,
                alignment: Alignment.center,
                child: Padding(
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
          ],
          bottom: PreferredSize(
            preferredSize: const Size.fromHeight(kToolbarHeight + 25.0),
            child: Padding(
              padding: const EdgeInsets.only(top: 1.0),
              child: Container(
                decoration: BoxDecoration(
                  border: Border(
                    bottom: BorderSide(color: Colors.grey.shade300, width: 1.0),
                  ),
                ),
                child: GFTabBar(
                  length: 2,
                  controller: tabController,
                  tabBarColor: GFColors.WHITE,
                  labelColor: GFColors.PRIMARY,
                  unselectedLabelColor: Colors.black54,
                  indicatorColor: GFColors.PRIMARY,
                  indicatorWeight: 3.0,
                  tabs: const <Widget>[
                    Tab(icon: Icon(Icons.security_outlined), text: "딥페이크 탐지"),
                    Tab(icon: Icon(Icons.history_outlined), text: "탐지 기록"),
                  ],
                ),
              ),
            ),
          ),
        ),
        body: GFTabBarView(
          controller: tabController,
          children: <Widget>[
            //DetectionTab(isLoggedIn: isLoggedIn),
            DetectionTab(isLoggedIn: isLoggedIn, sharedUrl: _sharedText),
            HistoryTab(isLoggedIn: isLoggedIn),
          ],
        ),
      ),
    );
  }
}
