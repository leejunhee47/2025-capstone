//package com.capstone.backend.member.interceptor;
//
//
//import jakarta.servlet.http.HttpServletRequest;
//import jakarta.servlet.http.HttpServletResponse;
//import jakarta.servlet.http.HttpSession;
//import org.springframework.web.servlet.HandlerInterceptor;
//
//import java.io.IOException;
//
//public class LoginCheckerInterceptor implements HandlerInterceptor {
//
//    @Override
//    public boolean preHandle(HttpServletRequest request , HttpServletResponse response, Object handler) throws Exception{
//
//        HttpSession session = request.getSession();
//
//        if(session == null || session.getAttribute(SessionConst.LOGIN_MEMBER) == null){
//            return handleUnauthenticated(request,response);
//        }else{
//            return handleAuthenticated(request,response);
//        }
//    }
//
//    /**
//     * 인증되지 않은 사용자의 요청을 처리합니다.
//     * - 로그인/회원가입 페이지는 허용합니다.
//     * - 나머지 페이지는 로그인 페이지로 리디렉션합니다.
//     */
//    private boolean handleUnauthenticated(HttpServletRequest request, HttpServletResponse response) throws IOException {
//        String requestURI = request.getRequestURI();
//
//        if (requestURI.equals("/login") || requestURI.equals("/signup")) {
//            return true;
//        }
//
//        response.sendRedirect("/login?redirectUrl=" + requestURI);
//        return false;
//    }
//
//    /**
//     * 인증된 사용자의 요청을 처리합니다.
//     * - 로그인/회원가입 페이지 접근은 홈으로 리디렉션합니다.
//     * - 나머지 페이지는 허용합니다.
//     */
//    private boolean handleAuthenticated(HttpServletRequest request, HttpServletResponse response) throws IOException {
//        String requestURI = request.getRequestURI();
//
//        if (requestURI.equals("/login") || requestURI.equals("/signup")) {
//            response.sendRedirect("/");
//            return false;
//        }
//
//        return true;
//    }
//
//
//
//}
