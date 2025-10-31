//package com.capstone.backend.core.config;
//
//import bankapp.account.exceptions.IllegalTransferStateException;
//import bankapp.account.exceptions.InvalidAmountException;
//import bankapp.account.exceptions.PendingTransferNotFoundException;
//import bankapp.account.exceptions.PrimaryAccountNotFoundException;
//import lombok.extern.slf4j.Slf4j;
//import org.springframework.http.HttpStatus;
//import org.springframework.web.bind.annotation.ControllerAdvice;
//import org.springframework.web.bind.annotation.ExceptionHandler;
//import org.springframework.web.bind.annotation.ResponseStatus;
//import org.springframework.web.servlet.resource.NoResourceFoundException;
//
//@Slf4j
//@ControllerAdvice
//public class GlobalExceptionHandler {
//
//    /**
//     * 404 Not Found: 사용자가 잘못된 URL을 요청했을 때 처리합니다.
//     */
//    @ResponseStatus(HttpStatus.NOT_FOUND)
//    @ExceptionHandler(NoResourceFoundException.class)
//    public String handleNotFound(NoResourceFoundException e) {
//        log.warn("잘못된 URL 요청이 들어왔습니다 : {}", e.getMessage());
//        return "error/404";
//    }
//
//    /**
//     * 500 Internal Server Error: 서버 내부 로직에서 발생하는 예외를 처리합니다.
//     */
//    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
//    @ExceptionHandler({
//            InvalidAmountException.class,
//            PendingTransferNotFoundException.class,
//            IllegalTransferStateException.class,
//            IllegalStateException.class,
//            PrimaryAccountNotFoundException.class,
//            Exception.class // 가장 상위 타입인 Exception을 마지막에 두어 혹시 모를 다른 모든 서버 오류도 처리
//    })
//    public String handleServerException(Exception e) {
//        log.error("서버 오류 발생: {}", e.getMessage(), e);
//        return "error/500";
//    }
//}
//
///*
// * [💡 ExceptionHandler 처리 우선순위에 대한 메모]
// * 스프링의 @ControllerAdvice는 예외를 처리할 핸들러를 찾을 때,
// * 발생한 예외와 가장 가까운, 즉 가장 구체적인(자식) 타입의 핸들러를 우선적으로 선택합니다.
// * 이 클래스에서 NoResourceFoundException이 발생하면, 다음과 같은 순서로 핸들러를 찾습니다.
// * 1. @ExceptionHandler(NoResourceFoundException.class)가 있는가? -> handleNotFound() 발견! -> 선택!
// * 2. (만약 1번이 없었다면) @ExceptionHandler(Exception.class)가 있는가? -> handleServerException() 발견! -> 선택!
// * 따라서 특정 예외 핸들러와 일반적인 Exception 핸들러가 공존할 수 있으며,
// * Exception 핸들러는 다른 특정 핸들러에서 처리하지 못한 모든 예외를 담당하는 '최후의 보루' 역할을 합니다.
// */