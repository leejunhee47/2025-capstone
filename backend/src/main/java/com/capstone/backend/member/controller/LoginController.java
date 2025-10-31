package com.capstone.backend.member.controller;


import com.capstone.backend.core.common.SessionConst;
import com.capstone.backend.member.exceptions.IncorrectPasswordException;
import com.capstone.backend.member.exceptions.UsernameNotFoundException;
import com.capstone.backend.member.model.Member;
import com.capstone.backend.member.request.login.LoginRequest;
import com.capstone.backend.member.service.login.LoginService;
import jakarta.servlet.http.HttpSession;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.BindingResult;
import org.springframework.validation.FieldError;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.Map;


/**
 * 사용자 로그인과 관련된 웹 요청을 처리하는 컨트롤러
 */
@Slf4j
@RestController
@RequestMapping("/api/v1/auth")
public class LoginController {

    private final LoginService loginService;

    @Autowired
    public LoginController(LoginService loginService) { this.loginService = loginService; }




    /**
     * 사용자가 입력한 아이디와 비밀번호로 로그인을 처리
     *
     * @param loginRequest  로그인 폼에서 입력한 데이터 (DTO)
     * @param bindingResult 데이터 바인딩 및 검증 오류 객체
     * @param session       HTTP 세션
     * @return 성공 , 실패에 따라 {errorCode , message} 형식의 JSON 데이터 반환
     */
    @PostMapping("/login")
    public ResponseEntity<?> processLogin(
                                  @Validated @RequestBody LoginRequest loginRequest
                                , BindingResult bindingResult
                                , HttpSession session) {

        if (bindingResult.hasErrors()) {

            FieldError fieldError = bindingResult.getFieldError();

            if (fieldError == null) {
                return ResponseEntity.badRequest().body(Map.of(
                        "errorCode", "VALIDATION_FAILED",
                        "message", "알 수 없는 검증 오류가 발생했습니다."
                ));
            }

            String field = fieldError.getField();
            String errorCode;
            String message = switch (field) {

                case "username" -> {
                    errorCode = "INVALID_USERNAME";
                    yield "아이디 형식이 맞지 않습니다.";
                }
                case "password" -> {
                    errorCode = "INVALID_PASSWORD";
                    yield "비밀번호 형식이 맞지 않습니다.";
                }
                default -> {
                    errorCode = "VALIDATION_FAILED";
                    yield fieldError.getDefaultMessage();
                }

            };

            Map<String, String> errorBody = Map.of(
                    "errorCode", errorCode,
                    "message", message
            );
            return ResponseEntity.badRequest().body(errorBody);


        }

        try{
            Member member = loginService.login(loginRequest);
            session.setAttribute(SessionConst.LOGIN_MEMBER, member);

            Map<String, String> responseBody = Map.of(
                    "message", "로그인에 성공했습니다.",
                    "name", member.getName()
                    // 필요한 다른 정보 추가...
            );

            return ResponseEntity.ok(responseBody);


        }catch (UsernameNotFoundException | IncorrectPasswordException e){
            Map<String, String> errorBody = Map.of(
                    "errorCode", "INVALID_USER",
                    "message", "아이디 또는 비밀번호가 맞지 않습니다."
            );
            return ResponseEntity.badRequest().body(errorBody);
        }


    }
}
