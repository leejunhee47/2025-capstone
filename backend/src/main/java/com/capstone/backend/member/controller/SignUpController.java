package com.capstone.backend.member.controller;


import com.capstone.backend.member.exceptions.DuplicateUsernameException;
import com.capstone.backend.member.exceptions.PasswordMismatchException;
import com.capstone.backend.member.request.signup.SignUpRequest;
import com.capstone.backend.member.service.signup.SignUpService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.BindingResult;
import org.springframework.validation.FieldError;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

/**
 * 회원 가입과 관련된 웹 요청을 처리하는 컨트롤러
 */
@Slf4j
@RestController
@RequestMapping("/api/v1")
public class SignUpController {

    private final SignUpService signUpService;

    @Autowired
    public SignUpController(SignUpService signUpService) {
        this.signUpService = signUpService;
    }

    /**
     * 사용자가 입력한 정보로 회원가입을 처리
     * @param signUpRequest 회원가입 폼에서 입력한 데이터 (DTO)
     * @param bindingResult 데이터 바인딩 및 검증 오류 객체
     * @return 성공 , 실패에 따라 {errorCode , message} 형식의 JSON 데이터 반환
     */
    @PostMapping("/users")
    public ResponseEntity<?> processSignUp(@Validated @RequestBody SignUpRequest signUpRequest , BindingResult bindingResult){


        if(bindingResult.hasErrors()) {

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
                case "name" -> {
                    errorCode = "INVALID_NAME";
                    yield "이름 형식이 맞지 않습니다.";
                }
                case "username" -> {
                    errorCode = "INVALID_USERNAME";
                    yield "아이디 형식이 맞지 않습니다.";
                }
                case "password" -> {
                    errorCode = "INVALID_PASSWORD";
                    yield "비밀번호 형식이 맞지 않습니다.";
                }
                case "passwordConfirm" -> {
                    errorCode = "VALIDATION_FAILED";
                    yield "비밀번호 확인을 입력해주세요.";
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

        try {
            signUpService.signUp(signUpRequest);


        }catch (DuplicateUsernameException e){

            Map<String, String> errorBody = Map.of(
                    "errorCode", "DUPLICATE_USERNAME",
                    "message", "아이디가 중복 됩니다."
            );
            // 409 Conflict: 리소스 충돌 (중복된 ID)
            return ResponseEntity.status(HttpStatus.CONFLICT).body(errorBody);

        }catch(PasswordMismatchException e){

            Map<String, String> errorBody = Map.of(
                    "errorCode", "PASSWORD_MISMATCH",
                    "message", "비밀번호와 비밀번호 확인이 일치하지 않습니다."
            );

            return ResponseEntity.badRequest().body(errorBody);
        }

        return ResponseEntity.status(HttpStatus.CREATED).body(Map.of(
                "message", "회원가입이 성공적으로 완료되었습니다."
        ));
    }
}


