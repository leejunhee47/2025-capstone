package com.capstone.backend.member.service.signup;


import com.capstone.backend.member.exceptions.DuplicateUsernameException;
import com.capstone.backend.member.exceptions.PasswordMismatchException;
import com.capstone.backend.member.model.Member;
import com.capstone.backend.member.request.signup.SignUpRequest;

/**
 * 신규 회원 가입 비즈니스 로직의 명세를 정의합니다.
 */
public interface SignUpService {

    /**
     * 사용자가 입력한 정보를 바탕으로 회원 가입을 처리합니다.
     * 이 과정에서 아이디 중복 검사, 비밀번호 일치 여부 확인 유효성 검사를 수행합니다.
     *
     * @param signUpRequest 회원 가입에 필요한 사용자 정보를 담은 DTO
     * @return 성공적으로 생성된 회원 엔티티
     * @throws DuplicateUsernameException 입력된 사용자 아이디(username)가 이미 존재할 경우
     * @throws PasswordMismatchException 비밀번호와 비밀번호 확인 필드가 일치하지 않을 경우
     */
    void signUp(SignUpRequest signUpRequest) throws DuplicateUsernameException, PasswordMismatchException;
}
