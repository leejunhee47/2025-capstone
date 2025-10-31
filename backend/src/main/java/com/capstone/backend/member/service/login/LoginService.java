package com.capstone.backend.member.service.login;


import com.capstone.backend.member.exceptions.IncorrectPasswordException;
import com.capstone.backend.member.exceptions.UsernameNotFoundException;
import com.capstone.backend.member.model.Member;
import com.capstone.backend.member.request.login.LoginRequest;

/**
 * 사용자 로그인 비지니스 로직의 명세 정의
 */
public interface LoginService {
    /**
     * 사용자 아이디와 비밀번호를 받아 인증을 수행합니다.
     *
     * @param loginRequest 로그인에 필요한 사용자 아이디와 비밀번호를 담은 DTO
     * @return 인증에 성공한 회원 엔티티
     * @throws UsernameNotFoundException 주어진 아이디에 해당하는 회원을 찾을 수 없을 경우
     * @throws IncorrectPasswordException 아이디는 존재하지만 비밀번호가 일치하지 않을 경우
     */
    Member login(LoginRequest loginRequest) throws UsernameNotFoundException, IncorrectPasswordException;
}
