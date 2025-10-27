package com.capstone.backend.member.service.signup.component;

import com.capstone.backend.member.exceptions.DuplicateUsernameException;
import com.capstone.backend.member.exceptions.PasswordMismatchException;
import com.capstone.backend.member.repository.MemberRepository;
import com.capstone.backend.member.request.signup.SignUpRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class SignUpRequestValidator {

    private final MemberRepository memberRepository;

    @Autowired
    public SignUpRequestValidator(MemberRepository memberRepository) {
        this.memberRepository = memberRepository;
    }


    public void validate(SignUpRequest request) throws PasswordMismatchException, DuplicateUsernameException {
        // 1. 비밀번호 일치 여부 검증
        if (request.getPassword() == null || !request.getPassword().equals(request.getPasswordConfirm())) {
            throw new PasswordMismatchException("비밀번호가 일치하지 않습니다.");
        }

        // 2. 아이디(username) 중복 검증
        if (memberRepository.existsByUsername(request.getUsername())) {
            throw new DuplicateUsernameException("이미 존재하는 아이디입니다.");
        }
    }

}
