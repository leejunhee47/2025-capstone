package com.capstone.backend.member.request.login;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class LoginRequest {

    @NotBlank(message = "아이디를 입력해주세요.")
    @Size(min=4,max=20 , message = "아이디는 4자 이상 20자 이하로 입력해주세요.")
    private String username;

    @NotBlank(message = "비밀번호를 입력해주세요.")
    private String password;


    public LoginRequest(String username, String password) {
        this.username = username;
        this.password = password;
    }
}

