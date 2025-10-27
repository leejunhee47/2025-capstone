package com.capstone.backend.member.request.signup;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Data;

@Data
public class SignUpRequest {

    @Size(min=4, max=20, message = "{signup.username.size}")
    @Pattern(
            regexp = "^(?=.*[a-z])(?=.*[0-9])[a-z0-9]+$",
            message = "{signup.username.pattern}"
    )
    private String username;


    @Pattern(
            regexp = "^(?=.*[a-zA-Z])(?=.*[0-9])(?=.*[!@#$%^&*]).{8,16}$",
            message = "{signup.password.pattern}"
    )
    private String password;

    @NotBlank(message = "{signup.passwordConfirm.notblank}")
    private String passwordConfirm;

    @Size(min = 2, max = 50, message = "{signup.name.size}")
    @Pattern(regexp = "^[가-힣]+$", message = "{signup.name.pattern}")
    private String name;



    public SignUpRequest(String username, String password, String passwordConfirm, String name) {
        this.username = username;
        this.password = password;
        this.passwordConfirm = passwordConfirm;
        this.name = name;
    }

}
