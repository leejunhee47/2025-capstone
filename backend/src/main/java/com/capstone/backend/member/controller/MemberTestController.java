package com.capstone.backend.member.controller;

import com.capstone.backend.member.exceptions.DuplicateUsernameException;
import com.capstone.backend.member.exceptions.IncorrectPasswordException;
import com.capstone.backend.member.exceptions.PasswordMismatchException;
import com.capstone.backend.member.exceptions.UsernameNotFoundException;
import com.capstone.backend.member.model.Member;
import com.capstone.backend.member.request.login.LoginRequest;
import com.capstone.backend.member.request.signup.SignUpRequest;
import com.capstone.backend.member.service.login.LoginService;
import com.capstone.backend.member.service.signup.SignUpService;
import jakarta.servlet.http.HttpSession;
import jakarta.validation.Valid;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ModelAttribute;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;

@Slf4j
@Controller
@RequestMapping("/api/test/member")
public class MemberTestController {

    private final LoginService loginService;
    private final SignUpService signUpService;

    @Autowired
    public MemberTestController(LoginService loginService, SignUpService signUpService) {
        this.loginService = loginService;
        this.signUpService = signUpService;
    }

    @GetMapping("/home")
    public String home() {
        return "test/home";
    }

    @GetMapping("/signup")
    public String signUpForm(Model model) {
        // 기본 생성자가 없으면 여기서 에러가 발생하므로 DTO에 @NoArgsConstructor 추가 필수
        model.addAttribute("signUpRequest", new SignUpRequest());
        return "test/signup";
    }

    @PostMapping("/signup")
    public String signUp(@Valid @ModelAttribute SignUpRequest signUpRequest,
                         BindingResult bindingResult,
                         Model model) {

        // 1. DTO 어노테이션 검증 실패 시 (예: 비밀번호 패턴 불일치)
        if (bindingResult.hasErrors()) {
            log.info("Validation errors: {}", bindingResult.getAllErrors());
            return "test/signup"; // 입력값 유지한 채로 폼으로 돌아감
        }

        try {
            // 2. 서비스 로직 실행 (비즈니스 검증 포함)
            signUpService.signUp(signUpRequest);

            return "redirect:/api/test/member/home";

        } catch (DuplicateUsernameException e) {
            // 특정 필드(username)에 에러를 추가하여 화면에 표시
            bindingResult.rejectValue("username", "duplicate", "이미 사용 중인 아이디입니다.");
            return "test/signup";
        } catch (PasswordMismatchException e) {
            // 특정 필드(passwordConfirm)에 에러 추가
            bindingResult.rejectValue("passwordConfirm", "mismatch", "비밀번호가 일치하지 않습니다.");
            return "test/signup";
        } catch (Exception e) {
            model.addAttribute("errorMessage", "가입 중 오류 발생: " + e.getMessage());
            return "test/signup";
        }
    }


    @GetMapping("/login")
    public String loginForm(Model model) {
        model.addAttribute("loginRequest", new LoginRequest());
        return "test/login";
    }

    @PostMapping("/login")
    public String login(@Valid @ModelAttribute LoginRequest loginRequest,
                        BindingResult bindingResult,
                        HttpSession session, // 로그인 성공 시 세션 저장을 위해 필요
                        Model model) {

        // 1. 입력값 검증 (빈 값, 길이 제한 등)
        if (bindingResult.hasErrors()) {
            return "test/login";
        }

        try {
            // 2. 로그인 서비스 호출
            Member loginMember = loginService.login(loginRequest);

            // 3. 로그인 성공 시 세션에 회원 정보 저장 (key: "loginMember")
            session.setAttribute("loginMember", loginMember);

            log.info("로그인 성공: {}", loginMember.getUsername());
            return "redirect:/api/test/member/home";

        } catch (UsernameNotFoundException e) {
            // 아이디가 없는 경우 -> 아이디 필드에 에러 표시
            bindingResult.rejectValue("username", "notFound", "존재하지 않는 아이디입니다.");
            return "test/login";
        } catch (IncorrectPasswordException e) {
            // 비밀번호 틀린 경우 -> 글로벌 에러 또는 비밀번호 필드 에러 표시
            // 보안상 "아이디 또는 비번이 틀렸습니다"라고 뭉뚱그리기도 하지만, 테스트 단계니 명확히 구분합니다.
            bindingResult.rejectValue("password", "incorrect", "비밀번호가 일치하지 않습니다.");
            return "test/login";
        } catch (Exception e) {
            model.addAttribute("errorMessage", "로그인 중 오류 발생: " + e.getMessage());
            return "test/login";
        }
    }


    @GetMapping("/logout")
    public String logout(HttpSession session) {
        session.invalidate();
        return "redirect:/api/test/member/home";
    }



}
