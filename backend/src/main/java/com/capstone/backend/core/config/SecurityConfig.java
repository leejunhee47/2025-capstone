package com.capstone.backend.core.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;

@Configuration
public class SecurityConfig {

    @Bean
    public PasswordEncoder passwordEncoder(){
        return new BCryptPasswordEncoder();
    }

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
                // 2. (추가) CSRF 보호 기능을 비활성화합니다.
                // JSON API 서버는 세션 쿠키 대신 JWT 토큰 등을 사용하므로 CSRF가 필요 없는 경우가 많습니다.
                .csrf(AbstractHttpConfigurer::disable)

                .authorizeHttpRequests(auth -> auth
                        // "/**" 는 모든 URL 경로를 의미합니다.
                        // 모든 경로에 대한 요청을 허용합니다.
                        .requestMatchers("/**").permitAll()
                );

        return http.build();
    }


}


