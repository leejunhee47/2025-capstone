//package com.capstone.backend.core.config;
//
//
//import org.springframework.context.annotation.Configuration;
//import org.springframework.web.servlet.config.annotation.InterceptorRegistry;
//import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;
//
//@Configuration
//public class WebConfig implements WebMvcConfigurer {
//
//    @Override
//    public void addInterceptors(InterceptorRegistry registry){
//        registry.addInterceptor(new LoginCheckerInterceptor())
//                .order(1)
//                .addPathPatterns("/**")
//                .excludePathPatterns("/error" , "/signup/success");
//
//    }
//
//}
