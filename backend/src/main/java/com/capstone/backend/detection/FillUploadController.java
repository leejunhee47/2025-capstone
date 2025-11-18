package com.capstone.backend.detection;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Controller;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import java.io.File;


@Controller
@Slf4j
public class FillUploadController {

    private final WebClient webClient;

    @Value("${file.upload-dir}")
    private String uploadDir;

    @Value("${ai.url.request}")
    private String aiUrl;

    @Autowired
    public FillUploadController(WebClient webClient) {
        this.webClient = webClient;
    }


    @GetMapping("/upload/test")
    public String showUploadTestPage() {
        return "upload-test"; // "src/main/resources/templates/upload-test.html"을 찾음
    }

    @PostMapping("/upload/video")
    @ResponseBody // 2. 템플릿이 아닌 순수 문자열(Text)을 반환
    public String uploadVideo(@RequestParam("file") MultipartFile file ,
                              @RequestParam("requestId") String requestId) {

        if (file.isEmpty()) {
            return "파일이 비어있습니다.";
        }

        try {
            log.info("1. 파일 저장 시작");

            String originalFilename = file.getOriginalFilename();
            File dest = new File(uploadDir + originalFilename);
            file.transferTo(dest); // 이제 실제로 디스크에 파일이 생겼습니다.


            log.info("2. WebClient 요청 준비");

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("requestId", requestId);
            body.add("file", new FileSystemResource(dest));

            log.info("3. AI 서버로 전송 시작");

            String response = webClient.post()
                    .uri(aiUrl + "test")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(BodyInserters.fromValue(body))
                    .retrieve()
                    .bodyToMono(String.class)
                    .block();

            log.info("4. 전송 완료: " + response);






        } catch (Exception e) {
            log.error("에러 발생", e); // 로그에 에러 스택트레이스 출력
            return "파일 업로드 실패: " + e.getMessage();
        }

        return "성공";
    }
}
