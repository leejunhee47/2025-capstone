package com.capstone.backend.detection;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.io.File;
import java.io.IOException;

@Controller
public class FillUploadController {

    // 파일을 저장할 로컬 경로
    private final String uploadDir = "/Users/woo-in/UploadFileTest/";

    @GetMapping("/upload/test")
    public String showUploadTestPage() {
        return "upload-test"; // "src/main/resources/templates/upload-test.html"을 찾음
    }

    @PostMapping("/upload/video")
    @ResponseBody // 2. 템플릿이 아닌 순수 문자열(Text)을 반환
    public String uploadVideo(@RequestParam("file") MultipartFile file) {

        if (file.isEmpty()) {
            return "파일이 비어있습니다.";
        }

        try {
            String originalFilename = file.getOriginalFilename();
            File dest = new File(uploadDir + originalFilename);

            // 파일 저장
            file.transferTo(dest);


            return "파일 업로드 성공: " + originalFilename;

        } catch (IOException e) {
            return "파일 업로드 실패: " + e.getMessage();
        }
    }





}
