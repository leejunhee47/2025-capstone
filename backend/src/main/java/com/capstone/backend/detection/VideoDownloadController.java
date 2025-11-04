package com.capstone.backend.detection;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

@Controller
public class VideoDownloadController {

    private static final Logger logger = Logger.getLogger(VideoDownloadController.class.getName());

    // 저장할 경로 (FillUploadController에서 사용한 경로와 동일)
    private final String saveDir = "/Users/woo-in/UploadFileTest";

    /**
     * 1. URL 입력 폼을 보여주는 페이지 (GET)
     * http://localhost:8080/download/form
     */
    @GetMapping("/download/form")
    public String showDownloadForm() {
        return "download-test"; // /resources/templates/download-test.html
    }

    /**
     * 2. URL을 받아 터미널 명령을 실행하는 컨트롤러 (POST)
     * http://localhost:8080/download/process
     */
    @PostMapping("/download/process")
    @ResponseBody
    public String processVideoUrl(@RequestParam("url") String url) {

        if (url == null || url.trim().isEmpty()) {
            return "URL이 비어있습니다.";
        }

        try {
            // 1. 실행할 터미널 명령 정의
            // -P : 저장 경로 지정
            // --recode-video mp4 : mp4로 인코딩 (필요시)
            String[] command = {
                "yt-dlp",
                "-P",
                saveDir,
                "--recode-video",
                "mp4",
                url
            };

            // 2. 프로세스 빌더 생성
            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.redirectErrorStream(true); // 에러 스트림을 표준 출력 스트림으로 합침

            // 3. 프로세스 실행
            Process process = processBuilder.start();

            // 4. (중요) 프로세스가 완료될 때까지 대기
            //    (타임아웃을 설정하여 무한 대기 방지)
            if (!process.waitFor(5, TimeUnit.MINUTES)) { // 예: 5분 타임아웃
                 process.destroy();
                 logger.warning("다운로드 시간 초과: " + url);
                 return "실패: 다운로드 시간이 초과되었습니다.";
            }

            // 5. 프로세스 실행 결과 (로그) 읽기
            StringBuilder output = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append("\n");
                }
            }

            // 6. 종료 코드 확인
            int exitCode = process.exitValue();
            if (exitCode == 0) {
                logger.info("다운로드 성공: " + url);
                return "다운로드 성공. 로그:\n" + output.toString();
            } else {
                logger.warning("다운로드 실패 (Exit Code: " + exitCode + "): " + url + "\n" + output.toString());
                return "다운로드 실패. 로그:\n" + output.toString();
            }

        } catch (IOException | InterruptedException e) {
            Thread.currentThread().interrupt(); // InterruptedException 처리
            logger.severe("프로세스 실행 중 예외 발생: " + e.getMessage());
            return "실패: 서버 내부 오류가 발생했습니다. (" + e.getMessage() + ")";
        }

    }



}
