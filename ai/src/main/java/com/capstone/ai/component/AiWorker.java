package com.capstone.ai.component;

import com.capstone.ai.request.DetectionRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.*; // HttpHeaders, HttpEntity, MediaType, ResponseEntity 임포트
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate; // RestTemplate 임포트

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files; // Files 임포트
import java.util.ArrayList;
import java.util.List;

@Slf4j
@Component
public class AiWorker {

    private final AiJobQueue aiJobQueue;
    private final RestTemplate restTemplate; // (추가) POST 요청을 위한 RestTemplate

    @Value("${ai.script}")
    private String scriptPath;

    @Value("${ai.script.image}")
    private String imageScriptPath;

    @Value("${ai.result}")
    private String resultPath;

    @Value("${ai.result.img}")
    private String resultImgPath;

    @Value("${backend.url.request}")
    private String backendUrl;

    @Value("${ai.result.example1}")
    private String resultImgResult1;

    @Value("${ai.result.example2}")
    private String resultImgResult2;

    @Value("${ai.model.execute.path}")
    private String modelExecutePath;


    @Autowired
    AiWorker(AiJobQueue aiJobQueue, RestTemplate restTemplate) { // (수정) RestTemplate 주입
        this.aiJobQueue = aiJobQueue;
        this.restTemplate = restTemplate; // (추가)
    }

    // 1초(1000ms)마다 실행 (큐에 일감이 있는지 확인)
    @Scheduled(fixedDelay = 1000)
    public void checkQueueAndRunAi() {

        // 1. 큐에서 일감을 하나 꺼냄
        DetectionRequest detectionRequest = aiJobQueue.takeJob();

        // 2. 일감이 없으면(null) 그냥 넘어감
        if (detectionRequest == null) {
            return;
        }

        // 3. 일감이 있으면 AI 스크립트 실행 (여기서 실행 끝날 때까지 대기함)
        log.info("AI 분석 시작: ID={}, 파일={}", detectionRequest.getRequestId(), detectionRequest.getVideoPath());
        // Todo : fast api 호출 (더 빠름)
        processRequest(detectionRequest);
    }

    private void processRequest(DetectionRequest request) {
        log.info("AI 모델 프로세스 실행 시작: RequestID={}, Video={}", request.getRequestId(), request.getVideoPath());

        try {
            // 1. 명령어 및 인자 구성 (상대 경로 사용)
            List<String> command = new ArrayList<>();
            command.add("python");
            command.add("-m");
            command.add("src.xai.hybrid_pipeline"); // 모듈 실행 방식

            command.add("--video");
            command.add(request.getVideoPath()); // 요청받은 비디오 경로

            command.add("--mmms-model");
            command.add("models/checkpoints/mmms-ba_fulldata_best.pth"); // 루트 기준 상대경로

            command.add("--pia-model");
            command.add("models/checkpoints/pia-best.pth"); // 루트 기준 상대경로

            command.add("--output-dir");
            // 결과가 저장될 폴더 (고정 경로 사용 시 주의: 동시 요청 시 덮어쓰기 될 수 있음)
            command.add("outputs/xai/hybrid/demo_run");

            command.add("--device");
            command.add("cuda");

            // 2. ProcessBuilder 설정
            ProcessBuilder pb = new ProcessBuilder(command);

            // ★ 핵심 1: 작업 디렉토리를 프로젝트 루트로 고정
            // application.properties의 ai.model.execute.path 값이 정확해야 합니다.
            pb.directory(new File(modelExecutePath));

            // ★ 핵심 2: PYTHONPATH를 현재 위치(.)로 설정하여 모듈 Import 에러 방지
            pb.environment().put("PYTHONPATH", ".");

            // 표준 에러(stderr)를 표준 출력(stdout)으로 리다이렉트 (로그 보기 편함)
            pb.redirectErrorStream(true);

            // 3. 프로세스 시작
            Process process = pb.start();

            // 4. 프로세스 로그 실시간 출력 (버퍼링 방지용)
            // 이걸 안 해주면 버퍼가 꽉 차서 프로세스가 멈출(Hang) 수 있습니다.
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    log.info("[Python AI] " + line);
                }
            }

            // 5. 종료 대기
            int exitCode = process.waitFor();
            log.info("AI 프로세스 종료 코드: {}", exitCode);

            if (exitCode == 0) {
                // 성공 시 결과 파일 처리 로직 호출
                handleAiSuccess(request);
            } else {
                log.error("AI 프로세스가 비정상 종료되었습니다 (Exit Code: {}).", exitCode);
                // 필요 시 실패 처리 로직 추가
            }

        } catch (Exception e) {
            log.error("AI 프로세스 실행 중 예외 발생", e);
            Thread.currentThread().interrupt();
        }
    }

// ================= [새로 추가된 메서드] =================
    /**
     * AI 분석 완료 후 결과 파일(JSON, 이미지)을 찾아 백엔드로 전송
     */
    private void handleAiSuccess(DetectionRequest request) {
        try {
            // 1. 영상 파일명 추출 (확장자 제거)
            // 예: /path/to/video_123.mp4 -> video_123
            File videoFile = new File(request.getVideoPath());
            String fileName = videoFile.getName();
            int dotIndex = fileName.lastIndexOf('.');
            String baseName = (dotIndex == -1) ? fileName : fileName.substring(0, dotIndex);

            // 2. 결과 파일 경로 구성
            // modelExecutePath + outputs/xai/hybrid/demo_run
            File outputDir = new File(modelExecutePath, "outputs/xai/hybrid/demo_run");

            // 파일 1: JSON 결과
            File resultJsonFile = new File(outputDir, "result.json");

            // 파일 2: 이미지 결과 (파일명 규칙 적용)
            // 규칙 1: [영상 이름]_pia_xai.png
            File img1 = new File(outputDir, baseName + "_pia_xai.png");
            // 규칙 2: [영상 이름]_stage1_timeline.png
            File img2 = new File(outputDir, baseName + "_stage1_timeline.png");

            // 3. 파일 존재 여부 확인 및 전송
            if (resultJsonFile.exists() && img1.exists() && img2.exists()) {
                // JSON 파일 내용 읽기
                String jsonContent = Files.readString(resultJsonFile.toPath(), StandardCharsets.UTF_8);

                log.info("결과 파일 로드 성공. 백엔드 전송 시작. RequestId={}", request.getRequestId());

                // 백엔드 전송 메서드 호출
                sendResultToBackend(request.getRequestId(), jsonContent, img1, img2);
            } else {
                log.error("결과 파일 중 일부가 존재하지 않습니다. 경로를 확인하세요.");
                log.error("JSON: {}, Exists: {}", resultJsonFile.getPath(), resultJsonFile.exists());
                log.error("Img1: {}, Exists: {}", img1.getPath(), img1.exists());
                log.error("Img2: {}, Exists: {}", img2.getPath(), img2.exists());
            }

        } catch (IOException e) {
            log.error("결과 파일 읽기 실패: {}", e.getMessage());
        }
    }

    private void sendResultToBackend(String requestId, String jsonResult, File img1, File img2) {
        try {
            HttpHeaders mainHeaders = new HttpHeaders();
            mainHeaders.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();

            HttpHeaders jsonHeaders = new HttpHeaders();
            jsonHeaders.setContentType(MediaType.APPLICATION_JSON);
            HttpEntity<String> jsonEntity = new HttpEntity<>(jsonResult, jsonHeaders);

            body.add("result", jsonEntity);
            body.add("images", new FileSystemResource(img1));
            body.add("images", new FileSystemResource(img2));

            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, mainHeaders);

            String targetUrl = backendUrl.endsWith("/") ? backendUrl + "result" : backendUrl + "/result";

            ResponseEntity<String> response = restTemplate.postForEntity(
                    targetUrl,
                    requestEntity,
                    String.class
            );

            if (response.getStatusCode().is2xxSuccessful()) {
                log.info("백엔드 전송 성공 [RequestId: {}]", requestId);
            } else {
                log.warn("백엔드 전송 실패 Status: {}", response.getStatusCode());
            }

        } catch (Exception e) {
            log.error("백엔드 통신 오류 [RequestId: {}] : {}", requestId, e.getMessage());
        }
    }


}