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
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files; // Files 임포트

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
        // 1. 기존 분석 스크립트 (JSON 생성)
        boolean jsonSuccess = runDetectionScript(request);
        if (!jsonSuccess) return;

        // 2. (추가) 이미지 생성/저장 스크립트 실행
        boolean imgSuccess = runImageScript(request);
        if (!imgSuccess) return;

        // 3. 결과 파일들 읽기 및 전송 준비
        String outputJsonPath = resultPath + File.separator + request.getRequestId() + ".json";
        String outputImgPath1 = resultImgPath + File.separator + request.getRequestId() + "_1.jpg";
        String outputImgPath2 = resultImgPath + File.separator + request.getRequestId() + "_2.jpg";

        try {
            File jsonFile = new File(outputJsonPath);
            File imgFile1 = new File(outputImgPath1);
            File imgFile2 = new File(outputImgPath2);

            if (jsonFile.exists() && imgFile1.exists() && imgFile2.exists()) {
                String jsonResult = Files.readString(jsonFile.toPath(), StandardCharsets.UTF_8);

                // 4. 백엔드로 JSON + 이미지 전송
                sendResultToBackend(request.getRequestId(), jsonResult, imgFile1, imgFile2);

                // 5. 임시 파일 삭제 (청소)
                jsonFile.delete();
                imgFile1.delete();
                imgFile2.delete();
                log.info("임시 결과 파일(JSON/Images) 삭제 완료: ID={}", request.getRequestId());
            } else {
                log.error("결과 파일 중 일부가 누락되었습니다. ID={}", request.getRequestId());
            }

        } catch (Exception e) {
            log.error("결과 처리 중 오류 발생", e);
        }
    }

    private boolean runDetectionScript(DetectionRequest request) {
        String outputJsonPath = resultPath + File.separator + request.getRequestId() + ".json";
        try {
            ProcessBuilder processBuilder = new ProcessBuilder(
                    "python", scriptPath, outputJsonPath, request.getVideoPath(), request.getRequestId()
            );
            processBuilder.redirectOutput(ProcessBuilder.Redirect.INHERIT);
            processBuilder.redirectError(ProcessBuilder.Redirect.INHERIT);
            Process process = processBuilder.start();
            int exitCode = process.waitFor();
            return exitCode == 0;
        } catch (Exception e) {
            log.error("AI 분석 스크립트 실행 오류", e);
            return false;
        }
    }


    private boolean runImageScript(DetectionRequest request) {
        try {
            log.info("이미지 생성 스크립트 실행: ID={}", request.getRequestId());

            ProcessBuilder processBuilder = new ProcessBuilder(
                    "python",
                    imageScriptPath,        // 새로 만든 파이썬 스크립트 경로
                    request.getRequestId(), // 인자 1
                    resultImgResult1,       // 인자 2 (예제 이미지1)
                    resultImgResult2,       // 인자 3 (예제 이미지2)
                    resultImgPath           // 인자 4 (저장 폴더)
            );

            processBuilder.redirectOutput(ProcessBuilder.Redirect.INHERIT);
            processBuilder.redirectError(ProcessBuilder.Redirect.INHERIT);

            Process process = processBuilder.start();
            int exitCode = process.waitFor();

            if (exitCode == 0) {
                log.info("이미지 생성 완료: ID={}", request.getRequestId());
                return true;
            } else {
                log.error("이미지 스크립트 실패 (Exit Code: {})", exitCode);
                return false;
            }
        } catch (Exception e) {
            log.error("이미지 스크립트 실행 중 예외 발생", e);
            return false;
        }
    }



    private void sendResultToBackend(String requestId, String jsonResult, File img1, File img2) {
        try {
            // 1. 전체 요청의 헤더 (Multipart Form Data)
            HttpHeaders mainHeaders = new HttpHeaders();
            mainHeaders.setContentType(MediaType.MULTIPART_FORM_DATA);

            // Body 설정
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();

            // ================= [수정된 부분 시작] =================
            // 2. JSON 데이터 추가 시 헤더(application/json)를 명시해야 함
            HttpHeaders jsonHeaders = new HttpHeaders();
            jsonHeaders.setContentType(MediaType.APPLICATION_JSON);

            // 내용물(jsonResult)과 헤더(jsonHeaders)를 묶어서 HttpEntity로 만듦
            HttpEntity<String> jsonEntity = new HttpEntity<>(jsonResult, jsonHeaders);

            body.add("result", jsonEntity);
            // ================= [수정된 부분 끝] ===================

            // 3. 이미지 파일 추가 (FileSystemResource 사용)
            body.add("images", new FileSystemResource(img1));
            body.add("images", new FileSystemResource(img2));

            // 전체 요청 엔티티 생성
            HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, mainHeaders);

            String targetUrl = backendUrl.endsWith("/") ? backendUrl + "result" : backendUrl + "/result";
            log.info("백엔드 결과 전송 시도 [RequestId: {}] -> {}", requestId, targetUrl);

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