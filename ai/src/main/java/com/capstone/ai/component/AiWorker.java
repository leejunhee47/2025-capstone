package com.capstone.ai.component;

import com.capstone.ai.request.DetectionRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*; // HttpHeaders, HttpEntity, MediaType, ResponseEntity 임포트
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate; // RestTemplate 임포트

import java.io.File;
import java.nio.file.Files; // Files 임포트
import java.nio.file.Paths; // Paths 임포트

@Component
@Slf4j
public class AiWorker {

    private final AiJobQueue aiJobQueue;
    private final RestTemplate restTemplate; // (추가) POST 요청을 위한 RestTemplate

    @Value("${ai.script}")
    private String scriptPath;

    @Value("${ai.result}")
    private String resultPath;

    @Value("${backend.url.request}")
    private String backendUrl;


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
        runPythonScript(detectionRequest);
    }

    private void runPythonScript(DetectionRequest request) {

        // 특정 파일 결과 저장 경로
        String ThatResultPath = resultPath + File.separator + request.getRequestId() + ".json";

        try {
            ProcessBuilder processBuilder = new ProcessBuilder(
                    "python",  // 혹은 "python3" (환경에 따라 수정)
                    scriptPath,
                    ThatResultPath,
                    request.getVideoPath(),
                    request.getRequestId()
            );

            processBuilder.redirectOutput(ProcessBuilder.Redirect.INHERIT);
            processBuilder.redirectError(ProcessBuilder.Redirect.INHERIT);

            Process process = processBuilder.start();
            int exitCode = process.waitFor();

            if (exitCode == 0) {
                log.info("AI 분석 정상 종료: ID={}", request.getRequestId());

                // -----------------------------------------------------------------
                // (수정) TODO 구현 부분
                // -----------------------------------------------------------------
                try {
                    // 1. 결과 JSON 파일을 String으로 읽기
                    String jsonResult = Files.readString(Paths.get(ThatResultPath));

                    // 2. HTTP 요청 헤더 설정 (Content-Type: application/json)
                    HttpHeaders headers = new HttpHeaders();
                    headers.setContentType(MediaType.APPLICATION_JSON);

                    // 3. HTTP 요청 본문(Body) 구성
                    HttpEntity<String> entity = new HttpEntity<>(jsonResult, headers);

                    // 4. POST 요청 전송
                    String targetUrl = backendUrl + "/result";
                    ResponseEntity<String> response = restTemplate.postForEntity(
                            targetUrl,
                            entity,
                            String.class // 백엔드 응답은 String으로 받음
                    );

                    // 5. 백엔드 응답 로깅
                    if (response.getStatusCode().is2xxSuccessful()) {
                        log.info("백엔드로 결과 전송 성공: ID={}, Status={}", request.getRequestId(), response.getStatusCode());

                        // (선택 사항) 성공적으로 보냈으니 임시 JSON 파일 삭제
                        Files.deleteIfExists(Paths.get(ThatResultPath));
                    } else {
                        log.warn("백엔드 전송 실패 (비-2xx 응답): ID={}, Status={}", request.getRequestId(), response.getStatusCode());
                    }

                } catch (Exception e) {
                    log.error("AI 성공 후, 백엔드 전송/파일처리 중 예외 발생: ID={}", request.getRequestId(), e);
                }
                // -----------------------------------------------------------------

            } else {
                log.error("AI 분석 비정상 종료 (Exit Code: {}): ID={}", exitCode, request.getRequestId());
                // (참고) 실패 시에도 임시 오류 JSON 파일이 남아있을 수 있습니다.
            }

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.error("AI 실행 중 중단됨", e);
        } catch (Exception e) {
            log.error("AI 실행 중 오류 발생", e);
        }
    }
}