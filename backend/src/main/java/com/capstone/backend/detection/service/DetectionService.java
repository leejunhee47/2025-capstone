package com.capstone.backend.detection.service;

import com.capstone.backend.detection.exceptions.InvalidRequestId;
import com.capstone.backend.detection.model.DetectionRequest;
import com.capstone.backend.detection.model.DetectionStatus;
import com.capstone.backend.detection.repository.DetectionRequestRepository;
import com.capstone.backend.member.model.Member;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.MediaType;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.File;
import java.util.concurrent.TimeUnit;

@Service
@Slf4j
public class DetectionService {

    private final DetectionRequestRepository detectionRequestRepository;
    private final WebClient webClient;

    @Value("${file.upload-dir}")
    private String uploadDir;

    @Value("${ai.url.request}")
    private String aiUrl;

    @Autowired
    public DetectionService(DetectionRequestRepository detectionRequestRepository ,  WebClient webClient) {
        this.detectionRequestRepository = detectionRequestRepository;
        this.webClient = webClient;
    }

    @Transactional
    public DetectionRequest saveNewDetectionRequest(Member member,
                                                    String callbackUrl){
        // (기존과 동일)
        DetectionRequest newRequest = DetectionRequest.builder()
                .member(member)
                .callbackUrl(callbackUrl)
                .build();

        return detectionRequestRepository.save(newRequest);
    }
    @Async
    @Transactional
    public void startFileDetection(DetectionRequest request , String savedFilePath) throws IllegalArgumentException{

        DetectionRequest managedRequest = detectionRequestRepository.findById(request.getRequestId())
                .orElse(null);

        if (managedRequest == null) {
            throw new IllegalArgumentException("DetectionRequest not found with ID: " + request.getRequestId());
        }

        try {

            managedRequest.setVideoPath(savedFilePath);
            managedRequest.setStatus(DetectionStatus.PROCESSING);
            detectionRequestRepository.save(managedRequest);

            // AI 서버에 분석 요청
            log.info("AI 서버 분석 요청 시작. Request ID: {}", managedRequest.getRequestId());

            // HTTP 요청 본문(Body) 생성
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();

            body.add("requestId" , managedRequest.getRequestId()); // 요청 ID
            body.add("file", new FileSystemResource(savedFilePath)); // 파일

            webClient.post()
                    .uri(aiUrl + "request")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(BodyInserters.fromValue(body))
                    .retrieve() // 응답 받기
                    .toBodilessEntity() // 응답 본문은 무시 (상태 코드만 확인)
                    .block(); // @Async 내부이므로 block() 사용 가능

            log.info("AI 서버 분석 요청 성공. Request ID: {}", managedRequest.getRequestId());

        }catch (Exception e) {
            log.error("비동기 파일 처리 중 오류 발생 (RequestId: {})", managedRequest.getRequestId(), e);
            managedRequest.setStatus(DetectionStatus.FAILED);
            detectionRequestRepository.save(managedRequest);
            throw new RuntimeException(managedRequest.getRequestId() +"", e);
        }
    }

    @Async
    @Transactional
    public void startUrlDetection(DetectionRequest request ,  String url) {

        DetectionRequest managedRequest = detectionRequestRepository.findById(request.getRequestId())
                .orElse(null);

        if (managedRequest == null) {
            throw new IllegalArgumentException("DetectionRequest not found with ID: " + request.getRequestId());
        }

        if (url == null || url.trim().isEmpty()) {
            managedRequest.setStatus(DetectionStatus.FAILED);
            detectionRequestRepository.save(managedRequest);
            throw new IllegalArgumentException("URL must not be null or empty for RequestId: " + request.getRequestId());
        }

        try {
            String outputFilePath = uploadDir + File.separator + managedRequest.getRequestId() + ".mp4";
            File downloadedFile = new File(outputFilePath);

            String[] command = {
                    "yt-dlp",
                    "-o",
                    outputFilePath,
                    "--recode-video",
                    "mp4",
                    url
            };

            ProcessBuilder processBuilder = new ProcessBuilder(command);
            processBuilder.redirectErrorStream(true);
            Process process = processBuilder.start();

            if (!process.waitFor(3, TimeUnit.MINUTES)) { // 예: 3분 타임아웃
                process.destroy();
                managedRequest.setStatus(DetectionStatus.FAILED);
                detectionRequestRepository.save(managedRequest);
                throw new RuntimeException("yt-dlp process timed out after 3 minutes for RequestId: " + request.getRequestId());
            }

            int exitCode = process.exitValue();
            if (exitCode != 0) {
                managedRequest.setStatus(DetectionStatus.FAILED);
                detectionRequestRepository.save(managedRequest);
                log.error("yt-dlp process failed (Exit Code: {}) for RequestId: {}", exitCode, request.getRequestId());
                throw new RuntimeException("yt-dlp process failed with exit code: " + exitCode);
            }

            // DB 갱신
            managedRequest.setVideoPath(outputFilePath);
            managedRequest.setStatus(DetectionStatus.PROCESSING);
            detectionRequestRepository.save(managedRequest);

            //  AI 서버에 분석 요청
            // TODO : AI 서버에 분석 요청 로직 작성
            log.info("AI 서버 분석 요청 시작. Request ID: {}", managedRequest.getRequestId());

            // HTTP 요청 본문(Body) 생성
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();

            body.add("requestId" , managedRequest.getRequestId()); // 요청 ID
            body.add("file", new FileSystemResource(downloadedFile)); // 파일

            webClient.post()
                    .uri(aiUrl + "request")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(BodyInserters.fromValue(body))
                    .retrieve() // 응답 받기
                    .toBodilessEntity() // 응답 본문은 무시 (상태 코드만 확인)
                    .block(); // @Async 내부이므로 block() 사용 가능

            log.info("AI 서버 분석 요청 성공. Request ID: {}", managedRequest.getRequestId());

        } catch (Exception e) {
            Thread.currentThread().interrupt(); // InterruptedException의 경우 스레드 상태 복원
            log.error("비동기 URL 처리 중 오류 발생 (RequestId: {})", request.getRequestId(), e);
            managedRequest.setStatus(DetectionStatus.FAILED);
            detectionRequestRepository.save(managedRequest);
            throw new RuntimeException("Error processing URL for RequestId: " + managedRequest.getRequestId(), e);
        }
    }


    @Transactional(readOnly = true)
    public DetectionRequest getDetectionByRequestId(Long requestId) throws InvalidRequestId {
        return detectionRequestRepository.findById(requestId)
                .orElseThrow(() -> new InvalidRequestId("Invalid Request ID: " + requestId));
    }


    @Transactional
    public void updateDetectionStatus(Long requestId, DetectionStatus status) throws InvalidRequestId {
        DetectionRequest request = getDetectionByRequestId(requestId);
        request.setStatus(status);
        detectionRequestRepository.save(request);
    }


    public void testDetection(){

        // 1. HTTP 요청 본문(Body) 생성
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("requestId", "1234");
        // 주의: 실제 파일 전송 테스트 시에는 null 대신 new FileSystemResource(...)를 넣어야 에러가 안 날 수 있습니다.
        body.add("callbackUrl", "hello1234");

        try {
            String response = webClient.post()
                    .uri(aiUrl + "test")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(BodyInserters.fromValue(body))
                    .retrieve()
                    .bodyToMono(String.class)
                    .block();

            // 3. 결과 확인
            System.out.println("응답 결과: " + response);

        } catch (Exception e) {
            // 에러 발생 시 로그 출력
            System.err.println("요청 실패: " + e.getMessage());
        }

    }
}