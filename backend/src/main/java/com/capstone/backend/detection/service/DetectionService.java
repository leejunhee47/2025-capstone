package com.capstone.backend.detection.service;

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
import org.springframework.web.multipart.MultipartFile;
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

    @Value("${ai.url.response}")
    private String aiCallbackUrl;



    @Autowired
    public DetectionService(DetectionRequestRepository detectionRequestRepository , WebClient webClient) {
        this.detectionRequestRepository = detectionRequestRepository;
        this.webClient = webClient;

    }

    /**
     * 유효성 검사를 통과한 요청을 DB에 'PENDING' 상태로 저장합니다.
     * @param member 요청을 보낸 사용자
     * @param callbackUrl 결과를 받을 URL
     * @return 저장된 DetectionRequest 엔티티
     */
    @Transactional
    public DetectionRequest saveNewDetectionRequest(Member member,
                                                    String callbackUrl){

        DetectionRequest newRequest = DetectionRequest.builder()
                .member(member)
                .callbackUrl(callbackUrl)
                .build();

        return detectionRequestRepository.save(newRequest);
    }



    // todo :  예외처리 작성

    @Async
    @Transactional
    public void startDetection(DetectionRequest request , MultipartFile file) throws IllegalArgumentException{

        DetectionRequest managedRequest = detectionRequestRepository.findById(request.getRequestId())
                .orElse(null);

        if (managedRequest == null) {
            throw new IllegalArgumentException("DetectionRequest not found with ID: " + request.getRequestId());
        }


        // TODO : 파일 경로 수정 필요
        String uniqueFileName = managedRequest.getRequestId() + "";
        File dest = new File(uploadDir + uniqueFileName);

        try {
            // 1. 파일 저장
            file.transferTo(dest);

            // 2. DB 갱신
            managedRequest.setVideoPath(dest.getPath());
            managedRequest.setStatus(DetectionStatus.PROCESSING);
            detectionRequestRepository.save(managedRequest);

            // 3. AI 서버에 분석 요청
            
            log.info("AI 서버 분석 요청 시작. Request ID: {}", managedRequest.getRequestId());

            // HTTP 요청 본문(Body) 생성
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();

            body.add("requestId" , managedRequest.getRequestId()); // 요청 ID
            body.add("file", new FileSystemResource(dest)); // 파일
            body.add("callbackUrl", aiCallbackUrl); // 콜백 URL

            webClient.post()
                    .uri(aiUrl + "/analyze") // AI 서버의 엔드포인트 (예: /analyze, /predict 등)
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(BodyInserters.fromValue(body))
                    .retrieve() // 응답 받기
                    .toBodilessEntity() // 응답 본문은 무시 (상태 코드만 확인)
                    .block(); // @Async 내부이므로 block() 사용 가능

            log.info("AI 서버 분석 요청 성공. Request ID: {}", managedRequest.getRequestId());





        }catch (Exception e) {
            managedRequest.setStatus(DetectionStatus.FAILED);
            detectionRequestRepository.save(managedRequest);
            throw new RuntimeException(managedRequest.getRequestId() +"", e);
        }
    }


    @Async
    @Transactional
    public void startDetection(DetectionRequest request ,  String url) {


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
                throw new RuntimeException("yt-dlp process failed with exit code: " + exitCode);
            }


            // 2. DB 갱신
            managedRequest.setVideoPath(outputFilePath);
            managedRequest.setStatus(DetectionStatus.PROCESSING);
            detectionRequestRepository.save(managedRequest);

            // 3. AI 서버에 분석 요청


            log.info("AI 서버 분석 요청 시작. Request ID: {}", managedRequest.getRequestId());

            // HTTP 요청 본문(Body) 생성
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();

            body.add("requestId" , managedRequest.getRequestId()); // 요청 ID
            body.add("file", new FileSystemResource(downloadedFile)); // 파일
            body.add("callbackUrl",aiCallbackUrl); // 콜백 URL

            webClient.post()
                    .uri(aiUrl + "/analyze") // AI 서버의 엔드포인트 (예: /analyze, /predict 등)
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(BodyInserters.fromValue(body))
                    .retrieve() // 응답 받기
                    .toBodilessEntity() // 응답 본문은 무시 (상태 코드만 확인)
                    .block(); // @Async 내부이므로 block() 사용 가능

            log.info("AI 서버 분석 요청 성공. Request ID: {}", managedRequest.getRequestId());


        } catch (Exception e) {
            Thread.currentThread().interrupt();
            managedRequest.setStatus(DetectionStatus.FAILED);
            detectionRequestRepository.save(managedRequest);
            throw new RuntimeException(managedRequest.getRequestId() +"", e);
        }
    }


    @Transactional
    public void saveResult(){

        // todo : 결과 형식이 결정 되고 , DTO 만든후 작성
    }



}




