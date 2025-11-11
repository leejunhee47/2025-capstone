package com.capstone.backend.detection.service;

import com.capstone.backend.detection.model.DetectionRequest;
import com.capstone.backend.detection.model.DetectionStatus;
import com.capstone.backend.detection.repository.DetectionRequestRepository;
import com.capstone.backend.member.model.Member;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;
import java.io.File;
import java.util.concurrent.TimeUnit;

@Service
@Slf4j
public class DetectionService {

    private final DetectionRequestRepository detectionRequestRepository;

    // [유지] /link API가 이 경로를 사용하므로, @Value를 삭제하면 안 됩니다.
    @Value("${file.upload-dir}")
    private String uploadDir;

    @Autowired
    public DetectionService(DetectionRequestRepository detectionRequestRepository) {
        this.detectionRequestRepository = detectionRequestRepository;
    }

    /**
     * 유효성 검사를 통과한 요청을 DB에 'PENDING' 상태로 저장합니다.
     */
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

    // 예외처리 작성

    @Async
    @Transactional
    // [수정] 1. "MultipartFile file"을 "String savedFilePath"로 변경
    // [수정] 2. 메소드 이름 변경 (startDetection -> startFileDetection)
    public void startFileDetection(DetectionRequest request , String savedFilePath) throws IllegalArgumentException{

        DetectionRequest managedRequest = detectionRequestRepository.findById(request.getRequestId())
                .orElse(null);

        if (managedRequest == null) {
            throw new IllegalArgumentException("DetectionRequest not found with ID: " + request.getRequestId());
        }

        try {
            // [삭제] 파일 저장 로직 (컨트롤러로 이동됨)
            
            // 4. DB 갱신
            managedRequest.setStoredFilePath(savedFilePath);
            managedRequest.setStatus(DetectionStatus.PROCESSING);
            detectionRequestRepository.save(managedRequest);

            // 5. AI 서버에 분석 요청
            // TODO : AI 서버에 분석 요청 로직 작성
            // (저장된 파일 경로: savedFilePath 사용)

        }catch (Exception e) {
            log.error("비동기 파일 처리 중 오류 발생 (RequestId: {})", managedRequest.getRequestId(), e);
            managedRequest.setStatus(DetectionStatus.FAILED);
            detectionRequestRepository.save(managedRequest);
            throw new RuntimeException(managedRequest.getRequestId() +"", e);
        }
    }


    @Async
    @Transactional
    // [수정] 2. 메소드 이름 변경 (startDetection -> startUrlDetection)
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

            // 2. DB 갱신
            managedRequest.setStoredFilePath(outputFilePath);
            managedRequest.setStatus(DetectionStatus.PROCESSING);
            detectionRequestRepository.save(managedRequest);

            // 3. AI 서버에 분석 요청
            // TODO : AI 서버에 분석 요청 로직 작성

        } catch (Exception e) {
            Thread.currentThread().interrupt(); // InterruptedException의 경우 스레드 상태 복원
            log.error("비동기 URL 처리 중 오류 발생 (RequestId: {})", request.getRequestId(), e);
            managedRequest.setStatus(DetectionStatus.FAILED);
            detectionRequestRepository.save(managedRequest);
            throw new RuntimeException("Error processing URL for RequestId: " + managedRequest.getRequestId(), e);
        }
    }
}