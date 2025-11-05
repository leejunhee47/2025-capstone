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

    @Value("${file.upload-dir}")
    private String uploadDir;

    @Autowired
    public DetectionService(DetectionRequestRepository detectionRequestRepository) {
        this.detectionRequestRepository = detectionRequestRepository;

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



    // 예외처리 작성

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
            managedRequest.setStoredFilePath(dest.getPath());
            managedRequest.setStatus(DetectionStatus.PROCESSING);
            detectionRequestRepository.save(managedRequest);

            // 3. AI 서버에 분석 요청
            // TODO : AI 서버에 분석 요청 로직 작성

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
            managedRequest.setStoredFilePath(outputFilePath);
            managedRequest.setStatus(DetectionStatus.PROCESSING);
            detectionRequestRepository.save(managedRequest);

            // 3. AI 서버에 분석 요청
            // TODO : AI 서버에 분석 요청 로직 작성

        } catch (Exception e) {
            Thread.currentThread().interrupt();
            managedRequest.setStatus(DetectionStatus.FAILED);
            detectionRequestRepository.save(managedRequest);
            throw new RuntimeException(managedRequest.getRequestId() +"", e);
        }
    }


}




