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
     * @param originalFilename 사용자가 업로드한 원본 파일명
     * @param fileFormat 파일 포맷 (MIME Type)
     * @param fileSize 파일 크기 (bytes)
     * @return 저장된 DetectionRequest 엔티티
     */
    @Transactional
    public DetectionRequest saveNewDetectionRequest(Member member,
                                                    String callbackUrl,
                                                    String originalFilename,
                                                    String fileFormat,
                                                    Long fileSize) {


        DetectionRequest newRequest = DetectionRequest.builder()
                .member(member)
                .callbackUrl(callbackUrl)
                .originalFilename(originalFilename)
                .fileFormat(fileFormat)
                .fileSize(fileSize)
                .build();

        return detectionRequestRepository.save(newRequest);
    }




    @Async
    @Transactional
    public void startDetection(DetectionRequest request , MultipartFile file) {

        // 다시 조회
        DetectionRequest managedRequest = detectionRequestRepository.findById(request.getRequestId())
                .orElse(null);

        if (managedRequest == null) {
            log.error("비동기 처리 시작 실패: RequestId {} 를 찾을 수 없습니다.", request.getRequestId());
            return;
        }

        String uniqueFileName = managedRequest.getRequestId() + "_" + file.getOriginalFilename();
        File dest = new File(uploadDir + uniqueFileName);

        try {
            // 1. 파일 저장
            log.info("[Async] 파일 저장 시작: {}", dest.getPath());
            file.transferTo(dest);
            log.info("[Async] 파일 저장 성공: {}", dest.getPath());

            // 2. DB 갱신
            managedRequest.setStoredFilePath(dest.getPath());
            managedRequest.setStatus(DetectionStatus.PROCESSING);
            detectionRequestRepository.save(managedRequest);

            // 3. AI 서버에 분석 요청
            // TODO : AI 서버에 분석 요청 로직 작성

        }catch (Exception e) {

            log.error("[Async] 파일 저장 실패: {}", e.getMessage());
            managedRequest.setStatus(DetectionStatus.FAILED);
            detectionRequestRepository.save(managedRequest);

        }
    }

}

