package com.capstone.backend.detection.controller;

import com.capstone.backend.core.common.SessionConst;
import com.capstone.backend.detection.model.DetectionRequest;
import com.capstone.backend.detection.response.DetectionResponse;
import com.capstone.backend.detection.response.ErrorResponse;
import com.capstone.backend.detection.service.DetectionService;
import com.capstone.backend.member.model.Member;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.validator.routines.UrlValidator;
import org.apache.tika.Tika;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value; // @Value 임포트
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.io.*;

@Slf4j
@RestController
@RequestMapping("/api/v1/detection")
public class DetectionController {

    private final Tika tika = new Tika();
    private final DetectionService detectionService;

    // [추가] 파일 저장을 위해 컨트롤러에도 uploadDir 주입
    @Value("${file.upload-dir}")
    private String uploadDir;

    @Autowired
    public DetectionController(DetectionService detectionService) { this.detectionService = detectionService; }

    @PostMapping("/upload")
    public ResponseEntity<?> processUploadDetectionRequest(@RequestParam("file") MultipartFile file ,
                                                           @RequestParam("callbackUrl") String callbackUrl ,
                                                           @SessionAttribute(value = SessionConst.LOGIN_MEMBER, required = false) Member loginMember){

        // 1. 세션 유효성 검사
        if (loginMember == null) {
            ErrorResponse errorResponse = new ErrorResponse(
                    "UNAUTHORIZED",
                    "로그인이 필요합니다. (세션이 만료되었거나, 로그인하지 않은 사용자입니다.)"
            );
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(errorResponse);
        }

        // 1-1. URL 유효성 체크
        if(!isValidCallbackUrl(callbackUrl)){
            ErrorResponse errorResponse = new ErrorResponse(
                    "INVALID_CALLBACK_URL",
                    "유효하지 않은 콜백 URL 입니다. (http/https 형식이 아니거나, 내부망/localhost 주소일 수 없습니다.)"
            );
            return ResponseEntity.badRequest().body(errorResponse);
        }

        // 1-2. 영상 유효성 체크
        String mimeType = validateAndGetMimeType(file);
        if (mimeType == null || !mimeType.startsWith("video/")) {
            log.warn("유효하지 않은 파일 수신: {}", file.getOriginalFilename());
            ErrorResponse errorResponse = new ErrorResponse(
                    "INVALID_FILE_TYPE",
                    "유효하지 않은 파일이거나 비디오 파일(mp4, mov 등)이 아닙니다."
            );
            return ResponseEntity.badRequest().body(errorResponse);
        }

        // 2. DB 에 요청 저장 및 파일 저장
        DetectionRequest newRequest;
        File dest; 

        try {
            // 2-1. DB에 먼저 저장
            newRequest = detectionService.saveNewDetectionRequest(loginMember, callbackUrl);

            // 2-2. 파일 저장 로직
            String originalFilename = file.getOriginalFilename() != null ? file.getOriginalFilename() : "video.mp4";
            String uniqueFileName = newRequest.getRequestId() + "_" + originalFilename;
            dest = new File(uploadDir + File.separator + uniqueFileName);

            // 2-3. 파일 저장 (동기식으로 먼저 실행)
            file.transferTo(dest);

            // 3. 비동기 처리 시작
            // [수정] 3.1 메소드 이름 변경 (startDetection -> startFileDetection)
            detectionService.startFileDetection(newRequest , dest.getPath());

        } catch (Exception e) {
            log.error("파일 저장 또는 비동기 처리 시작 중 오류 발생", e);
            ErrorResponse errorResponse = new ErrorResponse(
                    "SERVER_ERROR",
                    "서버 에러 입니다."
            );
            return ResponseEntity
                    .status(HttpStatus.INTERNAL_SERVER_ERROR) // 500
                    .body(errorResponse);
        }

        // 사용자에게 즉시 응답
        DetectionResponse response = new DetectionResponse(newRequest.getRequestId());
        return ResponseEntity.accepted().body(response);
    }

    @PostMapping("/link")
    public ResponseEntity<?> processLinkDetectionRequest(@RequestParam("videoUrl") String videoUrl ,
                                                           @RequestParam("callbackUrl") String callbackUrl ,
                                                           @SessionAttribute(value = SessionConst.LOGIN_MEMBER, required = false) Member loginMember){
        
        // 1. 세션 유효성 검사
        if (loginMember == null) {
            ErrorResponse errorResponse = new ErrorResponse(
                    "UNAUTHORIZED",
                    "로그인이 필요합니다. (세션이 만료되었거나, 로그인하지 않은 사용자입니다.)"
            );
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(errorResponse);
        }

        // 1-1. URL 유효성 체크
        if(!isValidCallbackUrl(callbackUrl)){
            ErrorResponse errorResponse = new ErrorResponse(
                    "INVALID_CALLBACK_URL",
                    "유효하지 않은 콜백 URL입니다. (http/https 형식이 아니거나, 내부망/localhost 주소일 수 없습니다.)"
            );
            return ResponseEntity.badRequest().body(errorResponse);
        }

        // 2. DB 에 요청 저장
        DetectionRequest newRequest;
        try {
            newRequest = detectionService.saveNewDetectionRequest(loginMember, callbackUrl);
            // [수정] 3.1 메소드 이름 변경 (startDetection -> startUrlDetection)
            detectionService.startUrlDetection(newRequest , videoUrl);
        } catch (Exception e) {
            ErrorResponse errorResponse = new ErrorResponse(
                    "SERVER_ERROR",
                    "서버 에러 입니다."
            );
            return ResponseEntity
                    .status(HttpStatus.INTERNAL_SERVER_ERROR) // 500
                    .body(errorResponse);
        }

        // 사용자에게 즉시 응답
        DetectionResponse response = new DetectionResponse(newRequest.getRequestId());
        return ResponseEntity.accepted().body(response);
    }

    // ... (private validateAndGetMimeType, isValidCallbackUrl, formatFileSize 메소드는 기존과 동일) ...
    private String validateAndGetMimeType(MultipartFile file) {
        if (file.isEmpty()) { return null; }
        try (InputStream inputStream = file.getInputStream()) {
            return tika.detect(inputStream);
        } catch (Exception e) {
            log.error("파일 형식 검사 중 오류 발생", e);
            return null;
        }
    }
    
    private boolean isValidCallbackUrl(String callbackUrl) {
        if (callbackUrl == null || callbackUrl.isBlank()) { return false; }
        String[] schemes = {"http", "https"};
        UrlValidator urlValidator = new UrlValidator(schemes);
        if (!urlValidator.isValid(callbackUrl)) { return false; }
        try {
            java.net.URL url = new java.net.URL(callbackUrl);
            String host = url.getHost();
            String ipRegex = "^(127\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3})|" +
                    "^(localhost)|" +
                    "^(10\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3})|" +
                    "^(172\\.(1[6-9]|2[0-9]|3[0-1])\\.\\d{1,3}\\.\\d{1,3})|" +
                    "^(192\\.168\\.\\d{1,3}\\.\\d{1,3})|" +
                    "^(0\\.0\\.0\\.0)|" +
                    "^(::1)$";
            if (host.matches(ipRegex)) {
                log.warn("SSRF가 의심되는 Callback URL 차단: {}", callbackUrl);
                return false;
            }
        } catch (Exception e) {
            log.warn("Callback URL 파싱 실패: {}", callbackUrl, e);
            return false;
        }
        return true;
    }
    
    private String formatFileSize(long bytes) {
        if (bytes < 1024) return bytes + " B";
        int exp = (int) (Math.log(bytes) / Math.log(1024));
        char pre = "KMGTPE".charAt(exp - 1);
        return String.format("%.1f %sB", bytes / Math.pow(1024, exp), pre);
    }
}