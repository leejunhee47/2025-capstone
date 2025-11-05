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

    @Autowired
    public DetectionController(DetectionService detectionService) { this.detectionService = detectionService; }



    @PostMapping("/upload")
    public ResponseEntity<?> processUploadDetectionRequest(@RequestParam("file") MultipartFile file ,
                                                           @RequestParam("callbackUrl") String callbackUrl ,
                                                           @SessionAttribute(value = SessionConst.LOGIN_MEMBER, required = false) Member loginMember){


        // 1. 세션 유효성 검사 (가장 먼저 수행)
        if (loginMember == null) {
            ErrorResponse errorResponse = new ErrorResponse(
                    "UNAUTHORIZED",
                    "로그인이 필요합니다. (세션이 만료되었거나, 로그인하지 않은 사용자입니다.)"
            );
            // 401 Unauthorized 상태 코드와 함께 에러 응답 반환
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


        // 2. DB 에 요청 저장
        DetectionRequest newRequest;
        try {
            newRequest = detectionService.saveNewDetectionRequest(loginMember, callbackUrl);
            // 3. 비동기 처리 시작 (파일 저장 , AI 요청 , DB 수정)
            detectionService.startDetection(newRequest , file);
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




    @PostMapping("/link")
    public ResponseEntity<?> processLinkDetectionRequest(@RequestParam("videoUrl") String videoUrl ,
                                                           @RequestParam("callbackUrl") String callbackUrl ,
                                                           @SessionAttribute(value = SessionConst.LOGIN_MEMBER, required = false) Member loginMember){

        // 1. 세션 유효성 검사 (가장 먼저 수행)
        if (loginMember == null) {
            ErrorResponse errorResponse = new ErrorResponse(
                    "UNAUTHORIZED",
                    "로그인이 필요합니다. (세션이 만료되었거나, 로그인하지 않은 사용자입니다.)"
            );
            // 401 Unauthorized 상태 코드와 함께 에러 응답 반환
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
        // todo: 유효하지 않은 링크인지 , 서버에러 인지 구별을 못함 (구별하는 것 apk 로 개발해야할듯)
        DetectionRequest newRequest;
        try {
            newRequest = detectionService.saveNewDetectionRequest(loginMember, callbackUrl);
            // 3. 비동기 처리 시작 (파일 추출 및 저장 , AI 요청 , DB 수정)
            detectionService.startDetection(newRequest , videoUrl);
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



















    /**
     * Tika를 사용해 실제 파일 형식을 검사하고 MIME Type을 반환합니다.
     * @param file 검사할 MultipartFile
     * @return 비디오 파일이면 MIME Type (e.g., "video/mp4"), 아니면 null
     */
    private String validateAndGetMimeType(MultipartFile file) {
        if (file.isEmpty()) {
            return null;
        }

        try (InputStream inputStream = file.getInputStream()) {
            return tika.detect(inputStream);
        } catch (Exception e) {
            log.error("파일 형식 검사 중 오류 발생", e);
            return null;
        }
    }
    private boolean isValidCallbackUrl(String callbackUrl) {
        if (callbackUrl == null || callbackUrl.isBlank()) {
            return false;
        }

        String[] schemes = {"http", "https"};
        UrlValidator urlValidator = new UrlValidator(schemes);

        if (!urlValidator.isValid(callbackUrl)) {
            return false;
        }

        try {
            java.net.URL url = new java.net.URL(callbackUrl);
            String host = url.getHost();

            // SSRF 방지를 위한 Private IP/localhost 정규식 검사
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

    /**
     * 파일 크기(byte)를 읽기 쉬운 문자열(KB, MB, GB)로 변환합니다.
     * @param bytes 파일 크기 (long)
     * @return "123.5 MB" 형태의 문자열
     */
    private String formatFileSize(long bytes) {
        if (bytes < 1024) return bytes + " B";
        int exp = (int) (Math.log(bytes) / Math.log(1024));
        char pre = "KMGTPE".charAt(exp - 1);
        return String.format("%.1f %sB", bytes / Math.pow(1024, exp), pre);
    }

}
