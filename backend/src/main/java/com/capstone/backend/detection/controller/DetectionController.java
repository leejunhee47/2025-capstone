package com.capstone.backend.detection.controller;

import com.capstone.backend.core.common.SessionConst;
import com.capstone.backend.detection.exceptions.InvalidRequestId;
import com.capstone.backend.detection.model.DetectionRequest;
import com.capstone.backend.detection.model.DetectionStatus;
import com.capstone.backend.detection.repository.DetectionRequestRepository;
import com.capstone.backend.detection.response.AiResultResponse;
import com.capstone.backend.detection.response.AnalysisResultResponse;
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
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/v1/detection")
public class DetectionController {

    private final Tika tika = new Tika();
    private final DetectionService detectionService;

    private final DetectionRequestRepository detectionRequestRepository;

    // [추가] 파일 저장을 위해 컨트롤러에도 uploadDir 주입
    @Value("${file.upload-dir}")
    private String uploadDir;

    @Autowired
    public DetectionController(DetectionService detectionService, DetectionRequestRepository detectionRequestRepository) { 
	this.detectionService = detectionService; 
	this.detectionRequestRepository = detectionRequestRepository;
    }

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



    @PostMapping("/result")
    public ResponseEntity<?> getDetectionResult(@RequestBody AiResultResponse aiResultResponse) {
        String requestId = aiResultResponse.getRequestId();
        Long fileSize = aiResultResponse.getAnalysisResultResponse().getFileSize();

        log.info("AI 서버로부터 분석 결과 수신 (RequestId: {})", requestId);
        log.info("AI 서버로부터 분석 결과 (FileSize: {})", fileSize);

        detectionService.updateDetectionStatus(Long.parseLong(requestId),DetectionStatus.COMPLETED);
        // TODO: 수신한 jsonResultBody (JSON 문자열)를 파싱(ObjectMapper 등)하여 데이터베이스에 저장

        return ResponseEntity.ok().build();
    }


    @PostMapping("/polling")
    public ResponseEntity<?> passDetectionResult(@RequestParam("requestId") Long requestId){

        try {
            // 1. Service에서 DetectionRequest 조회
            DetectionRequest request = detectionService.getDetectionByRequestId(requestId);
            DetectionStatus status = request.getStatus();

            // 2. 상태에 따라 다른 HTTP Status Code 반환
            switch (status) {
                case COMPLETED:
                    // 완료: 200 OK (성공)
                    // todo : DB 에서 결과 조회 및 반환
                    return ResponseEntity.ok(status);

                case PENDING:
                case PROCESSING:
                    // 대기중 or 처리중: 202 Accepted (아직 처리 중이므로 클라이언트가 재시도)
                    return ResponseEntity.status(HttpStatus.ACCEPTED).body(status);

                case FAILED:
                    // 실패: 500 Internal Server Error (서버 처리 중 오류 발생)
                    return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(status);

                default:
                    // 혹시 모를 예외 상태
                    log.warn("Unknown detection status encountered: {}", status);
                    return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Unknown status");
            }

        } catch (InvalidRequestId e) {
            // ID를 찾지 못한 경우 (Service에서 발생시킨 예외): 404 Not Found
            log.warn("Polling request for non-existent ID: {}", requestId);
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(e.getMessage());

        } catch (Exception e) {
            // 그 외 알 수 없는 에러: 500 Internal Server Error
            log.error("Error during polling for requestId: {}", requestId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("An unexpected error occurred.");
        }
    }
    
    // 탐지 기록 목록 조회
    @GetMapping("/history")
    public ResponseEntity<?> getDetectionHistory(@SessionAttribute(value = SessionConst.LOGIN_MEMBER, required = false) Member loginMember) {
        if (loginMember == null) return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(new ErrorResponse("UNAUTHORIZED", "로그인이 필요합니다."));

        List<DetectionRequest> requests = detectionRequestRepository.findAllByMemberOrderByCreatedAtDesc(loginMember);
        List<Map<String, Object>> historyList = new ArrayList<>();

        for (DetectionRequest req : requests) {
            Map<String, Object> item = new HashMap<>();
            item.put("requestId", req.getRequestId());
            item.put("date", req.getCreatedAt().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")));
            item.put("status", req.getStatus().name());
            
            // 썸네일이나 제목 등은 실제 데이터가 없으므로 임시 값
            item.put("title", "분석 영상 #" + req.getRequestId());
            
            // 결과 간략 정보 (DB에 저장이 안되어 있으므로 랜덤/임시값 처리)
            if (req.getStatus() == DetectionStatus.COMPLETED) {
                item.put("result", "안전"); // 또는 "위험" (나중에 DB에서 가져와야 함)
            } else {
                item.put("result", req.getStatus().name());
            }
            historyList.add(item);
        }

        return ResponseEntity.ok(historyList);
    }

    // 상세 결과 조회
    @GetMapping("/record/{requestId}")
    public ResponseEntity<?> getRecordDetail(@PathVariable("requestId") Long requestId,
                                             @SessionAttribute(value = SessionConst.LOGIN_MEMBER, required = false) Member loginMember) {
        if (loginMember == null) return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(new ErrorResponse("UNAUTHORIZED", "로그인이 필요합니다."));

        DetectionRequest request = detectionRequestRepository.findById(requestId).orElse(null);
        if (request == null) return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Not Found");

        // 테스트를 위해 가짜 데이터 설정
        Map<String, Object> detail = new HashMap<>();
        detail.put("requestId", request.getRequestId());
        detail.put("status", request.getStatus().name());
        detail.put("date", request.getCreatedAt().toString());

        Map<String, Double> probabilities = new HashMap<>();
        probabilities.put("real", 0.78);
        probabilities.put("fake", 0.22);
        
        detail.put("probabilities", probabilities);
        detail.put("durationSec", 15.015);
        detail.put("verdict", "real"); // "real" or "fake"

        Map<String, Object> summary = new HashMap<>();
        summary.put("title", "진짜 영상으로 판정 (신뢰도: 78.1%)");
        summary.put("detailed_explanation", "전체 1개 구간을 분석한 결과, 부자연스러운 음성-입모양 불일치가 감지되지 않았습니다.");
        detail.put("summary", summary);

        return ResponseEntity.ok(detail);
    }

    @GetMapping("/test")
    public ResponseEntity<?> test() {
        detectionService.testDetection();
        return ResponseEntity.ok().build();
    }

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