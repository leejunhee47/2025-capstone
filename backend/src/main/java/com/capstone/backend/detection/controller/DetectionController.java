package com.capstone.backend.detection.controller;

import com.capstone.backend.core.common.SessionConst;
import com.capstone.backend.detection.exceptions.InvalidRequestId;
import com.capstone.backend.detection.model.DetectionRequest;
import com.capstone.backend.detection.model.DetectionStatus;
import com.capstone.backend.detection.response.*;
import com.capstone.backend.detection.service.DetectionService;
import com.capstone.backend.member.model.Member;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.validator.routines.UrlValidator;
import org.apache.tika.Tika;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value; // @Value 임포트
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;


@Slf4j
@RestController
@RequestMapping("/api/v1/")
public class DetectionController {

    private final Tika tika = new Tika();
    private final DetectionService detectionService;

    @Value("${file.upload-dir}")
    private String uploadDir;

    @Value("${result.img}")
    private String resultImgDir;



    @Autowired
    public DetectionController(DetectionService detectionService) { this.detectionService = detectionService; }

    // 클라이언트 -> 백엔드 서버 (업로드)
    @PostMapping("detection/upload")
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
//            String uniqueFileName = newRequest.getRequestId() + "_" + originalFilename;
            dest = new File(uploadDir + File.separator + originalFilename);

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
        DetectionCheckResponse response = new DetectionCheckResponse(newRequest.getRequestId());
        return ResponseEntity.accepted().body(response);
    }

    // 클라이언트 -> 백엔드 서버 (링크)
    @PostMapping("detection/link")
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
            detectionService.subUrlDetection(newRequest, videoUrl);
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
        DetectionCheckResponse response = new DetectionCheckResponse(newRequest.getRequestId());
        return ResponseEntity.accepted().body(response);
    }


    // AI 서버 -> 백엔드 서버 (결과: JSON + Images)
    @PostMapping(value = "detection/result", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<?> getDetectionResult(
            @RequestPart("result") AiResultResponse aiResultResponse,
            @RequestPart(value = "images", required = false) List<MultipartFile> images
    ) {

        log.info("결과 수신됨. Metadata: {}", aiResultResponse.getMetadata());

        try {
            // 1. RequestId 추출 (파일명 생성을 위해)
            String rawId = aiResultResponse.getMetadata().getRequestId();
            // 숫자만 추출
            String requestIdStr = rawId.replaceAll("[^0-9]", "");

            List<String> savedImagePaths = new ArrayList<>();

            // 2. 이미지 파일 저장 로직
            if (images != null && !images.isEmpty()) {
                File dir = new File(resultImgDir);
                if (!dir.exists()) {
                    dir.mkdirs();
                }

                int index = 1;
                for (MultipartFile img : images) {
                    if (img.isEmpty()) continue;

                    // 파일명: {requestId}_{index}.jpg (확장자는 원본 따르거나 고정)
                    // 여기서는 간단히 jpg로 가정하거나 원본 파일명 활용 가능.
                    // 요구사항: requestId로 식별
                    String ext = "jpg";
                    // (필요시) String originalExt = FilenameUtils.getExtension(img.getOriginalFilename());

                    String fileName = requestIdStr + "_" + index + "." + ext;
                    File dest = new File(resultImgDir + File.separator + fileName);

                    img.transferTo(dest);

                    savedImagePaths.add(dest.getAbsolutePath());
                    index++;
                }
            }

            // 3. 서비스 호출 (JSON 결과 + 저장된 이미지 경로 리스트 전달)
            detectionService.processDetectionResult(aiResultResponse, savedImagePaths);

        } catch (IOException e) {
            log.error("이미지 저장 중 오류 발생", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Image save failed");
        } catch (IllegalArgumentException e) {
            log.warn("Illegal argument detected: {}", e.getMessage());
            return ResponseEntity.badRequest().body(e.getMessage());
        } catch (Exception e) {
            log.error("결과 처리 중 알 수 없는 오류 발생", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Server error");
        }

        return ResponseEntity.ok().build();
    }




    // 백엔드 서버 -> 클라이언트 (간단 결과)
    @PostMapping("detection/result/polling")
    public ResponseEntity<?> passDetectionResult(@RequestParam("requestId") Long requestId){

        try {
            // 1. Service 에서 DetectionRequest 조회
            DetectionRequest request = detectionService.getDetectionByRequestId(requestId);
            DetectionStatus status = request.getStatus();

            // 2. 상태에 따라 다른 HTTP Status Code 반환
            switch (status) {
                case COMPLETED:
                    // 완료: 200 OK (성공)
                    return ResponseEntity.ok(detectionService.getDetectionBriefResponse(requestId));

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
            // ID를 찾지 못한 경우 (Service 에서 발생시킨 예외): 404 Not Found
            log.warn("Polling request for non-existent ID: {}", requestId);
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(e.getMessage());

        } catch (Exception e) {
            // 그 외 알 수 없는 에러: 500 Internal Server Error
            log.error("Error during polling for requestId: {}", requestId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("An unexpected error occurred.");
        }
    }

    // 백엔드 서버 -> 클라이언트 (썸네일)
    @PostMapping("detection/thumbnail/polling")
    public ResponseEntity<?> passDetectionThumbnail(@RequestParam("requestId") Long requestId){

        try {
            // 1. 요청 정보 조회
            DetectionRequest request = detectionService.getDetectionByRequestId(requestId);
            DetectionStatus status = request.getStatus();

            // 2. 상태에 따른 분기 처리
            switch (status) {
                case PENDING:
                    // 아직 썸네일 생성 전 -> 202 Accepted
                    return ResponseEntity.status(HttpStatus.ACCEPTED).body("Thumbnail generating...");

                case PROCESSING:
                case COMPLETED:
                    // 썸네일 생성 완료됨 (Service 로직상 PROCESSING 이면 썸네일 존재) -> 200 OK + Image
                    String path = request.getThumbnailPath();

                    // 경로 유효성 검사
                    if (path == null) {
                        log.error("Status is {}, but thumbnail path is null. RequestID: {}", status, requestId);
                        return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Thumbnail path not found");
                    }

                    File file = new File(path);
                    if (!file.exists()) {
                        log.error("Thumbnail file missing at path: {}", path);
                        return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Thumbnail file missing");
                    }

                    // 이미지 리소스 생성
                    Resource resource = new FileSystemResource(file);

                    // Content-Type 결정 (기본 jpg 설정, 필요시 파일 probeContentType 사용)
                    MediaType mediaType = MediaType.IMAGE_JPEG;
                    try {
                        String contentType = Files.probeContentType(Paths.get(path));
                        if (contentType != null) {
                            mediaType = MediaType.parseMediaType(contentType);
                        }
                    } catch (Exception e) {
                        log.warn("Failed to determine content type, using default image/jpeg");
                    }

                    return ResponseEntity.ok()
                            .contentType(mediaType)
                            .body(resource);

                case FAILED:
                    // 실패 상태 -> 404 Not Found (클라이언트가 폴링 중단하도록)
                    return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Detection Failed");

                default:
                    return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Unknown Status");
            }

        } catch (InvalidRequestId e) {
            // 잘못된 ID -> 404 Not Found
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(e.getMessage());
        } catch (Exception e) {
            log.error("Error retrieving thumbnail for requestId: {}", requestId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Server Error");
        }


    }

    // 백엔드 서버 -> 클라이언트 (상세 결과)
    @GetMapping("record/{resultId}")
    public ResponseEntity<?> getDetectionRecordDetail(@PathVariable("resultId") Long resultId) {
        try {
            // 1. 서비스 호출하여 상세 정보 가져오기
            DetectionDetailResponse response = detectionService.getDetectionDetailByResultId(resultId);

            // 2. 200 OK와 함께 데이터 반환
            return ResponseEntity.ok(response);

        } catch (IllegalArgumentException e) {
            // 3. ID에 해당하는 결과가 없을 경우 404 Not Found
            log.warn("Record request for non-existent Result ID: {}", resultId);
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(e.getMessage());

        } catch (Exception e) {
            // 4. 기타 서버 에러 500
            log.error("Error retrieving detection record for resultId: {}", resultId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("An unexpected error occurred.");
        }
    }

    // 백엔드 서버 -> 클라이언트 (페이지 결과)
    @GetMapping("record")
    public ResponseEntity<?> getDetectionRecords(
            @RequestParam(value = "limit", defaultValue = "10") int limit,
            @RequestParam(value = "cursor", required = false) Long cursor,
            @SessionAttribute(value = SessionConst.LOGIN_MEMBER, required = false) Member loginMember) {

        // 1. 로그인 세션 확인
        if (loginMember == null) {
            ErrorResponse errorResponse = new ErrorResponse(
                    "UNAUTHORIZED",
                    "로그인이 필요합니다."
            );
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(errorResponse);
        }

        try {
            // 2. 서비스 호출
            DetectionListResponse response = detectionService.getDetectionList(loginMember, limit, cursor);

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("Error retrieving detection list for memberId: {}", loginMember.getMemberId(), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("An unexpected error occurred.");
        }
    }


    // 백엔드 서버 -> 클라이언트 (결과 이미지 경로)
    @GetMapping("record/{resultId}/report")
    public ResponseEntity<?> getDetectionReport(@PathVariable("resultId") Long resultId) {
        try {
            // 서비스 호출
            DetectionImgUrlResponse response = detectionService.getDetectionReportResponse(resultId);
            return ResponseEntity.ok(response);

        } catch (IllegalArgumentException e) {
            // 3. ID에 해당하는 결과가 없을 경우 404 Not Found
            log.warn("Record request for non-existent Result ID: {}", resultId);
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(e.getMessage());

        } catch (Exception e) {
            // 4. 기타 서버 에러 500
            log.error("Error retrieving detection record for resultId: {}", resultId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("An unexpected error occurred.");
        }
    }


    // 백엔드 서버 -> 클라이언트 (썸네일 반환)
    @GetMapping("record/{resultId}/thumbnail")
    public ResponseEntity<?> getDetectionThumbnail(@PathVariable("resultId") Long resultId) {

        try {
            // 1. 서비스에서 파일 객체 가져오기 (검증 로직 포함됨)
            File file = detectionService.getThumbnailFileByResultId(resultId);
            Resource resource = new FileSystemResource(file);

            // 2. Content-Type 결정 (기본 jpg 설정, 파일 probeContentType 사용)
            MediaType mediaType = MediaType.IMAGE_JPEG;
            try {
                String contentType = Files.probeContentType(file.toPath());
                if (contentType != null) {
                    mediaType = MediaType.parseMediaType(contentType);
                }
            } catch (Exception e) {
                log.warn("Failed to determine content type for resultId {}, using default image/jpeg", resultId);
            }

            // 3. 응답 반환
            return ResponseEntity.ok()
                    .contentType(mediaType)
                    .body(resource);

        } catch (IllegalArgumentException e) {
            // ID가 없거나 파일이 없는 경우 404
            log.warn("Thumbnail not found for Result ID: {}", resultId);
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(e.getMessage());

        } catch (Exception e) {
            // 그 외 서버 에러 500
            log.error("Error retrieving thumbnail for resultId: {}", resultId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }



    // 백엔드 서버 -> 클라이언트 (리포트를 반환)
    @GetMapping("detection/report/image/{reportId}")
    public ResponseEntity<?> getReportImage(@PathVariable("reportId") Long reportId) {
        try {
            // 1. 파일 가져오기
            File file = detectionService.getReportImageFile(reportId);
            Resource resource = new FileSystemResource(file);

            // 2. MIME 타입 추론
            String contentType = Files.probeContentType(file.toPath());
            MediaType mediaType = (contentType != null)
                    ? MediaType.parseMediaType(contentType)
                    : MediaType.IMAGE_JPEG; // 기본값

            // 3. 이미지 바이너리 반환
            return ResponseEntity.ok()
                    .contentType(mediaType)
                    .body(resource);

        } catch (IllegalArgumentException e) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null); // 이미지 없음
        } catch (Exception e) {
            log.error("Error serving image for reportId: {}", reportId, e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }



    @GetMapping("detection/test")
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