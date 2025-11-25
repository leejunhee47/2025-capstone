package com.capstone.backend.detection.service;

import com.capstone.backend.detection.exceptions.InvalidRequestId;
import com.capstone.backend.detection.model.DetectionReport;
import com.capstone.backend.detection.repository.DetectionReportRepository;
import com.capstone.backend.detection.response.*;
import com.capstone.backend.detection.model.DetectionRequest;
import com.capstone.backend.detection.model.DetectionResult;
import com.capstone.backend.detection.model.DetectionStatus;
import com.capstone.backend.detection.repository.DetectionRequestRepository;
import com.capstone.backend.detection.repository.DetectionResultRepository;
import com.capstone.backend.member.model.Member;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.http.MediaType;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.servlet.support.ServletUriComponentsBuilder;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@Service
@Slf4j
public class DetectionService {

    private final DetectionRequestRepository detectionRequestRepository;
    private final DetectionResultRepository detectionResultRepository;
    private final DetectionReportRepository detectionReportRepository;
    private final WebClient webClient;

    @Value("${file.upload-dir}")
    private String uploadDir;


    @Value("${thumbnail.upload-dir}")
    private String thumbnailUploadDir;

    @Value("${ai.url.request}")
    private String aiUrl;


    @Autowired
    public DetectionService(DetectionRequestRepository detectionRequestRepository ,
                            DetectionResultRepository detectionResultRepository ,
                            DetectionReportRepository detectionReportRepository ,
                            WebClient webClient) {
        this.detectionRequestRepository = detectionRequestRepository;
        this.detectionResultRepository = detectionResultRepository;
        this.detectionReportRepository = detectionReportRepository;
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


            // 썸네일 따놓기
            String thumbnailPath = extractThumbnail(savedFilePath, managedRequest.getRequestId());
            managedRequest.setThumbnailPath(thumbnailPath);


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

            // 썸네일 따놓기
            String thumbnailPath = extractThumbnail(outputFilePath, managedRequest.getRequestId());
            managedRequest.setThumbnailPath(thumbnailPath);


            managedRequest.setStatus(DetectionStatus.PROCESSING);
            detectionRequestRepository.save(managedRequest);



            //  AI 서버에 분석 요청
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

    private String extractThumbnail(String videoPath, Long requestId) throws IOException, InterruptedException {
        // 썸네일 저장 디렉토리 생성
        File thumbnailDir = new File(thumbnailUploadDir);
        if (!thumbnailDir.exists()) {
            thumbnailDir.mkdirs();
        }

        // 저장될 썸네일 파일 경로 (예: /path/to/thumbs/10.jpg)
        String outputThumbnailPath = thumbnailUploadDir + File.separator + requestId + ".jpg";

        // FFmpeg 명령어 구성
        // -i [입력영상] -ss 00:00:01 (1초 지점) -vframes 1 (1프레임만) -y (덮어쓰기) [출력이미지]
        String[] command = {
                "ffmpeg",
                "-i", videoPath,
                "-ss", "00:00:01.000",
                "-vframes", "1",
                "-y",
                outputThumbnailPath
        };

        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.redirectErrorStream(true);

        // 프로세스 실행
        Process process = processBuilder.start();

        // 10초 타임아웃
        if (!process.waitFor(10, TimeUnit.SECONDS)) {
            process.destroy();
            log.error("Thumbnail extraction timed out for RequestId: {}", requestId);
            throw new RuntimeException("Thumbnail extraction timed out");
        }

        if (process.exitValue() != 0) {
            log.error("Thumbnail extraction failed with exit code: {}", process.exitValue());
            // 썸네일 실패 시 null을 반환하거나 예외를 던짐 (여기선 예외 던져서 FAILED 처리 되도록 함)
            throw new RuntimeException("Thumbnail extraction failed");
        }

        log.info("Thumbnail extracted successfully: {}", outputThumbnailPath);
        return outputThumbnailPath;
    }

    @Transactional(readOnly = true)
    public DetectionRequest getDetectionByRequestId(Long requestId) throws InvalidRequestId {
        return detectionRequestRepository.findById(requestId)
                .orElseThrow(() -> new InvalidRequestId("Invalid Request ID: " + requestId));
    }

    @Transactional
    public void processDetectionResult(AiResultResponse response, List<String> imagePaths) throws IllegalArgumentException{

        String rawId = response.getMetadata().getRequestId();
        Long targetRequestId = null;

        // 1. Request ID 파싱
        try {
            String numStr = rawId.replaceAll("[^0-9]", "");
            targetRequestId = Long.parseLong(numStr);
        } catch (NumberFormatException e) {
            log.error("Request ID 파싱 실패. AI Response ID: {}", rawId);
            throw new IllegalArgumentException("Invalid Request ID format from AI: " + rawId);
        }

        log.info("결과 저장 프로세스 시작. Request ID: {}", targetRequestId);

        // 2. DB 에서 원본 요청 조회
        DetectionRequest request = detectionRequestRepository.findById(targetRequestId)
                .orElseThrow(() -> new IllegalArgumentException("DetectionRequest not found"));

        // 3. DTO -> Entity 변환 (DetectionResult 생성)
        DetectionResult result = DetectionResult.createFrom(request, response);

        // 4. 결과 Entity 저장
        detectionResultRepository.save(result);

        // 5. (추가) 결과 이미지들(DetectionReport) 저장
        if (imagePaths != null && !imagePaths.isEmpty()) {
            int sequence = 1;
            for (String path : imagePaths) {
                DetectionReport report = new DetectionReport();
                report.setImgPath(path);
                report.setSequence(sequence++);
                report.setDetectionResult(result);
                detectionReportRepository.save(report);
            }
            log.info("결과 이미지 {}장 저장 완료.", imagePaths.size());
        }

        // 6. 요청 상태 업데이트 (COMPLETED)
        request.setStatus(DetectionStatus.COMPLETED);
        detectionRequestRepository.save(request);

        log.info("분석 결과 저장 및 상태 업데이트 완료. Request ID: {}", targetRequestId);
    }

    @Transactional(readOnly = true)
    public DetectionBriefResponse getDetectionBriefResponse(Long requestId) throws InvalidRequestId {
        DetectionResult result = detectionResultRepository.findByDetectionRequest_RequestId(requestId)
                .orElseThrow(() -> new IllegalArgumentException("Result not found for Request ID: " + requestId));

        return DetectionBriefResponse.from(result);
    }

    @Transactional(readOnly = true)
    public DetectionDetailResponse getDetectionDetailByResultId(Long resultId) {
        // 1. resultId로 결과 조회
        DetectionResult result = detectionResultRepository.findById(resultId)
                .orElseThrow(() -> new IllegalArgumentException("Detection result not found with ID: " + resultId));

        // 2. DTO 변환 후 반환
        return DetectionDetailResponse.from(result);
    }

    /**
     * 커서 기반 페이지네이션 조회
     */
    @Transactional(readOnly = true)
    public DetectionListResponse getDetectionList(Member member, int limit, Long cursor) {

        // 다음 페이지 존재 여부를 확인하기 위해 limit + 1개 조회
        Pageable pageable = PageRequest.of(0, limit + 1);
        List<DetectionResult> results;

        if (cursor == null) {
            // 첫 페이지 요청
            results = detectionResultRepository.findByDetectionRequest_Member_memberIdOrderByResultIdDesc(member.getMemberId(), pageable);
        } else {
            // 커서 이후 요청 (resultId < cursor)
            results = detectionResultRepository.findByResultIdLessThanAndDetectionRequest_Member_memberIdOrderByResultIdDesc(cursor, member.getMemberId(), pageable);
        }

        // 다음 페이지 여부 확인
        boolean hasNextPage = false;
        if (results.size() > limit) {
            hasNextPage = true;
            results.remove(limit); // 확인용으로 가져온 마지막 데이터 제거 (클라이언트에게는 limit개만 전달)
        }

        // 다음 커서 계산 (리스트의 마지막 아이템 ID)
        Long nextCursor = null;
        if (!results.isEmpty()) {
            nextCursor = results.get(results.size() - 1).getResultId();
        }

        // Entity -> BriefResponse DTO 변환
        List<DetectionBriefResponse> dtos = results.stream()
                .map(DetectionBriefResponse::from)
                .collect(Collectors.toList());

        // 최종 응답 객체 생성
        return DetectionListResponse.builder()
                .items(dtos)
                .pageInfo(DetectionListResponse.PageInfo.builder()
                        .nextCursor(hasNextPage ? nextCursor : null) // 다음 페이지 없으면 null
                        .hasNextPage(hasNextPage)
                        .build())
                .build();
    }

    /**
     * 결과 ID를 받아 이미지 리포트 URL 리스트를 포함한 응답을 반환
     */
    @Transactional(readOnly = true)
    public DetectionImgUrlResponse getDetectionReportResponse(Long resultId) {

        // 1. DetectionResult 조회
        DetectionResult result = detectionResultRepository.findById(resultId)
                .orElseThrow(() -> new IllegalArgumentException("Detection result not found with ID: " + resultId));

        // 2. DetectionReport 리스트 조회 (Lazy Loading 해결을 위해 트랜잭션 안에서 접근)
        List<DetectionReport> reports = result.getDetectionReports();

        // 3. Report 엔티티 -> ImageInfo DTO 변환
        List<DetectionImgUrlResponse.ReportImageInfo> imageInfos = reports.stream()
                .map(report -> {
                    // 클라이언트가 이미지를 볼 수 있는 API URL 생성
                    // 예: http://host:port/api/v1/detection/report/image/{reportId}
                    String imageUrl = ServletUriComponentsBuilder.fromCurrentContextPath()
                            .path("/api/v1/detection/report/image/")
                            .path(String.valueOf(report.getReportId()))
                            .toUriString();

                    return DetectionImgUrlResponse.ReportImageInfo.builder()
                            .sequence(report.getSequence())
                            .url(imageUrl)
                            .build();
                })
                .collect(Collectors.toList());

        // 4. 최종 응답 DTO 생성
        return DetectionImgUrlResponse.builder()
                .resultId(result.getResultId())
                .createdAt(result.getCreatedAt())
                .images(imageInfos)
                .build();
    }

    /**
     * (추가) Report ID로 실제 이미지 파일 경로 조회 (이미지 서빙용)
     */
    @Transactional(readOnly = true)
    public File getReportImageFile(Long reportId) {
        DetectionReport report = detectionReportRepository.findById(reportId)
                .orElseThrow(() -> new IllegalArgumentException("Report not found with ID: " + reportId));

        String path = report.getImgPath();
        File file = new File(path);

        if (!file.exists()) {
            throw new IllegalArgumentException("Image file does not exist on server");
        }
        return file;
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