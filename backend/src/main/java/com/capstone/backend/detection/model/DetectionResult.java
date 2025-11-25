package com.capstone.backend.detection.model;


import com.capstone.backend.detection.response.AiResultResponse;
import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor
@Builder
public class DetectionResult {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long resultId;

    // 원본 요청과의 1:1 관계
    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "request_id")
    private DetectionRequest detectionRequest;


    // --- [1] Metadata 정보 ---
//    private String aiVideoId;          // video_id
//    private String aiRequestId;        // request_id (문자열 그대로 저장)
//    private LocalDateTime processedAt; // processed_at (시간 타입으로 변환)
//    private Double processingTimeMs;   // processing_time_ms
//    private String pipelineVersion;    // pipeline_version

    // --- [2] Video Info ---
    private Double durationSec;
    private Integer totalFrames;
    private Double fps;
    private String resolution;


//    @Column(length = 500)
//    private String originalPath;       // 경로가 길 수 있으므로 길이 넉넉하게

    // --- [3] Detection Info ---
    private String verdict;
    private Double confidence;
    private Double probabilityReal;
    private Double probabilityFake;
    private Integer suspiciousFrameCount;
    private Double suspiciousFrameRatio;

    // --- [4] Summary Info ---
    private String summaryTitle;
    private String summaryRiskLevel;
    private String summaryPrimaryReason;
    private Integer summarySuspiciousIntervalCount;

    // List -> String 변환 (예: "ㄱ,ㄲ,ㄴ")
    private String summaryTopSuspiciousPhonemes;

    @Column(columnDefinition = "TEXT") // 긴 텍스트
    private String summaryDetailedExplanation;

    @OneToMany(
            mappedBy = "detectionResult",
            fetch = FetchType.LAZY
    )
    private List<DetectionReport> detectionReports = new ArrayList<>();


    @CreationTimestamp
    @Column(updatable = false, nullable = false)
    private LocalDateTime createdAt;

    /**
     * DTO -> Entity 변환 메서드 (모든 필드 매핑)
     */
    public static DetectionResult createFrom(DetectionRequest request, AiResultResponse dto) {

        // 1. List -> String 변환
        String phonemesString = "";
        if (dto.getSummary().getTopSuspiciousPhonemes() != null) {
            phonemesString = String.join(",", dto.getSummary().getTopSuspiciousPhonemes());
        }
//
//        // 2. String(ISO Date) -> LocalDateTime 변환
//        // 예: "2025-11-18T07:38:46.802351Z" 파싱
//        LocalDateTime parsedTime = null;
//        if (dto.getMetadata().getProcessedAt() != null) {
//            try {
//                parsedTime = LocalDateTime.parse(dto.getMetadata().getProcessedAt(), DateTimeFormatter.ISO_DATE_TIME);
//            } catch (Exception e) {
//                // 파싱 실패 시 null 처리하거나 로그 남김
//                parsedTime = LocalDateTime.now();
//            }
//        }


        return DetectionResult.builder()
                .detectionRequest(request)

                // Metadata
//                .aiVideoId(dto.getMetadata().getVideoId())
//                .aiRequestId(dto.getMetadata().getRequestId())
//                .processedAt(parsedTime)
//                .processingTimeMs(dto.getMetadata().getProcessingTimeMs())
//                .pipelineVersion(dto.getMetadata().getPipelineVersion())

                // Video Info
                .durationSec(dto.getVideoInfo().getDurationSec())
                .totalFrames(dto.getVideoInfo().getTotalFrames())
                .fps(dto.getVideoInfo().getFps())
                .resolution(dto.getVideoInfo().getResolution())
//                .originalPath(dto.getVideoInfo().getOriginalPath())

                // Detection
                .verdict(dto.getDetection().getVerdict())
                .confidence(dto.getDetection().getConfidence())
                .probabilityReal(dto.getDetection().getProbabilities().getReal())
                .probabilityFake(dto.getDetection().getProbabilities().getFake())
                .suspiciousFrameCount(dto.getDetection().getSuspiciousFrameCount())
                .suspiciousFrameRatio(dto.getDetection().getSuspiciousFrameRatio())

                // Summary
                .summaryTitle(dto.getSummary().getTitle())
                .summaryRiskLevel(dto.getSummary().getRiskLevel())
                .summaryPrimaryReason(dto.getSummary().getPrimaryReason())
                .summarySuspiciousIntervalCount(dto.getSummary().getSuspiciousIntervalCount())
                .summaryTopSuspiciousPhonemes(phonemesString)
                .summaryDetailedExplanation(dto.getSummary().getDetailedExplanation())
                .build();
    }





}
