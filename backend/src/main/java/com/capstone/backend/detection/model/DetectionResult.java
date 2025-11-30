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
@NoArgsConstructor
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

    @Builder.Default
    @OneToMany(
            mappedBy = "detectionResult",
            fetch = FetchType.LAZY
    )
    private List<DetectionReport> detectionReports = new ArrayList<>();


    @CreationTimestamp
    @Column(updatable = false, nullable = false)
    private LocalDateTime createdAt;

    /**
     * DTO -> Entity 변환 메서드 (수정됨)
     */
    public static DetectionResult createFrom(DetectionRequest request, AiResultResponse dto) {


        // todo : null , 0 넣은 부분들 팀원들에게 알림
        // 1. 사라진 필드들에 대한 기본값 처리
        // Phonemes가 JSON에 없으므로 빈 문자열 또는 null 처리
        String phonemesString = "";
        // (만약 나중에 JSON에 다시 생긴다면 dto.getSummary()... 로 복구)

        return DetectionResult.builder()
                .detectionRequest(request)

                // Metadata (주석 처리된 부분은 필요 시 해제)
                // .aiVideoId(dto.getMetadata().getVideoId()) ...

                // Video Info
                .durationSec(dto.getVideoInfo().getDurationSec())
                .totalFrames(dto.getVideoInfo().getTotalFrames())
                .fps(dto.getVideoInfo().getFps())
                .resolution(dto.getVideoInfo().getResolution())

                // Detection
                .verdict(dto.getDetection().getVerdict())
                .confidence(dto.getDetection().getConfidence())
                .probabilityReal(dto.getDetection().getProbabilities().getReal())
                .probabilityFake(dto.getDetection().getProbabilities().getFake())

                // [변경] JSON에 값이 없으므로 0 또는 0.0으로 설정
                .suspiciousFrameCount(0)
                .suspiciousFrameRatio(0.0)

                // Summary
                .summaryTitle(dto.getSummary().getTitle())
                .summaryRiskLevel(dto.getSummary().getRiskLevel())
                .summaryPrimaryReason(dto.getSummary().getPrimaryReason())

                // [변경] JSON에 값이 없으므로 0으로 설정
                .summarySuspiciousIntervalCount(0)

                // [변경] JSON에 값이 없으므로 빈 문자열 설정
                .summaryTopSuspiciousPhonemes(phonemesString)

                .summaryDetailedExplanation(dto.getSummary().getDetailedExplanation())
                .build();
    }





}
