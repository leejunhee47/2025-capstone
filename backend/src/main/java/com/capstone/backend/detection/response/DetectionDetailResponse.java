package com.capstone.backend.detection.response;

import com.capstone.backend.detection.model.DetectionResult;
import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

@Getter
@Builder
public class DetectionDetailResponse {

    // 생성 일시
    private LocalDateTime createdAt;

    // [Video Info]
    private Double durationSec;
    private Integer totalFrames;
    private Double fps;
    private String resolution;

    // [Detection Info]
    private String verdict;
    private Double confidence;
    private Double probabilityReal;
    private Double probabilityFake;
    private Integer suspiciousFrameCount;
    private Double suspiciousFrameRatio;

    // [Summary Info]
    private String summaryTitle;
    private String summaryRiskLevel;
    private String summaryPrimaryReason;
    private Integer summarySuspiciousIntervalCount;
    private List<String> summaryTopSuspiciousPhonemes; // List 형태로 변환하여 반환
    private String summaryDetailedExplanation;

    public static DetectionDetailResponse from(DetectionResult result) {
        // DB에 저장된 "ㄱ,ㄲ,ㄴ" 형태의 문자열을 List로 변환
        List<String> phonemes = Collections.emptyList();
        if (result.getSummaryTopSuspiciousPhonemes() != null && !result.getSummaryTopSuspiciousPhonemes().isEmpty()) {
            phonemes = Arrays.asList(result.getSummaryTopSuspiciousPhonemes().split(","));
        }

        return DetectionDetailResponse.builder()
                .createdAt(result.getCreatedAt())

                // Video Info
                .durationSec(result.getDurationSec())
                .totalFrames(result.getTotalFrames())
                .fps(result.getFps())
                .resolution(result.getResolution())

                // Detection Info
                .verdict(result.getVerdict())
                .confidence(result.getConfidence())
                .probabilityReal(result.getProbabilityReal())
                .probabilityFake(result.getProbabilityFake())
                .suspiciousFrameCount(result.getSuspiciousFrameCount())
                .suspiciousFrameRatio(result.getSuspiciousFrameRatio())

                // Summary Info
                .summaryTitle(result.getSummaryTitle())
                .summaryRiskLevel(result.getSummaryRiskLevel())
                .summaryPrimaryReason(result.getSummaryPrimaryReason())
                .summarySuspiciousIntervalCount(result.getSummarySuspiciousIntervalCount())
                .summaryTopSuspiciousPhonemes(phonemes)
                .summaryDetailedExplanation(result.getSummaryDetailedExplanation())
                .build();
    }

}
