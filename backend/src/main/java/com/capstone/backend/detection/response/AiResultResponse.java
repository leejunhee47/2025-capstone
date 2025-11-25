package com.capstone.backend.detection.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

import java.util.List;

@Getter
@Setter
@ToString
@NoArgsConstructor
public class AiResultResponse {

    private Metadata metadata;

    @JsonProperty("video_info")
    private VideoInfo videoInfo;

    private Detection detection;

    private Summary summary;


    // --- 1. Metadata DTO ---
    @Getter @Setter @NoArgsConstructor
    public static class Metadata {
        @JsonProperty("video_id")
        private String videoId;        // AI 서버 내부 관리 ID

        @JsonProperty("request_id")
        private String requestId;      // 요청 ID

        @JsonProperty("processed_at")
        private String processedAt;    // 처리 시각 (ISO 8601 String)

        @JsonProperty("processing_time_ms")
        private Double processingTimeMs; // 처리 소요 시간

        @JsonProperty("pipeline_version")
        private String pipelineVersion;  // 모델 버전
    }

    // --- 2. VideoInfo DTO ---
    @Getter @Setter @NoArgsConstructor
    public static class VideoInfo {
        @JsonProperty("duration_sec")
        private Double durationSec;

        @JsonProperty("total_frames")
        private Integer totalFrames;

        private Double fps;
        private String resolution;

        @JsonProperty("original_path")
        private String originalPath;   // AI 서버 내의 파일 경로
    }

    // --- 3. Detection DTO ---
    @Getter @Setter @NoArgsConstructor
    public static class Detection {
        private String verdict;        // real / fake
        private Double confidence;     // 종합 신뢰도

        private Probabilities probabilities;

        @JsonProperty("suspicious_frame_count")
        private Integer suspiciousFrameCount;

        @JsonProperty("suspicious_frame_ratio")
        private Double suspiciousFrameRatio;
    }

    @Getter @Setter @NoArgsConstructor
    public static class Probabilities {
        private Double real;
        private Double fake;
    }


    // --- 4. Summary DTO ---
    @Getter @Setter @NoArgsConstructor
    public static class Summary {
        private String title;

        @JsonProperty("risk_level")
        private String riskLevel;

        @JsonProperty("primary_reason")
        private String primaryReason;

        @JsonProperty("suspicious_interval_count")
        private Integer suspiciousIntervalCount;

        @JsonProperty("top_suspicious_phonemes")
        private List<String> topSuspiciousPhonemes; // ["ㄱ", "ㄴ"]

        @JsonProperty("detailed_explanation")
        private String detailedExplanation;
    }


}
