package com.capstone.backend.detection.response;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties; // 추가
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
@JsonIgnoreProperties(ignoreUnknown = true) // 중요: 정의되지 않은 필드(detail_view 등)는 무시
public class AiResultResponse {

    private Metadata metadata;

    @JsonProperty("video_info")
    private VideoInfo videoInfo;

    private Detection detection;

    private Summary summary;

    // --- 1. Metadata DTO ---
    @Getter @Setter @NoArgsConstructor
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Metadata {
        @JsonProperty("video_id")
        private String videoId;

        @JsonProperty("request_id")
        private String requestId;

        @JsonProperty("processed_at")
        private String processedAt;

        @JsonProperty("processing_time_ms")
        private Double processingTimeMs;

        @JsonProperty("pipeline_version")
        private String pipelineVersion;
    }

    // --- 2. VideoInfo DTO ---
    @Getter @Setter @NoArgsConstructor
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class VideoInfo {
        @JsonProperty("duration_sec")
        private Double durationSec;

        @JsonProperty("total_frames")
        private Integer totalFrames;

        private Double fps;
        private String resolution;

        @JsonProperty("original_path")
        private String originalPath;
    }

    // --- 3. Detection DTO (수정됨) ---
    @Getter @Setter @NoArgsConstructor
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Detection {
        private String verdict;
        private Double confidence;
        private Probabilities probabilities;

        // JSON에서 사라진 필드들은 DTO에서도 제거하거나 @JsonIgnore 처리해야 깔끔합니다.
        // 기존 코드 호환성을 위해 남겨둔다면 null로 들어옵니다.
        // 여기서는 JSON에 없는 필드는 제거했습니다.
    }

    @Getter @Setter @NoArgsConstructor
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Probabilities {
        private Double real;
        private Double fake;
    }

    // --- 4. Summary DTO (수정됨) ---
    @Getter @Setter @NoArgsConstructor
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Summary {
        private String title;

        @JsonProperty("risk_level")
        private String riskLevel;

        @JsonProperty("primary_reason")
        private String primaryReason;

        @JsonProperty("detailed_explanation")
        private String detailedExplanation;

        // JSON에서 사라진 suspicious_interval_count, top_suspicious_phonemes 제거
    }
}