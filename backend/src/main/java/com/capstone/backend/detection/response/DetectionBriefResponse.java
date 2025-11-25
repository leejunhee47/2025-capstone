package com.capstone.backend.detection.response;

import com.capstone.backend.detection.model.DetectionResult;
import lombok.Builder;
import lombok.Getter;
import java.time.LocalDateTime;

@Getter
@Builder
public class DetectionBriefResponse {
    private Long resultId;
    private Double durationSec;
    private Double probabilityReal;
    private Double probabilityFake;
    private LocalDateTime createdAt;

    public static DetectionBriefResponse from(DetectionResult result) {
        return DetectionBriefResponse.builder()
                .resultId(result.getResultId())
                .durationSec(result.getDurationSec())
                .probabilityReal(result.getProbabilityReal())
                .probabilityFake(result.getProbabilityFake())
                .createdAt(result.getCreatedAt())
                .build();
    }
}
