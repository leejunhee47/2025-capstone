package com.capstone.backend.detection.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class DetectionImgUrlResponse {

    private Long resultId;
    private LocalDateTime createdAt;
    private List<ReportImageInfo> images;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ReportImageInfo {
        private Integer sequence;
        private String url; // 클라이언트가 접근 가능한 이미지 URL
    }
}
