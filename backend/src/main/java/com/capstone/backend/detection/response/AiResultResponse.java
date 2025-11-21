package com.capstone.backend.detection.response;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@JsonIgnoreProperties(ignoreUnknown = true) // JSON에 있지만 DTO에 없는 필드는 무시
public class AiResultResponse {

    // JSON의 "request_id" 필드를 Java의 "requestId" 필드에 매핑
    @JsonProperty("request_id")
    private String requestId;

    private String status; // "success" 등

    @JsonProperty("source_file")
    private String sourceFile;

    @JsonProperty("analysis_results")
    private AnalysisResultResponse analysisResultResponse;
}
