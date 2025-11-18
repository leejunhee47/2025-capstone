package com.capstone.backend.detection.response;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@JsonIgnoreProperties(ignoreUnknown = true)
public class AnalysisResultResponse {

    @JsonProperty("file_size_bytes")
    private long fileSize;

    @JsonProperty("duration_seconds")
    private double duration;

    private String resolution;
    private int width;
    private int height;
    private double fps;

    @JsonProperty("total_frames")
    private int totalFrames;
}
