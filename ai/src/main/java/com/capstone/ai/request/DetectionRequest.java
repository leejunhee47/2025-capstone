package com.capstone.ai.request;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;

@Data
@Slf4j
public class DetectionRequest {

    private String requestId;
    private String videoPath;

    public DetectionRequest(String requestId, String videoPath) {
        this.requestId = requestId;
        this.videoPath = videoPath;
    }
}
