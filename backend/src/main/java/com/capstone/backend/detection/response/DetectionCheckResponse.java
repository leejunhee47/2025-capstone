package com.capstone.backend.detection.response;

import lombok.Data;

@Data
public class DetectionCheckResponse {


    private Long requestId;
    public DetectionCheckResponse(Long requestId) {
        this.requestId = requestId;
    }
}
