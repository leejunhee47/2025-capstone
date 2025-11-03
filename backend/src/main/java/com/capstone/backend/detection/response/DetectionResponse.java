package com.capstone.backend.detection.response;

import lombok.Data;

@Data
public class DetectionResponse {

    private Long requestId;
    private String fileFormat;
    private String fileSize;

    public DetectionResponse(Long requestId, String fileFormat, String fileSize) {
        this.requestId = requestId;
        this.fileFormat = fileFormat;
        this.fileSize = fileSize;
    }
}
