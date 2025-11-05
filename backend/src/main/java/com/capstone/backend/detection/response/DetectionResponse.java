package com.capstone.backend.detection.response;

import lombok.Data;

@Data
public class DetectionResponse {

    private Long requestId;


//    private String fileFormat;
//    private String fileSize;


    public DetectionResponse(Long requestId) {
        this.requestId = requestId;
    }
}
