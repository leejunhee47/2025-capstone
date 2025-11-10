package com.capstone.backend.detection.response;

import lombok.Data;

@Data
public class DetectionResponse {

    private Long requestId;





    public DetectionResponse(Long requestId) {
        this.requestId = requestId;
    }
}
