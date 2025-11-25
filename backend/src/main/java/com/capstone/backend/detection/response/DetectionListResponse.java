package com.capstone.backend.detection.response;


import lombok.Builder;
import lombok.Getter;

import java.util.List;

@Getter
@Builder
public class DetectionListResponse {

    private List<DetectionBriefResponse> items;
    private PageInfo pageInfo;

    @Getter
    @Builder
    public static class PageInfo {
        private Long nextCursor;
        private boolean hasNextPage;
    }


}
