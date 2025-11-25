package com.capstone.backend.detection.model;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Entity
@NoArgsConstructor
public class DetectionReport {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long reportId;

    // 저장 이미지 경로
    private String imgPath;

    // 이미지 순서
    private Integer sequence;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "result_id")
    private DetectionResult detectionResult;


    // 연관관계 편의 메서드
    public void setDetectionResult(DetectionResult detectionResult){
        if(this.detectionResult != null){
            this.detectionResult.getDetectionReports().remove(this);
        }
        this.detectionResult = detectionResult;
        this.detectionResult.getDetectionReports().add(this);
    }


}
