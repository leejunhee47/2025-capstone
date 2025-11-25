package com.capstone.backend.detection.model;

import com.capstone.backend.member.model.Member;
import jakarta.persistence.*;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;


@Data
@Entity
@NoArgsConstructor
public class DetectionRequest {


    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long requestId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "member_id", nullable = false)
    private Member member;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private DetectionStatus status;

    @Column(nullable = false, length = 2048)
    private String callbackUrl;


    // 영상 저장 후

//    @Column
//    private String originalFilename;
//
//    @Column
//    private String fileFormat;
//
//    @Column
//    private Long fileSize;

    @Column // 원본 영상 경로
    private String videoPath;

    @Column // 썸네일 경로
    private String thumbnailPath;


    @Column // 분석 결과 경로
    private String resultPath;

    // --------

    @CreationTimestamp
    @Column(updatable = false, nullable = false)
    private LocalDateTime createdAt;


    @Builder
    public DetectionRequest(Member member, String callbackUrl) {
        this.member = member;
        this.callbackUrl = callbackUrl;
        this.status = DetectionStatus.PENDING;
    }

}
