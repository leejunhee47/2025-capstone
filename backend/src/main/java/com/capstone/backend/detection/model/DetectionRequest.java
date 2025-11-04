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

    @Column(nullable = false)
    private String originalFilename;

    @Column(nullable = false)
    private String fileFormat;

    @Column(nullable = false)
    private Long fileSize;

    @Column // S3 저장위치 , 처음엔 null
    private String storedFilePath;


    @CreationTimestamp
    @Column(updatable = false, nullable = false)
    private LocalDateTime createdAt;


    @Builder
    public DetectionRequest(Member member, String callbackUrl, String originalFilename,
                            String fileFormat, Long fileSize) {


        this.member = member;
        this.callbackUrl = callbackUrl;
        this.originalFilename = originalFilename;
        this.fileFormat = fileFormat;
        this.fileSize = fileSize;

        this.status = DetectionStatus.PENDING;
    }

}
