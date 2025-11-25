package com.capstone.backend.detection.repository;

import com.capstone.backend.detection.model.DetectionResult;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface DetectionResultRepository extends JpaRepository<DetectionResult, Long> {

    Optional<DetectionResult> findByDetectionRequest_RequestId(Long requestId);
    // 1. 첫 페이지 조회 (커서 없음) : 멤버의 기록을 최신순(ID 내림차순)으로 조회
    List<DetectionResult> findByDetectionRequest_Member_memberIdOrderByResultIdDesc(Long memberId, Pageable pageable);
    // 2. 다음 페이지 조회 (커서 있음) : 커서보다 작은 ID를 가진 기록을 최신순으로 조회
    List<DetectionResult> findByResultIdLessThanAndDetectionRequest_Member_memberIdOrderByResultIdDesc(Long resultId, Long memberId, Pageable pageable);

}
