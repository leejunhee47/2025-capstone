package com.capstone.backend.detection.repository;

import com.capstone.backend.detection.model.DetectionRequest;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DetectionRequestRepository extends JpaRepository<DetectionRequest, Long> {
}
