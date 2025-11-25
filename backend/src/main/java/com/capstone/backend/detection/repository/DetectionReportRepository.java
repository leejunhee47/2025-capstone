package com.capstone.backend.detection.repository;

import com.capstone.backend.detection.model.DetectionReport;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DetectionReportRepository extends JpaRepository<DetectionReport, Long> {
}
