package com.capstone.backend.detection.repository;

import com.capstone.backend.detection.model.DetectionRequest;
import com.capstone.backend.member.model.Member;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface DetectionRequestRepository extends JpaRepository<DetectionRequest, Long> {
	List<DetectionRequest> findAllByMemberOrderByCreatedAtDesc(Member member);
}
