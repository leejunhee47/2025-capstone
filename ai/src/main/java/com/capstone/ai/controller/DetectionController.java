package com.capstone.ai.controller;

import com.capstone.ai.component.AiJobQueue;
import com.capstone.ai.request.DetectionRequest;
import com.capstone.ai.response.ErrorResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.io.File;


@Slf4j
@RestController
@RequestMapping("/api/ai")
public class DetectionController {

    private final AiJobQueue aiJobQueue; // 위에서 만든 큐 주입

    // [추가] 파일 저장을 위해 컨트롤러에도 uploadDir 주입
    @Value("${file.upload-dir}")
    private String uploadDir;

    DetectionController(AiJobQueue aiJobQueue){
        this.aiJobQueue = aiJobQueue;
    }

    @PostMapping("/test")
    public ResponseEntity<?> processtest(@RequestParam("requestId") String requestId ,
                                         @RequestParam("file") MultipartFile file){

        // 1. 지정된 위치에 파일 저장
        File dest;
        try {
            String originalFilename = file.getOriginalFilename() != null ? file.getOriginalFilename() : "video.mp4";
            dest = new File(uploadDir + File.separator + originalFilename);

            // 2-3. 파일 저장 (동기식으로 먼저 실행)
            file.transferTo(dest);

        }catch (Exception e) {
            log.error("파일 저장 또는 비동기 처리 시작 중 오류 발생", e);
            ErrorResponse errorResponse = new ErrorResponse(
                    "SERVER_ERROR",
                    "AI 서버 에러 입니다."
            );
            return ResponseEntity
                    .status(HttpStatus.INTERNAL_SERVER_ERROR) // 500
                    .body(errorResponse);
        }

        // 2. 큐에 요청 저장
        aiJobQueue.addJob(new DetectionRequest(requestId, dest.getPath()));

        // 3. 접수 성공
        return ResponseEntity.ok("Success");
    }

    @PostMapping("/request")
    public ResponseEntity<?> processDetection(@RequestParam("requestId") String requestId,
                                              @RequestParam("file") MultipartFile file){

        // 1. 지정된 위치에 파일 저장
        File dest;
        try {
            String originalFilename = file.getOriginalFilename() != null ? file.getOriginalFilename() : "video.mp4";
            dest = new File(uploadDir + File.separator + originalFilename);

            // 2-3. 파일 저장 (동기식으로 먼저 실행)
            file.transferTo(dest);

        }catch (Exception e) {
            log.error("파일 저장 또는 비동기 처리 시작 중 오류 발생", e);
            ErrorResponse errorResponse = new ErrorResponse(
                    "SERVER_ERROR",
                    "AI 서버 에러 입니다."
            );
            return ResponseEntity
                    .status(HttpStatus.INTERNAL_SERVER_ERROR) // 500
                    .body(errorResponse);
        }


        // 2. 큐에 요청 저장
        aiJobQueue.addJob(new DetectionRequest(requestId, dest.getPath()));

        // 3. 접수 성공
        return ResponseEntity.ok("Success");

    }
}
