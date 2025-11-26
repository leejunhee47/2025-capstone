package com.capstone.backend.detection.controller;

import com.capstone.backend.detection.model.DetectionRequest;
import com.capstone.backend.detection.model.DetectionStatus;
import com.capstone.backend.detection.response.DetectionBriefResponse;
import com.capstone.backend.detection.service.DetectionService;
import com.capstone.backend.member.model.Member;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpSession;
import lombok.extern.slf4j.Slf4j;
import org.apache.tika.Tika;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;

@Slf4j
@Controller
@RequestMapping("/api/test/detection")
public class DetectionTestController {

    private final Tika tika = new Tika();

    private final DetectionService detectionService;

    @Value("${file.upload-dir}")
    private String uploadDir;

    @Value("${result.img}")
    private String resultImgDir;

    @Autowired
    public DetectionTestController(DetectionService detectionService) { this.detectionService = detectionService; }

    /**
     * 1. 영상 업로드 폼 화면 보여주기 (GET)
     */
    @GetMapping("/upload")
    public String uploadForm(Model model) {
        // 테스트 편의를 위해 기본 콜백 URL을 미리 넣어줄 수 있습니다.
        model.addAttribute("defaultCallbackUrl", "http://localhost:8080/api/callback/test");
        return "test/upload";
    }


    /**
     * 2. 영상 업로드 처리 (POST)
     */
    @PostMapping("/upload")
    public String processUploadDetectionRequest(
            @RequestParam("file") MultipartFile file,
            @RequestParam("callbackUrl") String callbackUrl,
            @SessionAttribute(name = "loginMember", required = false) Member loginMember,
            Model model) {

        log.info("upload call");

        // 1. 세션 유효성 검사
        if (loginMember == null) {
            return "redirect:/api/test/member/login"; // 로그인 페이지로 리다이렉트
        }


        // 1-2. 영상 유효성 체크
        String mimeType = validateAndGetMimeType(file);
        if (mimeType == null || !mimeType.startsWith("video/")) {
            log.warn("유효하지 않은 파일 수신: {}", file.getOriginalFilename());
            model.addAttribute("errorMessage", "유효하지 않은 파일이거나 비디오 파일이 아닙니다.");
            model.addAttribute("defaultCallbackUrl", callbackUrl);
            return "test/upload";
        }

        // 2. DB 에 요청 저장 및 파일 저장
        try {
            // 2-1. DB에 먼저 저장 (Service 메서드 호출)
            // 반환 타입인 DetectionRequest(혹은 엔티티)를 받아서 ID를 추출한다고 가정합니다.
            DetectionRequest newRequest = detectionService.saveNewDetectionRequest(loginMember, callbackUrl);

            // 2-2. 파일 저장 경로 설정
            // newRequest.getRequestId() 메서드가 있다고 가정합니다.
            String originalFilename = file.getOriginalFilename() != null ? file.getOriginalFilename() : "video.mp4";
            String uniqueFileName = newRequest.getRequestId() + "_" + originalFilename;
            File dest = new File(uploadDir + File.separator + uniqueFileName);

            // 디렉토리가 없으면 생성
            if (!dest.getParentFile().exists()) {
                dest.getParentFile().mkdirs();
            }

            // 2-3. 파일 저장 (동기식)
            file.transferTo(dest);


            // 3. 비동기 처리 시작
            detectionService.startFileDetection(newRequest, dest.getPath());

            // 성공 시 메시지를 담아 홈으로 가거나 결과 페이지로 이동
            model.addAttribute("message", "영상 업로드가 완료되었습니다! 분석이 시작됩니다. (ID: " + newRequest.getRequestId() + ")");

            return "redirect:/api/test/detection/result/" + newRequest.getRequestId();

        } catch (Exception e) {
            log.error("파일 저장 또는 비동기 처리 시작 중 오류 발생", e);
            model.addAttribute("errorMessage", "서버 에러가 발생했습니다: " + e.getMessage());
            model.addAttribute("defaultCallbackUrl", callbackUrl);
            return "test/upload";
        }
    }


    /**
     * 3. AI -> 백엔드는 실제 컨트롤러 사용
     */


    /**
     * 3. 결과 페이지 보여주기 (GET)
     * 업로드 직후 이동할 페이지
     */
    @GetMapping("/result/{requestId}")
    public String resultPage(@PathVariable("requestId") Long requestId, Model model) {
        // 뷰(HTML)에서 버튼 링크를 만들 때 사용할 requestId를 모델에 담습니다.
        model.addAttribute("requestId", requestId);
        return "test/result";
    }

    /**
     * [추가] 간단 결과 조회 (Polling 로직 테스트)
     * DTO: DetectionBriefResponse (resultId, durationSec, probabilityReal, probabilityFake, createdAt)
     */
    @GetMapping("/result/simple/{requestId}")
    public String simpleResult(@PathVariable("requestId") Long requestId, Model model) {

        model.addAttribute("requestId", requestId);

        try {
            // 1. Service에서 요청 정보 조회 (상태 확인용)
            DetectionRequest request = detectionService.getDetectionByRequestId(requestId);
            DetectionStatus status = request.getStatus();

            model.addAttribute("status", status); // 상태값 전달 (COMPLETED, PENDING 등)

            // 2. 상태에 따른 분기 처리
            switch (status) {
                case COMPLETED:
                    // 완료 시: DetectionBriefResponse 객체를 받아와서 모델에 담음
                    DetectionBriefResponse response = detectionService.getDetectionBriefResponse(requestId);
                    model.addAttribute("result", response);
                    break;

                case PENDING:
                case PROCESSING:
                    // 진행 중일 때 보여줄 메시지
                    model.addAttribute("message", "현재 AI가 영상을 분석하고 있습니다. 잠시 후 '새로고침'을 눌러주세요.");
                    break;

                case FAILED:
                    model.addAttribute("errorMessage", "분석에 실패했습니다. (서버 오류 또는 AI 처리 실패)");
                    break;
            }

        } catch (Exception e) {
            log.error("테스트 페이지 조회 중 에러", e);
            model.addAttribute("errorMessage", "요청하신 ID를 찾을 수 없거나 에러가 발생했습니다.");
        }

        return "test/simple-result";
    }

    /**
     * [수정] 썸네일 확인 페이지
     * 역할: RequestId를 통해 ResultId를 구하고,
     * 1) GET /api/v1/record/{resultId}/thumbnail (최종 결과 조회용)
     * 2) POST /api/v1/detection/thumbnail/polling (폴링용)
     * 두 가지를 모두 테스트할 수 있는 페이지로 이동합니다.
     */
    @GetMapping("/result/thumbnail/{requestId}")
    public String thumbnailResult(@PathVariable("requestId") Long requestId, Model model) {
        model.addAttribute("requestId", requestId);

        try {
            // 1. 요청 상태 조회
            DetectionRequest request = detectionService.getDetectionByRequestId(requestId);
            DetectionStatus status = request.getStatus();
            model.addAttribute("status", status);

            // 2. 완료 상태라면 ResultId 추출 (GET 방식 테스트를 위해)
            if (status == DetectionStatus.COMPLETED) {
                DetectionBriefResponse brief = detectionService.getDetectionBriefResponse(requestId);
                model.addAttribute("resultId", brief.getResultId());
            } else if (status == DetectionStatus.PROCESSING) {
                model.addAttribute("message", "아직 분석 중입니다. (폴링 방식 테스트 가능)");
            } else {
                model.addAttribute("message", "분석 실패 또는 대기 중입니다.");
            }

        } catch (Exception e) {
            log.error("썸네일 페이지 진입 중 에러", e);
            model.addAttribute("errorMessage", "데이터 조회 중 에러가 발생했습니다.");
        }

        return "test/thumbnail-result"; // 아래 작성할 HTML로 이동
    }

    /**
     * [수정] 상세 결과 조회 페이지 (GET)
     * 역할: requestId를 받아서, 실제 API 호출에 필요한 resultId를 찾고, HTML을 반환한다.
     * 실제 데이터 로딩: HTML 내의 JS가 '/api/v1/record/{resultId}'를 호출함.
     */
    @GetMapping("/result/detail/{requestId}")
    public String detailResult(@PathVariable("requestId") Long requestId, Model model) {

        model.addAttribute("requestId", requestId);

        try {
            // 1. 상태 확인
            DetectionRequest request = detectionService.getDetectionByRequestId(requestId);
            DetectionStatus status = request.getStatus();
            model.addAttribute("status", status);

            // 2. 완료 상태라면 ResultId를 조회하여 모델에 담음
            if (status == DetectionStatus.COMPLETED) {
                // (BriefResponse 등에서 resultId를 추출하는 방식 사용)
                DetectionBriefResponse brief = detectionService.getDetectionBriefResponse(requestId);
                Long resultId = brief.getResultId();

                model.addAttribute("resultId", resultId); // View에서 JS가 사용할 ID
            } else {
                model.addAttribute("message", "분석이 완료되지 않아 상세 결과를 조회할 수 없습니다.");
            }

        } catch (Exception e) {
            log.error("상세 페이지 진입 중 에러", e);
            model.addAttribute("errorMessage", "데이터 준비 중 오류가 발생했습니다.");
        }

        return "test/detail-result";
    }


    /**
     * [추가] 결과 보고서(이미지 리스트) 테스트 페이지
     * 역할: requestId를 받아 resultId를 조회한 뒤, HTML 폼으로 넘겨줌.
     * 실제 데이터 로딩: HTML 내 JS가 '/api/v1/record/{resultId}/report'를 호출.
     */
    @GetMapping("/result/report/{requestId}")
    public String reportResult(@PathVariable("requestId") Long requestId, Model model) {
        model.addAttribute("requestId", requestId);

        try {
            // 1. 상태 확인
            DetectionRequest request = detectionService.getDetectionByRequestId(requestId);
            DetectionStatus status = request.getStatus();
            model.addAttribute("status", status);

            // 2. 완료 상태라면 ResultId를 조회하여 모델에 담음
            if (status == DetectionStatus.COMPLETED) {
                // BriefResponse 등을 통해 resultId 추출 (Service에 해당 기능이 있다고 가정)
                DetectionBriefResponse brief = detectionService.getDetectionBriefResponse(requestId);

                // 뷰로 resultId 전달 (AJAX 호출용)
                model.addAttribute("resultId", brief.getResultId());
            } else {
                model.addAttribute("message", "분석이 완료되지 않아 보고서를 조회할 수 없습니다.");
            }

        } catch (Exception e) {
            log.error("보고서 페이지 진입 중 에러", e);
            model.addAttribute("errorMessage", "데이터 준비 중 오류가 발생했습니다.");
        }

        return "test/report-result"; // 아래 작성할 HTML 파일명
    }

    /**
     * [추가] 유저별 결과 조회 테스트 페이지 (View 반환)
     */
    @GetMapping("/records")
    public String userRecords(Model model) {
        return "test/user-records";
    }

    /**
     * [추가] 테스트용 강제 로그인 (세션 생성)
     * 역할: 원하는 memberId를 입력하면, 해당 ID를 가진 Member 객체를 세션에 심어줍니다.
     * 이를 통해 실제 로그인을 거치지 않고도 /api/v1/record API를 테스트할 수 있습니다.
     */
    @PostMapping("/auth/mock")
    @ResponseBody
    public String mockLogin(@RequestParam("memberId") Long memberId, HttpServletRequest request) {
        // 1. 임의의 Member 객체 생성 (테스트용)
        // 실제로는 DB에서 조회해야 정확하지만, Service가 ID만 사용한다면 이것으로 충분할 수 있습니다.
        // 필요하다면 MemberRepository를 주입받아 실제 조회 로직으로 변경하세요.
        Member mockMember = new Member();
        mockMember.setMemberId(memberId);
        // mockMember.setEmail("test@example.com"); // 필요 시 추가 설정

        // 2. 세션에 저장 (키 값은 "loginMember"로 가정)
        // DetectionController에서 사용하는 @SessionAttribute 이름과 일치해야 합니다.
        HttpSession session = request.getSession();
        session.setAttribute("loginMember", mockMember);

        return "Login Success as Member ID: " + memberId;
    }


    // --- Helper Methods ---
    private String validateAndGetMimeType(MultipartFile file) {
        try {
            // Tika를 사용하여 실제 파일 내용 기반 감지
            return tika.detect(file.getInputStream());
        } catch (IOException e) {
            return null;
        }
    }


}
