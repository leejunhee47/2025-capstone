"""
간단한 Hybrid Pipeline 테스트 스크립트
리팩토링된 모듈들이 정상적으로 작동하는지 확인
"""
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.xai.hybrid_pipeline import HybridXAIPipeline

def test_pipeline():
    """파이프라인 기본 동작 테스트"""
    
    # 모델 경로 설정 (테스트 파일에서 가져옴)
    MMMS_MODEL = Path(r"E:\capstone\mobile_deepfake_detector\models\kfold\fold_1\mmms-ba_best_val_94_57.pth")
    PIA_MODEL = Path(r"E:\capstone\mobile_deepfake_detector\outputs\pia_aug50\checkpoints\best.pth")
    MMMS_CONFIG = Path("E:/capstone/mobile_deepfake_detector/configs/train_teacher_korean.yaml")
    PIA_CONFIG = Path("E:/capstone/mobile_deepfake_detector/configs/train_pia.yaml")
    
    # 테스트 비디오 (짧은 비디오 선택)
    TEST_VIDEO = Path(r"E:\capstone\real_deepfake_dataset\003.딥페이크\1.Training\원천데이터\train_탐지방해\변조영상\01_변조영상\01_변조영상\0e23d546a5f952542a00_e37c8c26b0c1c0714c74_1_0016.mp4")
    
    print("=" * 80)
    print("Hybrid Pipeline 테스트 시작")
    print("=" * 80)
    
    # 파일 존재 확인
    print("\n[0/4] 파일 존재 확인 중...")
    if not MMMS_MODEL.exists():
        print(f"❌ MMMS 모델 파일 없음: {MMMS_MODEL}")
        return False
    if not PIA_MODEL.exists():
        print(f"❌ PIA 모델 파일 없음: {PIA_MODEL}")
        return False
    if not TEST_VIDEO.exists():
        print(f"❌ 테스트 비디오 파일 없음: {TEST_VIDEO}")
        return False
    print("✅ 모든 필수 파일 존재 확인!")
    
    # 1. 파이프라인 초기화
    print("\n[1/4] 파이프라인 초기화 중...")
    try:
        pipeline = HybridXAIPipeline(
            mmms_model_path=str(MMMS_MODEL),
            pia_model_path=str(PIA_MODEL),
            mmms_config_path=str(MMMS_CONFIG),
            pia_config_path=str(PIA_CONFIG),
            device="cuda"
        )
        print("✅ 파이프라인 초기화 성공!")
    except Exception as e:
        print(f"❌ 파이프라인 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. 비디오 처리
    print(f"\n[2/4] 비디오 처리 중: {TEST_VIDEO.name}")
    try:
        result = pipeline.process_video(
            video_path=str(TEST_VIDEO),
            video_id="test_video",
            threshold=0.6,
            output_dir="outputs/xai/test",
            save_visualizations=True
        )
        print("✅ 비디오 처리 완료!")
    except Exception as e:
        print(f"❌ 비디오 처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 결과 검증
    print("\n[3/4] 결과 검증 중...")
    required_keys = ['metadata', 'detection', 'summary', 'stage1_timeline', 'stage2_interval_analysis']
    missing_keys = [key for key in required_keys if key not in result]
    
    if missing_keys:
        print(f"❌ 필수 키 누락: {missing_keys}")
        return False
    
    print("✅ 결과 구조 검증 통과!")
    
    # 4. 결과 출력
    print("\n[4/4] 결과 출력")
    print("=" * 80)
    print("테스트 결과")
    print("=" * 80)
    print(f"판정: {result['detection']['verdict'].upper()}")
    print(f"신뢰도: {result['detection']['confidence']:.1%}")
    print(f"의심 프레임 비율: {result['detection']['suspicious_frame_ratio']:.1f}%")
    print(f"의심 구간 수: {len(result['stage1_timeline']['suspicious_intervals'])}")
    print(f"Stage2 분석 구간 수: {len(result['stage2_interval_analysis'])}")
    print(f"요약: {result['summary']['title']}")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)

