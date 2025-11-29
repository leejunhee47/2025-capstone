"""
리팩토링된 Hybrid Pipeline 초기화 테스트
실제 실행 없이 초기화만 확인
"""
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_pipeline_initialization():
    """리팩토링된 파이프라인 초기화 테스트"""
    
    print("=" * 80)
    print("리팩토링된 Hybrid Pipeline 초기화 테스트")
    print("=" * 80)
    
    # 모델 경로 설정
    MMMS_MODEL = Path(r"E:\capstone\mobile_deepfake_detector\models\kfold\fold_1\mmms-ba_best_val_94_57.pth")
    PIA_MODEL = Path(r"E:\capstone\mobile_deepfake_detector\outputs\pia_aug50\checkpoints\best.pth")
    MMMS_CONFIG = Path("E:/capstone/mobile_deepfake_detector/configs/train_teacher_korean.yaml")
    PIA_CONFIG = Path("E:/capstone/mobile_deepfake_detector/configs/train_pia.yaml")
    
    # 파일 존재 확인
    print("\n[1/4] 파일 존재 확인...")
    if not MMMS_MODEL.exists():
        print(f"  SKIP: MMMS 모델 파일 없음: {MMMS_MODEL}")
        return False
    if not PIA_MODEL.exists():
        print(f"  SKIP: PIA 모델 파일 없음: {PIA_MODEL}")
        return False
    print("  OK: 모든 모델 파일 존재")
    
    # Import 테스트
    print("\n[2/4] 모듈 Import 테스트...")
    try:
        from src.xai.hybrid_pipeline import HybridXAIPipeline
        from src.xai.stage1_scanner import Stage1Scanner
        from src.xai.stage2_analyzer import Stage2Analyzer
        print("  OK: 모든 모듈 import 성공")
    except Exception as e:
        print(f"  FAIL: Import 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Stage1Scanner 초기화 테스트
    print("\n[3/4] Stage1Scanner 초기화 테스트...")
    try:
        stage1 = Stage1Scanner(
            model_path=str(MMMS_MODEL),
            config_path=str(MMMS_CONFIG),
            device="cuda"
        )
        print("  OK: Stage1Scanner 초기화 성공")
    except Exception as e:
        print(f"  FAIL: Stage1Scanner 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # HybridXAIPipeline 초기화 테스트
    print("\n[4/4] HybridXAIPipeline 초기화 테스트...")
    try:
        pipeline = HybridXAIPipeline(
            mmms_model_path=str(MMMS_MODEL),
            pia_model_path=str(PIA_MODEL),
            mmms_config_path=str(MMMS_CONFIG),
            pia_config_path=str(PIA_CONFIG),
            device="cuda"
        )
        print("  OK: HybridXAIPipeline 초기화 성공")
        print(f"  - Stage1: {type(pipeline.stage1).__name__}")
        print(f"  - Stage2: {type(pipeline.stage2).__name__}")
        print(f"  - Aggregator: {type(pipeline.aggregator).__name__}")
    except Exception as e:
        print(f"  FAIL: HybridXAIPipeline 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("SUCCESS: 리팩토링된 파이프라인이 정상적으로 초기화됩니다!")
    print("=" * 80)
    print("\n다음 단계:")
    print("1. 실제 비디오로 전체 파이프라인 테스트 실행")
    print("2. pytest test_hybrid_pipeline.py -v 로 전체 테스트 스위트 실행")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = test_pipeline_initialization()
    sys.exit(0 if success else 1)

