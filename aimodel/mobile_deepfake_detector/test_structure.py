"""
리팩토링된 모듈 구조 검증 테스트
코드 구조와 import 경로만 확인 (실제 실행 없음)
"""
import ast
from pathlib import Path

def check_file_structure(file_path: Path) -> dict:
    """파일의 클래스와 함수 구조 확인"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
        
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        return {
            'exists': True,
            'classes': classes,
            'functions': functions,
            'error': None
        }
    except Exception as e:
        return {
            'exists': False,
            'classes': [],
            'functions': [],
            'error': str(e)
        }

def test_module_structure():
    """모든 리팩토링된 모듈의 구조 확인"""
    
    print("=" * 80)
    print("리팩토링된 모듈 구조 검증")
    print("=" * 80)
    
    base_path = Path(__file__).parent / "src" / "xai"
    
    modules = {
        "FeatureExtractor": base_path / "feature_extractor.py",
        "Stage1Scanner": base_path / "stage1_scanner.py",
        "Stage2Analyzer": base_path / "stage2_analyzer.py",
        "IntervalDetector": base_path / "interval_detector.py",
        "ResultAggregator": base_path / "result_aggregator.py",
        "HybridXAIPipeline": base_path / "hybrid_pipeline.py",
    }
    
    print("\n[1/3] 파일 존재 및 구조 확인...")
    all_exist = True
    for name, file_path in modules.items():
        result = check_file_structure(file_path)
        if result['exists']:
            print(f"  OK  {name:25s} -> {file_path.name}")
            print(f"       Classes: {', '.join(result['classes'])}")
            if result['functions']:
                print(f"       Functions: {', '.join(result['functions'][:5])}...")
        else:
            print(f"  FAIL {name:25s} -> {file_path.name}: {result['error']}")
            all_exist = False
    
    # Import 경로 확인
    print("\n[2/3] Import 경로 확인...")
    # hybrid_pipeline.py에서 직접 import하는 것들만 확인
    hybrid_pipeline_imports = [
        ("Stage1Scanner", "from .stage1_scanner import Stage1Scanner"),
        ("Stage2Analyzer", "from .stage2_analyzer import Stage2Analyzer"),
        ("ResultAggregator", "from .result_aggregator import ResultAggregator"),
    ]
    
    hybrid_pipeline_file = base_path / "hybrid_pipeline.py"
    if hybrid_pipeline_file.exists():
        with open(hybrid_pipeline_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for name, import_line in hybrid_pipeline_imports:
            if import_line in content:
                print(f"  OK  {name:25s} import found in hybrid_pipeline.py")
            else:
                print(f"  FAIL {name:25s} import NOT found in hybrid_pipeline.py")
                all_exist = False
        
        # Stage1Scanner가 FeatureExtractor를 import하는지 확인
        stage1_file = base_path / "stage1_scanner.py"
        if stage1_file.exists():
            with open(stage1_file, 'r', encoding='utf-8') as f:
                stage1_content = f.read()
            if "from .feature_extractor import FeatureExtractor" in stage1_content:
                print(f"  OK  FeatureExtractor      import found in stage1_scanner.py")
            else:
                print(f"  FAIL FeatureExtractor      import NOT found in stage1_scanner.py")
                all_exist = False
    
    # __init__.py 확인
    print("\n[3/3] __init__.py 확인...")
    init_file = base_path / "__init__.py"
    if init_file.exists():
        with open(init_file, 'r', encoding='utf-8') as f:
            init_content = f.read()
        
        expected_exports = [
            "HybridXAIPipeline",
            "Stage1Scanner",
            "Stage2Analyzer",
            "FeatureExtractor",
            "IntervalDetector",
            "ResultAggregator",
        ]
        
        for export in expected_exports:
            if export in init_content:
                print(f"  OK  {export:25s} exported in __init__.py")
            else:
                print(f"  FAIL {export:25s} NOT exported in __init__.py")
                all_exist = False
    else:
        print("  FAIL __init__.py not found")
        all_exist = False
    
    # Backward compatibility 확인
    print("\n[4/4] Backward Compatibility 확인...")
    compat_file = base_path / "hybrid_mmms_pia_explainer.py"
    if compat_file.exists():
        with open(compat_file, 'r', encoding='utf-8') as f:
            compat_content = f.read()
        
        if "from .hybrid_pipeline import HybridXAIPipeline" in compat_content:
            print("  OK  HybridXAIPipeline re-exported for backward compat")
        else:
            print("  FAIL HybridXAIPipeline NOT re-exported")
            all_exist = False
        
        if "from .stage1_scanner import Stage1Scanner" in compat_content:
            print("  OK  Stage1Scanner re-exported for backward compat")
        else:
            print("  FAIL Stage1Scanner NOT re-exported")
            all_exist = False
        
        if "from .stage2_analyzer import Stage2Analyzer" in compat_content:
            print("  OK  Stage2Analyzer re-exported for backward compat")
        else:
            print("  FAIL Stage2Analyzer NOT re-exported")
            all_exist = False
    else:
        print("  FAIL hybrid_mmms_pia_explainer.py not found")
        all_exist = False
    
    print("\n" + "=" * 80)
    print("검증 결과")
    print("=" * 80)
    if all_exist:
        print("SUCCESS: 모든 모듈 구조가 올바르게 구성되어 있습니다!")
        print("\n다음 단계:")
        print("1. 필요한 의존성 설치 (whisperx 등)")
        print("2. 실제 비디오로 통합 테스트 실행")
        print("3. pytest로 전체 테스트 스위트 실행")
    else:
        print("FAIL: 일부 모듈 구조에 문제가 있습니다.")
    print("=" * 80)
    
    return all_exist

if __name__ == "__main__":
    success = test_module_structure()
    exit(0 if success else 1)

