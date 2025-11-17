"""
병렬 처리를 활용한 개선된 전처리 스크립트

개선사항:
1. 멀티프로세싱으로 속도 향상 (CPU 코어 활용)
2. 재시작 가능한 체크포인트 기능
3. 상세한 진행상황 표시
4. 더 나은 에러 핸들링
5. 실시간 통계 표시
"""
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import logging
import gc
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from typing import Dict, List, Tuple, Optional

# av_module 추가
sys.path.insert(0, str(Path(__file__).parent / "av_module"))
sys.path.insert(0, str(Path(__file__).parent / "mobile_deepfake_detector"))

from src.data.preprocessing import ShortsPreprocessor, match_phoneme_to_frames
from src.utils.hybrid_phoneme_aligner_v2 import HybridPhonemeAligner

# 로깅 설정
logging.basicConfig(
    level=logging.ERROR,  # ERROR 이상만 출력 (더 조용하게)
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def process_single_video(
    args: Tuple[int, Dict, str, Dict, Path]
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    단일 비디오 처리 (멀티프로세싱용)

    Args:
        args: (idx, sample, base_path, config, output_dir)

    Returns:
        (success_dict, fail_dict): 성공/실패 정보
    """
    idx, sample, base_path, config, output_dir = args

    # Logger 초기화 (worker 프로세스용)
    logger = logging.getLogger(__name__)

    # video_id 추출
    video_id = sample['video_id']

    try:
        # Preprocessor 초기화 (각 프로세스마다)
        preprocessor = ShortsPreprocessor(config)

        # 비디오 경로
        video_path = Path(base_path) / sample['video_path']

        if not video_path.exists():
            return None, {
                'idx': idx,
                'video_id': sample['video_id'],
                'reason': 'File not found'
            }

        # 전처리
        result = preprocessor.process_video(
            str(video_path),
            extract_audio=True,
            extract_lip=True
        )

        # 유효성 검증
        if result['frames'] is None or result['audio'] is None or result['lip'] is None:
            return None, {
                'idx': idx,
                'video_id': sample['video_id'],
                'reason': 'Failed to extract features'
            }

        # 최소 길이 확인
        if result['frames'].shape[0] < 5 or result['audio'].shape[0] < 10:
            return None, {
                'idx': idx,
                'video_id': sample['video_id'],
                'reason': f'Too short (frames={result["frames"].shape[0]}, audio={result["audio"].shape[0]})'
            }

        # 저장
        output_path = output_dir / f"{idx:05d}.npz"

        # Phase 2: Extract real phoneme labels using HybridPhonemeAligner
        num_frames = result['frames'].shape[0]

        # Initialize HybridPhonemeAligner (singleton pattern to avoid re-initialization)
        if not hasattr(process_single_video, 'aligner'):
            process_single_video.aligner = HybridPhonemeAligner(
                whisper_model="base",  # Use base model for good accuracy + speed balance
                device="cuda",  # GPU mode with CUDA 12.x (2-3x faster than CPU)
                compute_type="float16"  # FP16 precision for GPU optimization
            )

        # Get actual FPS and total frames from video
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else num_frames
        cap.release()

        # Calculate actual frame indices (uniform sampling)
        # This matches the logic in VideoPreprocessor._uniform_sampling()
        if total_frames <= num_frames:
            frame_indices = np.arange(num_frames, dtype=np.float32)
        else:
            step = total_frames / num_frames
            frame_indices = np.array([int(i * step) for i in range(num_frames)], dtype=np.float32)

        # Create timestamps using actual frame indices and FPS
        timestamps = frame_indices / fps

        # Perform phoneme alignment
        try:
            logger.info(f"[1/3] Phoneme alignment started for {video_id}")
            alignment = process_single_video.aligner.align_video(video_path)

            # Extract phoneme intervals from alignment
            phoneme_intervals = [
                {'phoneme': p, 'start': s, 'end': e}
                for p, (s, e) in zip(alignment['phonemes'], alignment['intervals'])
            ]

            # Match phonemes to frames using PIA-main approach
            phoneme_labels = match_phoneme_to_frames(phoneme_intervals, timestamps)

            # Debug print
            unique_phonemes = set(phoneme_labels)
            logger.info(f"  ✓ Phoneme: shape={phoneme_labels.shape}, unique={len(unique_phonemes)}, phonemes={unique_phonemes}")

        except Exception as e:
            # Fallback to silence labels if alignment fails
            logging.warning(f"Phoneme alignment failed for {sample['video_id']}: {e}")
            phoneme_labels = np.array(['<sil>'] * num_frames, dtype=object)

        # Extract MAR geometry features (geo_dim=1, PIA paper)
        from mobile_deepfake_detector.src.utils.enhanced_mar_extractor import EnhancedMARExtractor

        if not hasattr(process_single_video, 'mar_extractor'):
            process_single_video.mar_extractor = EnhancedMARExtractor()

        try:
            logger.info(f"[2/3] MAR extraction started for {video_id}")
            mar_result = process_single_video.mar_extractor.extract_from_video(video_path, max_frames=num_frames)
            geometry = np.array(mar_result['mar_vertical']).reshape(-1, 1)  # (T, 1) - PIA paper format

            # Debug print
            nonzero_ratio = np.sum(geometry != 0) / geometry.size
            mean_val = np.mean(geometry[geometry != 0]) if nonzero_ratio > 0 else 0
            logger.info(f"  ✓ MAR: shape={geometry.shape}, nonzero={nonzero_ratio:.1%}, mean={mean_val:.4f}, range=[{np.min(geometry):.4f}, {np.max(geometry):.4f}]")
        except Exception as e:
            logger.warning(f"MAR extraction failed for {video_id}: {e}, using zeros")
            geometry = np.zeros((num_frames, 1), dtype=np.float32)

        # Extract ArcFace embeddings (512-dim)
        from mobile_deepfake_detector.src.utils.arcface_extractor import ArcFaceExtractor

        if not hasattr(process_single_video, 'arcface_extractor'):
            process_single_video.arcface_extractor = ArcFaceExtractor(device="cuda", model_name="buffalo_l")

        try:
            logger.info(f"[3/3] ArcFace extraction started for {video_id}")
            arcface = process_single_video.arcface_extractor.extract_from_video(
                video_path,
                max_frames=num_frames,
                show_progress=False
            )  # (T, 512)

            # Debug print
            zero_frames = np.sum(np.all(arcface == 0, axis=1))
            valid_frames = num_frames - zero_frames
            avg_norm = np.mean(np.linalg.norm(arcface, axis=1))
            logger.info(f"  ✓ ArcFace: shape={arcface.shape}, valid={valid_frames}/{num_frames} ({valid_frames/num_frames:.1%}), avg_norm={avg_norm:.4f}")
        except Exception as e:
            logger.warning(f"ArcFace extraction failed for {video_id}: {e}, using zeros")
            arcface = np.zeros((num_frames, 512), dtype=np.float32)

        np.savez_compressed(
            output_path,
            frames=result['frames'],
            audio=result['audio'],
            lip=result['lip'],
            arcface=arcface,                 # ✅ REAL: Per-frame ArcFace embeddings (512-dim)
            geometry=geometry,               # ✅ REAL: Per-frame MAR geometry (1-dim)
            phoneme_labels=phoneme_labels,   # ✅ REAL: Per-frame phoneme labels
            timestamps=timestamps,           # ✅ REAL: Per-frame timestamps (actual FPS)
            label=1 if sample['label'] == 'fake' else 0,
            video_id=sample['video_id']
        )

        success_dict = {
            'idx': idx,
            'output_path': str(output_path.relative_to(output_dir.parent)),
            'video_id': sample['video_id'],
            'label': sample['label'],
            'frames_count': result['frames'].shape[0],
            'audio_count': result['audio'].shape[0]
        }

        # Force garbage collection to free memory
        gc.collect()

        return success_dict, None

    except Exception as e:
        return None, {
            'idx': idx,
            'video_id': sample.get('video_id', 'unknown'),
            'reason': str(e)
        }


def load_checkpoint(checkpoint_path: Path) -> Tuple[List, List, set]:
    """체크포인트 로드"""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        successful = data.get('successful', [])
        failed = data.get('failed', [])
        processed_indices = set(data.get('processed_indices', []))

        return successful, failed, processed_indices

    return [], [], set()


def save_checkpoint(
    checkpoint_path: Path,
    successful: List,
    failed: List,
    processed_indices: set
):
    """체크포인트 저장"""
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump({
            'successful': successful,
            'failed': failed,
            'processed_indices': list(processed_indices)
        }, f, indent=2, ensure_ascii=False)


def preprocess_dataset_parallel(
    split: str = 'train',
    num_workers: int = None,
    batch_size: int = 50,
    resume: bool = True,
    max_videos: int = None
):
    """
    병렬 처리로 데이터셋 전처리

    Args:
        split: 'train', 'val', 'test'
        num_workers: 병렬 작업 수 (None = CPU 코어 수 - 1)
        batch_size: 체크포인트 저장 주기
        resume: 이전 작업 재개 여부
        max_videos: 테스트용 최대 비디오 수 (None = 전체 처리)
    """
    print(f"\n{'='*80}")
    print(f"병렬 전처리: {split.upper()} SPLIT")
    print(f"{'='*80}\n")

    # 워커 수 결정
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"설정:")
    print(f"  - CPU 코어: {cpu_count()}개")
    print(f"  - 병렬 작업자: {num_workers}개")
    print(f"  - 체크포인트 주기: {batch_size}개마다")
    print()

    # 경로 설정
    index_file = Path(f"preprocessed_data_real/{split}_index.json")  # 인덱스는 기존 사용
    output_dir = Path(f"preprocessed_data_phoneme/{split}")  # ✨ 새 폴더 (음소 라벨 포함)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(f"preprocessed_data_phoneme/{split}_checkpoint.json")  # ✨ 새 폴더

    # 인덱스 로드
    print(f"[1/5] 인덱스 로드: {index_file}")
    with open(index_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    print(f"  - 총 샘플: {len(samples)}개")

    # 체크포인트 로드
    successful, failed, processed_indices = [], [], set()
    if resume and checkpoint_path.exists():
        successful, failed, processed_indices = load_checkpoint(checkpoint_path)
        print(f"\n[체크포인트 복구]")
        print(f"  - 이미 처리됨: {len(processed_indices)}개")
        print(f"  - 성공: {len(successful)}개")
        print(f"  - 실패: {len(failed)}개")

    # 처리할 샘플 필터링
    samples_to_process = [
        (idx, sample) for idx, sample in enumerate(samples)
        if idx not in processed_indices
    ]

    # 테스트용 최대 개수 제한
    if max_videos is not None and len(samples_to_process) > max_videos:
        print(f"  [TEST] 테스트 모드: {len(samples_to_process)}개 중 {max_videos}개만 처리")
        samples_to_process = samples_to_process[:max_videos]

    if not samples_to_process:
        print("\n모든 샘플이 이미 처리되었습니다!")
        return successful, failed

    print(f"  - 처리 대기: {len(samples_to_process)}개")
    print()

    # 설정
    print("[2/5] 전처리 설정")
    config = {
        'target_fps': 30,
        'max_frames': 50,
        'frame_size': [224, 224],
        'max_duration': 60,
        'min_duration': 15,
        'sample_rate': 16000,
        'n_mfcc': 40,
        'hop_length': 512,
        'n_fft': 2048,
        'lip_size': [112, 112]
    }
    print(f"  - 프레임: {config['max_frames']}개, {config['frame_size']}")
    print(f"  - 오디오: {config['n_mfcc']} MFCC, {config['sample_rate']}Hz")
    print(f"  - 입술: {config['lip_size']}")
    print()

    # 병렬 처리 준비
    print(f"[3/5] 병렬 처리 시작 ({num_workers} workers)")
    base_path = "E:/capstone"

    # 작업 인자 준비
    tasks = [
        (idx, sample, base_path, config, output_dir)
        for idx, sample in samples_to_process
    ]

    # 시작 시간
    start_time = time.time()

    # 병렬 처리
    batch_count = 0
    try:
        with Pool(processes=num_workers) as pool:
            # imap_unordered로 결과를 순서와 상관없이 받음 (더 빠름)
            with tqdm(total=len(tasks), desc=f"Processing {split}",
                     ncols=100, unit="video") as pbar:

                for result in pool.imap_unordered(process_single_video, tasks):
                    success_dict, fail_dict = result

                    if success_dict:
                        successful.append(success_dict)
                        processed_indices.add(success_dict['idx'])

                    if fail_dict:
                        failed.append(fail_dict)
                        processed_indices.add(fail_dict['idx'])

                    pbar.update(1)

                    # 주기적 체크포인트 저장
                    batch_count += 1
                    if batch_count % batch_size == 0:
                        save_checkpoint(checkpoint_path, successful, failed, processed_indices)

                        # 진행상황 표시
                        elapsed = time.time() - start_time
                        speed = len(processed_indices) / elapsed
                        eta = (len(samples) - len(processed_indices)) / speed if speed > 0 else 0

                        pbar.set_postfix({
                            'success': len(successful),
                            'fail': len(failed),
                            'speed': f'{speed:.1f}v/s',
                            'ETA': f'{eta/60:.1f}m'
                        })

    except KeyboardInterrupt:
        print("\n\n중단됨! 체크포인트 저장 중...")
        save_checkpoint(checkpoint_path, successful, failed, processed_indices)
        print("체크포인트 저장 완료. 나중에 재개할 수 있습니다.")
        raise

    # 최종 체크포인트 저장
    save_checkpoint(checkpoint_path, successful, failed, processed_indices)

    # 통계
    elapsed_time = time.time() - start_time
    print()
    print(f"[4/5] 처리 완료 (소요 시간: {elapsed_time/60:.1f}분)")
    print(f"  - 처리 속도: {len(samples_to_process)/elapsed_time:.2f} videos/sec")
    print()

    # 결과 저장
    print(f"[5/5] 결과 저장")

    # 성공한 샘플 인덱스
    success_index_path = Path(f"preprocessed_data_phoneme/{split}_preprocessed_index.json")  # ✨ 새 폴더
    with open(success_index_path, 'w', encoding='utf-8') as f:
        json.dump(successful, f, indent=2, ensure_ascii=False)
    print(f"  [OK] 성공: {success_index_path} ({len(successful)} 샘플)")

    # 실패한 샘플 로그
    if failed:
        failed_log_path = Path(f"preprocessed_data_phoneme/{split}_failed.json")  # ✨ 새 폴더
        with open(failed_log_path, 'w', encoding='utf-8') as f:
            json.dump(failed, f, indent=2, ensure_ascii=False)
        print(f"  [XX] 실패: {failed_log_path} ({len(failed)} 샘플)")

    # 체크포인트 삭제 (완료되었으므로)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  [OK] 체크포인트 파일 삭제")

    print()
    print(f"{'='*80}")
    print(f"완료!")
    print(f"{'='*80}")
    print(f"  성공: {len(successful)}/{len(samples)} ({len(successful)/len(samples)*100:.1f}%)")
    print(f"  실패: {len(failed)}/{len(samples)} ({len(failed)/len(samples)*100:.1f}%)")
    print(f"  총 소요 시간: {elapsed_time/60:.1f}분")
    print(f"  평균 처리 속도: {len(samples_to_process)/elapsed_time:.2f} videos/sec")
    print()

    return successful, failed


def print_status():
    """현재 전처리 상태 확인"""
    print("\n" + "="*80)
    print("전처리 상태 확인")
    print("="*80 + "\n")

    for split in ['train', 'val', 'test']:
        index_file = Path(f"preprocessed_data_real/{split}_index.json")
        preprocessed_file = Path(f"preprocessed_data_real/{split}_preprocessed_index.json")
        checkpoint_file = Path(f"preprocessed_data_real/{split}_checkpoint.json")

        if not index_file.exists():
            print(f"{split.upper()}: 인덱스 파일 없음")
            continue

        with open(index_file, 'r', encoding='utf-8') as f:
            total = len(json.load(f))

        preprocessed = 0
        if preprocessed_file.exists():
            with open(preprocessed_file, 'r', encoding='utf-8') as f:
                preprocessed = len(json.load(f))

        checkpoint_count = 0
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                checkpoint_count = len(checkpoint_data.get('processed_indices', []))

        status = "[OK] 완료" if preprocessed >= total else "[>>] 진행중" if checkpoint_count > 0 else "[  ] 미처리"

        print(f"{split.upper():6s}: {status:8s} | {preprocessed:4d}/{total:4d} ({preprocessed/total*100:5.1f}%)", end="")
        if checkpoint_count > 0 and checkpoint_count != preprocessed:
            print(f" [체크포인트: {checkpoint_count}]")
        else:
            print()

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="병렬 처리 비디오 전처리",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 상태 확인
  python preprocess_parallel.py --status

  # Train split 처리 (자동으로 CPU 코어 수 감지)
  python preprocess_parallel.py --split train

  # Val split 처리 (4개 워커 사용)
  python preprocess_parallel.py --split val --workers 4

  # 모든 split 처리
  python preprocess_parallel.py --split all --workers 6

  # 처음부터 다시 시작 (체크포인트 무시)
  python preprocess_parallel.py --split train --no-resume
        """
    )

    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'val', 'test', 'all'],
                        help='처리할 split (기본: all)')
    parser.add_argument('--workers', type=int, default=None,
                        help='병렬 작업자 수 (기본: CPU 코어 - 1)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='체크포인트 저장 주기 (기본: 50)')
    parser.add_argument('--no-resume', action='store_true',
                        help='체크포인트 무시하고 처음부터 시작')
    parser.add_argument('--status', action='store_true',
                        help='현재 전처리 상태만 확인')
    parser.add_argument('--max-videos', type=int, default=None,
                        help='테스트용: 처리할 최대 비디오 수 (기본: 전체)')

    args = parser.parse_args()

    # 상태 확인만
    if args.status:
        print_status()
        sys.exit(0)

    print("\n" + "="*80)
    print("병렬 비디오 전처리 스크립트")
    print("="*80)
    print()
    print("개선사항:")
    print("  + 멀티프로세싱으로 속도 향상")
    print("  + 재시작 가능한 체크포인트")
    print("  + 실시간 진행상황 표시")
    print("  + 상세한 통계 정보")
    print()

    # Split 결정
    if args.split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]

    # 전체 통계
    total_success = 0
    total_failed = 0
    total_time = 0

    for split in splits:
        start = time.time()
        successful, failed = preprocess_dataset_parallel(
            split=split,
            num_workers=args.workers,
            batch_size=args.batch_size,
            resume=not args.no_resume,
            max_videos=args.max_videos
        )
        elapsed = time.time() - start

        total_success += len(successful)
        total_failed += len(failed)
        total_time += elapsed

    # 최종 요약
    if len(splits) > 1:
        print("\n" + "="*80)
        print("전체 전처리 완료!")
        print("="*80)
        print(f"  총 성공: {total_success}")
        print(f"  총 실패: {total_failed}")
        print(f"  성공률: {total_success/(total_success+total_failed)*100:.1f}%")
        print(f"  총 소요 시간: {total_time/60:.1f}분")
        print()

    print("다음 단계:")
    print("  1. python preprocess_parallel.py --status  # 상태 확인")
    print("  2. 전처리된 데이터로 학습 시작")
    print()
