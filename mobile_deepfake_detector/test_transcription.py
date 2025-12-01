#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test for segment transcription
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.utils.wav2vec2_korean_phoneme_aligner import Wav2Vec2KoreanPhonemeAligner

aligner = Wav2Vec2KoreanPhonemeAligner()
result = aligner.align_video('E:/capstone/real_deepfake_dataset/003.딥페이크/1.Training/원천데이터/train_탐지방해/원본영상/01_원본영상/01_원본영상/026f9b9514f28f37a3fd_009.mp4')

print('\n' + '='*60)
print('세그먼트별 결과 (처음 5개)')
print('='*60)

for i in range(min(5, len(result['vad_segments']))):
    seg_start, seg_end = result['vad_segments'][i]
    text = result['segment_texts'][i] if i < len(result['segment_texts']) else ''

    print(f'\nSegment #{i+1} [{seg_start:.3f}s - {seg_end:.3f}s]')
    print(f'  텍스트: "{text}"')

    # 해당 세그먼트의 음소 개수 세기
    phoneme_count = sum(1 for p_start, _ in result['intervals'] if seg_start <= p_start <= seg_end)
    print(f'  음소 개수: {phoneme_count}개')

print('\n' + '='*60)
print(f'총 세그먼트: {len(result["vad_segments"])}개')
print(f'총 음소: {len(result["phonemes"])}개')
print(f'처리 시간: {result["duration"]:.2f}초')
print('='*60)
