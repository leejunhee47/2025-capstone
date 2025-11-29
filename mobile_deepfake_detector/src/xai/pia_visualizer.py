"""
PIA XAI Visualizer for Korean Deepfake Detection

Provides comprehensive visualization of XAI results:
- Branch contribution comparison (Visual, Geometry, Identity)
- Phoneme attention distribution analysis
- Temporal heatmap visualization (14 phonemes × 5 frames)
- Geometry (MAR) analysis and abnormality detection
- 4-way comparison (Random/Trained/Real/Fake)
- Interactive HTML dashboard generation
"""

import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Import Korean phoneme filtering
from ..utils.korean_phoneme_config import get_phoneme_vocab, is_kept_phoneme, phoneme_to_korean

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)


class PIAVisualizer:
    """
    Visualizer for PIA model XAI results.

    Supports:
    - Individual result visualization
    - Multi-case comparison (Random/Trained/Real/Fake)
    - High-quality PNG export (300 DPI)
    - Interactive HTML dashboard (plotly)

    Args:
        korean_font: Font name for Korean text (default: 'Malgun Gothic')
        figsize: Default figure size tuple (width, height)
        dpi: Output resolution for PNG files
        style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
    """

    def __init__(
        self,
        korean_font: str = 'Malgun Gothic',
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 300,
        style: str = 'whitegrid'
    ):
        self.korean_font = korean_font
        self.figsize = figsize
        self.dpi = dpi

        # Apply seaborn style
        sns.set_style(style)

        # Configure Korean font
        self._setup_korean_font()

        # Define color schemes
        self.colors = {
            'visual': '#3498db',      # Blue
            'geometry': '#2ecc71',    # Green
            'identity': '#e67e22',    # Orange
            'real': '#27ae60',        # Dark green
            'fake': '#e74c3c',        # Red
            'high': '#e74c3c',        # High importance - red
            'medium': '#f39c12',      # Medium importance - orange
            'low': '#95a5a6'          # Low importance - gray
        }

        # Korean phoneme to Korean character mapping
        self.phoneme_to_korean = {
            'A': 'ㅏ', 'B': 'ㅂ', 'BB': 'ㅃ', 'CHh': 'ㅊ',
            'E': 'ㅔ/ㅐ', 'EU': 'ㅡ', 'I': 'ㅣ', 'M': 'ㅁ',
            'O': 'ㅗ', 'Ph': 'ㅍ', 'U': 'ㅜ',
            'iA': 'ㅑ', 'iO': 'ㅛ', 'iU': 'ㅠ'
        }

    def _setup_korean_font(self):
        """Configure matplotlib to use Korean font."""
        try:
            plt.rcParams['font.family'] = self.korean_font
            plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
        except Exception as e:
            print(f"Warning: Failed to set Korean font '{self.korean_font}': {e}")
            print("Falling back to default font (Korean text may not display correctly)")

    def load_xai_result(self, json_path: str) -> Dict[str, Any]:
        """
        Load XAI result from JSON file.

        Args:
            json_path: Path to XAI result JSON file

        Returns:
            Dictionary containing XAI result data
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_multiple_results(self, result_paths: Dict[str, str]) -> Dict[str, Dict]:
        """
        Load multiple XAI results for comparison.

        Args:
            result_paths: Dict mapping case names to JSON file paths
                         e.g., {'random': 'path1.json', 'trained': 'path2.json', ...}

        Returns:
            Dictionary mapping case names to XAI result data
        """
        results = {}
        for case_name, json_path in result_paths.items():
            try:
                results[case_name] = self.load_xai_result(json_path)
                print(f"[OK] Loaded {case_name}: {json_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load {case_name}: {e}")

        return results

    def plot_branch_contributions(
        self,
        result: Dict[str, Any],
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot branch contributions as a pie chart.

        Args:
            result: XAI result dictionary
            ax: Matplotlib axes (if None, creates new figure)
            title: Plot title (if None, auto-generates from video info)

        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure

        # Extract branch contributions
        contributions = result['model_info']['branch_contributions']
        labels = ['시각 정보\n(입 모양)', '기하학 정보\n(MAR)', '신원 정보\n(얼굴)']
        sizes = [
            contributions['visual'] * 100,
            contributions['geometry'] * 100,
            contributions['identity'] * 100
        ]
        colors = [self.colors['visual'], self.colors['geometry'], self.colors['identity']]

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 11}
        )

        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
            autotext.set_fontsize(12)

        # Set title
        if title is None:
            video_id = result['video_info']['video_id']
            pred_label = result['detection']['prediction_label']
            confidence = result['detection']['confidence'] * 100
            title = f"Branch 기여도 분석\n{video_id}\n예측: {pred_label} ({confidence:.1f}%)"

        ax.set_title(title, fontsize=14, weight='bold', pad=20)

        return fig

    def plot_phoneme_attention(
        self,
        result: Dict[str, Any],
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        top_n: int = 14
    ) -> plt.Figure:
        """
        Plot phoneme attention distribution as a bar chart.

        Args:
            result: XAI result dictionary
            ax: Matplotlib axes (if None, creates new figure)
            title: Plot title
            top_n: Number of top phonemes to show (default: all 14)

        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Extract phoneme scores
        phoneme_data = result['phoneme_analysis']['phoneme_scores']

        # Filter to keep only Korean phonemes in KEEP_PHONEMES_KOREAN (14 phonemes)
        filtered_data = [
            p for p in phoneme_data
            if is_kept_phoneme(p.get('phoneme_mfa', p.get('phoneme', '')))
        ]

        # Sort by score (descending) and take top N
        sorted_data = sorted(filtered_data, key=lambda x: x['score'], reverse=True)[:top_n]

        # Prepare data for plotting
        phonemes = [f"{p['phoneme']}\n({p['phoneme_mfa']})" for p in sorted_data]
        scores = [p['score'] for p in sorted_data]
        importance_levels = [p['importance_level'] for p in sorted_data]

        # Map importance to colors
        bar_colors = [self.colors[level] for level in importance_levels]

        # Create bar chart
        bars = ax.barh(phonemes, scores, color=bar_colors, edgecolor='black', linewidth=0.5)

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            if score > 0.01:  # Only label significant scores
                ax.text(
                    score + 0.01, i, f'{score:.4f}',
                    va='center', fontsize=9, weight='bold'
                )

        # Formatting
        ax.set_xlabel('Attention Score', fontsize=12, weight='bold')
        ax.set_ylabel('Phoneme (음소)', fontsize=12, weight='bold')
        ax.set_xlim(0, max(scores) * 1.15 if scores else 1.0)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # Set title
        if title is None:
            title = '음소별 Attention 분포'
        ax.set_title(title, fontsize=14, weight='bold', pad=15)

        # Add legend for importance levels
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['high'], edgecolor='black', label='High (0.5+)'),
            Patch(facecolor=self.colors['medium'], edgecolor='black', label='Medium (0.1-0.5)'),
            Patch(facecolor=self.colors['low'], edgecolor='black', label='Low (<0.1)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        plt.tight_layout()
        return fig

    def plot_temporal_heatmap(
        self,
        result: Dict[str, Any],
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot temporal attention heatmap (14 phonemes × 5 frames).

        Args:
            result: XAI result dictionary
            ax: Matplotlib axes (if None, creates new figure)
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        # Extract temporal heatmap data
        heatmap_data = np.array(result['temporal_analysis']['heatmap'])  # (14, 5)

        # Get phoneme labels (filter to keep only KEEP_PHONEMES_KOREAN)
        phoneme_data = result['phoneme_analysis']['phoneme_scores']
        filtered_phonemes = [
            p for p in phoneme_data
            if is_kept_phoneme(p.get('phoneme_mfa', p.get('phoneme', '')))
        ]
        phoneme_labels = [
            f"{p['phoneme']} ({p['phoneme_mfa']})"
            for p in filtered_phonemes
        ]

        # Create heatmap
        sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap='YlOrRd',
            cbar_kws={'label': 'Attention Score'},
            yticklabels=phoneme_labels,
            xticklabels=[f'Frame {i+1}' for i in range(5)],
            annot=True,
            fmt='.3f',
            linewidths=0.5,
            linecolor='gray'
        )

        # Formatting
        ax.set_xlabel('Frames per Phoneme', fontsize=12, weight='bold')
        ax.set_ylabel('Phoneme (음소)', fontsize=12, weight='bold')

        # Set title
        if title is None:
            title = '시간별 Attention 분포 (Temporal Heatmap)'
        ax.set_title(title, fontsize=14, weight='bold', pad=15)

        plt.tight_layout()
        return fig

    def plot_geometry_analysis(
        self,
        result: Dict[str, Any],
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot geometry (MAR) analysis with abnormality detection using bar chart.

        Args:
            result: XAI result dictionary
            ax: Matplotlib axes (if None, creates new figure)
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 8))
        else:
            fig = ax.figure

        # Extract geometry data
        geom_analysis = result['geometry_analysis']
        mean_mar = geom_analysis['mean_mar']
        std_mar = geom_analysis['std_mar']
        abnormal_phonemes = geom_analysis.get('abnormal_phonemes', [])
        baseline_info = geom_analysis.get('baseline_info', {})

        # [NEW] In-video baseline 체크
        use_in_video = baseline_info.get('type') == 'in_video' if baseline_info else False

        if use_in_video:
            # In-video baseline: 영상 자체의 mean/std 사용
            print(f"[VISUALIZER] Using in-video baseline (mean={mean_mar:.3f}, std={std_mar:.3f})")
            phoneme_baseline = {}  # 개별 음소 baseline 사용 안함
        else:
            # [ORIGINAL] 음소별 global baseline 로드
            baseline_path = Path(__file__).parent.parent.parent / 'mar_baseline_pia_real_fixed.json'
            phoneme_baseline = {}
            try:
                with open(baseline_path, 'r', encoding='utf-8') as f:
                    baseline_data = json.load(f)
                    phoneme_baseline = baseline_data.get('phoneme_stats', {})
            except Exception as e:
                print(f"[WARNING] Failed to load baseline: {e}")

        # 모든 phoneme에 대한 정보 수집 (abnormal + normal)
        # phoneme_vocab에서 모든 phoneme 가져오기
        phoneme_vocab = get_phoneme_vocab()

        # abnormal_phonemes를 딕셔너리로 변환 (빠른 조회용)
        abnormal_dict = {p['phoneme_mfa']: p for p in abnormal_phonemes}

        # 시각화용 데이터 준비
        phoneme_names = []
        measured_mar_values = []
        expected_min_values = []
        expected_max_values = []
        expected_mean_values = []
        colors = []
        z_scores = []

        for phoneme_mfa in phoneme_vocab:
            phoneme_kr = phoneme_to_korean(phoneme_mfa)
            phoneme_names.append(phoneme_kr)

            # [FIX] baseline 계산 (in-video vs global)
            if use_in_video:
                # In-video baseline: 영상 평균 ± 2 표준편차
                baseline_mean = mean_mar
                effective_std = std_mar if std_mar > 0.01 else 0.1
                baseline_p10 = max(0, mean_mar - 2 * effective_std)
                baseline_p90 = mean_mar + 2 * effective_std
            else:
                # Global baseline: 음소별 기대값
                baseline_stats = phoneme_baseline.get(phoneme_mfa, {})
                baseline_mean = baseline_stats.get('mean', mean_mar)
                baseline_p10 = baseline_stats.get('p10', baseline_mean - 0.1)
                baseline_p90 = baseline_stats.get('p90', baseline_mean + 0.1)

            if phoneme_mfa in abnormal_dict:
                # 이상 phoneme
                abnormal = abnormal_dict[phoneme_mfa]
                measured_mar_values.append(abnormal['measured_mar'])
                expected_min_values.append(abnormal['expected_range'][0])
                expected_max_values.append(abnormal['expected_range'][1])
                expected_mean_values.append(abnormal['expected_mean'])
                z_score = abs(abnormal.get('z_score', 0))
                z_scores.append(z_score)

                # Z-score에 따라 색상 결정
                if z_score > 3:
                    colors.append('#d32f2f')  # 진한 빨강 (critical)
                elif z_score > 2:
                    colors.append('#ff6f00')  # 주황 (high)
                else:
                    colors.append('#fbc02d')  # 노랑 (medium)
            else:
                # 정상 phoneme - baseline 사용
                measured_mar_values.append(baseline_mean)  # baseline 평균으로 표시
                expected_min_values.append(baseline_p10)
                expected_max_values.append(baseline_p90)
                expected_mean_values.append(baseline_mean)
                z_scores.append(0)
                colors.append('#4caf50')  # 녹색 (정상)
        
        # Bar chart 그리기
        x_pos = np.arange(len(phoneme_names))
        width = 0.6
        
        # 먼저 기대 범위 배경 표시 (모든 phoneme에 대해)
        for i, (name, min_val, max_val, mean_val) in enumerate(
            zip(phoneme_names, expected_min_values, expected_max_values, expected_mean_values)
        ):
            if min_val is not None and max_val is not None:
                # P10-P90 범위 (회색 영역)
                ax.bar(i, max_val - min_val, width, bottom=min_val,
                       color='lightgray', alpha=0.3, edgecolor='gray', linewidth=0.5, zorder=1)
                # 기대 평균 (점선)
                ax.plot([i-width/2, i+width/2], [mean_val, mean_val],
                       color='blue', linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)
        
        # 측정값 막대
        for i, (name, measured, color, z_score) in enumerate(zip(phoneme_names, measured_mar_values, colors, z_scores)):
            if z_score > 0:
                # 이상 phoneme - 측정값 막대 (색상으로 심각도 표시)
                ax.bar(
                    i, measured, width,
                    color=color, alpha=0.7,
                    edgecolor='black', linewidth=1.5,
                    zorder=3
                )
            else:
                # 정상 phoneme - baseline 평균으로 작은 녹색 막대 표시
                ax.bar(
                    i, measured, width * 0.5,
                    color='#4caf50', alpha=0.4,
                    edgecolor='gray', linewidth=0.5,
                    zorder=2
                )
        
        # X축 설정
        ax.set_xlabel('음소 (Phoneme)', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAR 값 (Mouth Aspect Ratio)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(phoneme_names, rotation=0, ha='center', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 범례 추가
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4caf50', alpha=0.7, label='정상'),
            Patch(facecolor='#fbc02d', alpha=0.7, label='이상 (중간)'),
            Patch(facecolor='#ff6f00', alpha=0.7, label='이상 (높음)'),
            Patch(facecolor='#d32f2f', alpha=0.7, label='이상 (심각)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # 제목 설정
        if title is None:
            title = f'MAR Deviation 분석 (이상: {len(abnormal_phonemes)}/14)'
        ax.set_title(title, fontsize=14, weight='bold', pad=15)
        
        # 통계 정보 텍스트 박스
        stats_text = f"전체 MAR 평균: {mean_mar:.3f}\n표준편차: {std_mar:.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig

    def plot_detection_summary(
        self,
        result: Dict[str, Any],
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot detection result summary (prediction + confidence).

        Args:
            result: XAI result dictionary
            ax: Matplotlib axes (if None, creates new figure)
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Extract detection info
        detection = result.get('detection', {})
        verdict = detection.get('verdict', 'unknown')
        pred_label = detection.get('prediction_label')
        if pred_label is None:
            pred_label = 'FAKE' if verdict == 'fake' else 'REAL'
        confidence = detection.get('confidence', 0.0) * 100

        # Determine color
        color = self.colors['fake'] if pred_label == 'FAKE' else self.colors['real']

        # Create large text display
        ax.text(
            0.5, 0.6, pred_label,
            ha='center', va='center',
            fontsize=48, weight='bold',
            color=color,
            transform=ax.transAxes
        )

        ax.text(
            0.5, 0.35, f"신뢰도: {confidence:.1f}%",
            ha='center', va='center',
            fontsize=24,
            transform=ax.transAxes
        )

        # Add summary
        summary_dict = result.get('summary', {})
        summary = summary_dict.get('primary_reason') or summary_dict.get('detailed_explanation', '')
        ax.text(
            0.5, 0.15, summary,
            ha='center', va='center',
            fontsize=11,
            wrap=True,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
            transform=ax.transAxes
        )

        ax.axis('off')

        # Set title
        if title is None:
            video_id = result['video_info']['video_id']
            title = f"탐지 결과\n{video_id}"
        ax.set_title(title, fontsize=14, weight='bold', pad=15)

        return fig

    def plot_temporal_probability(
        self,
        result: Dict[str, Any],
        title: str = "시간별 Fake 확률 변화"
    ) -> plt.Figure:
        """
        Plot temporal fake probability for all frames as a line graph.

        Args:
            result: XAI result dictionary
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Extract frame-level probabilities (all 70 frames)
        temporal_analysis = result.get('temporal_analysis', {})
        frame_probs = temporal_analysis.get('frame_probabilities', [])

        if not frame_probs:
            print("[WARNING] No frame_probabilities found in result")
            return None

        # Prepare data - use all frames
        time_points = [item['timestamp_sec'] for item in frame_probs]
        fake_probs = [item['fake_probability'] * 100 for item in frame_probs]  # Convert to percentage
        real_probs = [item['real_probability'] * 100 for item in frame_probs]  # Convert to percentage

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot line graph with markers
        ax.plot(
            time_points,
            fake_probs,
            color=self.colors['fake'],
            linewidth=2.5,
            marker='o',
            markersize=4,
            markerfacecolor=self.colors['fake'],
            markeredgecolor='white',
            markeredgewidth=0.8,
            label='Fake 확률',
            zorder=3
        )

        # Optional: Add Real probability line (semi-transparent)
        ax.plot(
            time_points,
            real_probs,
            color=self.colors['real'],
            linewidth=2.0,
            marker='s',
            markersize=3,
            markerfacecolor=self.colors['real'],
            markeredgecolor='white',
            markeredgewidth=0.8,
            label='Real 확률',
            alpha=0.6,
            zorder=2,
            linestyle='--'
        )

        # Fill area under fake probability curve
        ax.fill_between(
            time_points,
            fake_probs,
            alpha=0.2,
            color=self.colors['fake'],
            zorder=1
        )

        # Styling
        ax.set_xlabel('시간 (초)', fontsize=14, fontweight='bold')
        ax.set_ylabel('확률 (%)', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        # Y-axis range [0, 100]
        ax.set_ylim(0, 100)

        # Add threshold line at 50%
        ax.axhline(
            y=50,
            color='gray',
            linestyle='--',
            linewidth=1.5,
            alpha=0.7,
            label='판정 임계값 (50%)',
            zorder=1
        )

        # Grid
        plt.grid(True)
        ax.grid(axis='both', alpha=0.3, linestyle='--', linewidth=0.8)

        # Legend
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

        # Overall info box
        overall_fake = result['detection']['probabilities']['fake'] * 100
        overall_real = result['detection']['probabilities']['real'] * 100
        prediction = result['detection']['prediction_label']

        info_text = f"전체 예측: {prediction}\nFake 확률: {overall_fake:.1f}%\nReal 확률: {overall_real:.1f}%"
        ax.text(
            0.02, 0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5)
        )

        plt.tight_layout()
        return fig

    def compare_4_cases(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ) -> Dict[str, plt.Figure]:
        """
        Generate comparison visualizations for 4 test cases.

        Args:
            results: Dict mapping case names to XAI results
                    Expected keys: 'random', 'trained', 'real', 'fake'
            save_path: Directory to save figures (if None, doesn't save)

        Returns:
            Dictionary mapping visualization names to figures
        """
        figures = {}

        # 1. Branch Contributions Comparison
        fig1, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig1.suptitle('Branch 기여도 비교 (4 Cases)', fontsize=16, weight='bold')

        for idx, (case_name, result) in enumerate(results.items()):
            ax = axes[idx // 2, idx % 2]
            self.plot_branch_contributions(result, ax=ax, title=f'{case_name.upper()}')

        plt.tight_layout()
        figures['branch_contributions'] = fig1

        # 2. Phoneme Attention Comparison
        fig2, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig2.suptitle('음소별 Attention 비교 (4 Cases)', fontsize=16, weight='bold')

        for idx, (case_name, result) in enumerate(results.items()):
            ax = axes[idx // 2, idx % 2]
            self.plot_phoneme_attention(result, ax=ax, title=f'{case_name.upper()}')

        plt.tight_layout()
        figures['phoneme_attention'] = fig2

        # 3. Geometry Analysis Comparison (Temporal Heatmap 대체)
        # [DEPRECATED] Temporal Heatmap은 단순히 attention score를 5프레임에 복제한 것으로
        # 실제 시간 변화를 보여주지 않아 유용성이 낮음
        # Geometry Analysis는 실제 MAR deviation과 Z-score 기반 이상 탐지를 제공
        fig3, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig3.suptitle('MAR Deviation 분석 비교 (4 Cases)', fontsize=16, weight='bold')

        for idx, (case_name, result) in enumerate(results.items()):
            ax = axes[idx // 2, idx % 2]
            self.plot_geometry_analysis(result, ax=ax, title=f'{case_name.upper()}')

        plt.tight_layout()
        figures['geometry_analysis'] = fig3

        # [DEPRECATED] Temporal Heatmap - 주석 처리
        # # 3. Temporal Heatmap Comparison
        # fig3, axes = plt.subplots(2, 2, figsize=(18, 14))
        # fig3.suptitle('시간별 Attention 비교 (4 Cases)', fontsize=16, weight='bold')
        #
        # for idx, (case_name, result) in enumerate(results.items()):
        #     ax = axes[idx // 2, idx % 2]
        #     self.plot_temporal_heatmap(result, ax=ax, title=f'{case_name.upper()}')
        #
        # plt.tight_layout()
        # figures['temporal_heatmap'] = fig3

        # 4. Detection Summary Comparison
        fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig4.suptitle('탐지 결과 비교 (4 Cases)', fontsize=16, weight='bold')

        for idx, (case_name, result) in enumerate(results.items()):
            ax = axes[idx // 2, idx % 2]
            self.plot_detection_summary(result, ax=ax, title=f'{case_name.upper()}')

        plt.tight_layout()
        figures['detection_summary'] = fig4

        # Save if path provided
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            for name, fig in figures.items():
                output_file = save_dir / f'{name}.png'
                fig.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
                print(f"[OK] Saved: {output_file}")

        return figures

    def save_all_visualizations(
        self,
        results: Dict[str, Dict],
        output_dir: str
    ):
        """
        Generate and save all visualizations.

        Args:
            results: Dict mapping case names to XAI results
            output_dir: Directory to save all outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Generating All Visualizations")
        print(f"{'='*80}\n")

        # Generate comparison figures
        figures = self.compare_4_cases(results, save_path=output_dir)

        print(f"\n{'='*80}")
        print(f"All visualizations saved to: {output_dir}")
        print(f"{'='*80}\n")

        return figures

    def plot_geometry_analysis_30fps(
        self,
        features: Dict[str, Any],
        suspicious_intervals: List[Dict],
        geometry_analysis: Dict[str, Any],
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot MAR analysis using full 30fps data within suspicious intervals.

        Instead of using 14×5=70 frames from PIA model, uses all frames from
        suspicious intervals and groups them by phoneme.

        Args:
            features: 30fps feature dict containing:
                - mar_30fps: (T,) MAR values
                - phoneme_labels_30fps: (T,) phoneme labels
                - timestamps_30fps: (T,) timestamps
            suspicious_intervals: List of suspicious interval dicts with start/end times
            geometry_analysis: Original PIA geometry analysis for baseline info
            ax: Matplotlib axes
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 8))
        else:
            fig = ax.figure

        # Extract 30fps data
        mar_30fps = features.get('mar_30fps', np.array([]))
        phoneme_labels_30fps = features.get('phoneme_labels_30fps', np.array([]))
        timestamps_30fps = features.get('timestamps_30fps', np.array([]))

        if len(mar_30fps) == 0 or len(phoneme_labels_30fps) == 0:
            ax.text(0.5, 0.5, 'No 30fps data available', ha='center', va='center',
                   fontsize=14, transform=ax.transAxes)
            ax.set_title(title or 'MAR Deviation Analysis (No Data)', fontsize=14, weight='bold')
            return fig

        # Get suspicious interval mask (combine all intervals)
        if suspicious_intervals and len(suspicious_intervals) > 0:
            interval_mask = np.zeros(len(timestamps_30fps), dtype=bool)
            for interval in suspicious_intervals:
                start = interval.get('start', 0)
                end = interval.get('end', timestamps_30fps[-1] if len(timestamps_30fps) > 0 else 0)
                interval_mask |= (timestamps_30fps >= start) & (timestamps_30fps <= end)
        else:
            # No intervals: use full video
            interval_mask = np.ones(len(timestamps_30fps), dtype=bool)

        # Filter data to suspicious intervals
        mar_interval = mar_30fps[interval_mask]
        phoneme_interval = phoneme_labels_30fps[interval_mask]

        # Get phoneme vocab (14 Korean phonemes)
        phoneme_vocab = get_phoneme_vocab()
        skip_tokens = {'<pad>', '<PAD>', '<unk>', '<UNK>', '', 'sil', 'sp', 'spn'}

        # [FIX] Load phoneme-specific baseline (instead of in-video baseline)
        baseline_path = Path(__file__).parent.parent.parent / 'mar_baseline_pia_real_fixed.json'
        phoneme_baseline = {}
        try:
            with open(baseline_path, 'r', encoding='utf-8') as f:
                baseline_data = json.load(f)
                phoneme_baseline = baseline_data.get('phoneme_stats', {})
        except Exception as e:
            print(f"[WARNING] Failed to load baseline: {e}")

        # Calculate per-phoneme MAR statistics
        phoneme_mar_stats = {}
        for phoneme in phoneme_vocab:
            mask = phoneme_interval == phoneme
            if np.sum(mask) > 0:
                mar_values = mar_interval[mask]
                phoneme_mar_stats[phoneme] = {
                    'mean': float(np.mean(mar_values)),
                    'std': float(np.std(mar_values)),
                    'count': int(np.sum(mask)),
                    'min': float(np.min(mar_values)),
                    'max': float(np.max(mar_values))
                }

        # Prepare visualization data
        phoneme_names = []
        measured_mar = []
        mar_stds = []
        frame_counts = []
        colors = []
        expected_means = []
        expected_p10s = []
        expected_p90s = []
        z_scores_list = []

        for phoneme in phoneme_vocab:
            phoneme_kr = phoneme_to_korean(phoneme)
            phoneme_names.append(phoneme_kr)

            # Get phoneme-specific baseline
            baseline_stats = phoneme_baseline.get(phoneme, {})
            baseline_mean = baseline_stats.get('mean', 0.3)
            baseline_std = baseline_stats.get('std', 0.09)
            baseline_p10 = baseline_stats.get('p10', baseline_mean - baseline_std)
            baseline_p90 = baseline_stats.get('p90', baseline_mean + baseline_std)

            expected_means.append(baseline_mean)
            expected_p10s.append(baseline_p10)
            expected_p90s.append(baseline_p90)

            if phoneme in phoneme_mar_stats:
                stats = phoneme_mar_stats[phoneme]
                measured_mar.append(stats['mean'])
                mar_stds.append(stats['std'])
                frame_counts.append(stats['count'])

                # [FIX] Calculate z-score using phoneme-specific baseline
                effective_std = max(baseline_std, 0.05)  # Minimum threshold for stability
                z_score = abs(stats['mean'] - baseline_mean) / effective_std
                z_scores_list.append(z_score)

                if z_score > 3:
                    colors.append('#d32f2f')  # Critical - dark red
                elif z_score > 2:
                    colors.append('#ff6f00')  # High - orange
                elif z_score > 1.5:
                    colors.append('#fbc02d')  # Medium - yellow
                else:
                    colors.append('#4caf50')  # Normal - green
            else:
                # No frames for this phoneme
                measured_mar.append(0)
                mar_stds.append(0)
                frame_counts.append(0)
                z_scores_list.append(0)
                colors.append('#9e9e9e')  # Gray for missing

        # Create bar chart
        x_pos = np.arange(len(phoneme_names))
        width = 0.6

        # [FIX] First draw P10-P90 expected range background for each phoneme
        for i, (p10, p90, exp_mean) in enumerate(zip(expected_p10s, expected_p90s, expected_means)):
            # P10-P90 range (gray area)
            ax.bar(i, p90 - p10, width, bottom=p10,
                   color='lightgray', alpha=0.3, edgecolor='gray', linewidth=0.5, zorder=1)
            # Expected mean (blue dashed line)
            ax.plot([i - width/2, i + width/2], [exp_mean, exp_mean],
                   color='blue', linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)

        # Draw measured MAR bars
        bars = ax.bar(x_pos, measured_mar, width, color=colors, alpha=0.7,
                     edgecolor='black', linewidth=1, zorder=3)

        # Add error bars for std
        ax.errorbar(x_pos, measured_mar, yerr=mar_stds, fmt='none',
                   ecolor='black', capsize=3, capthick=1, alpha=0.6, zorder=4)

        # Add frame count and z-score annotations
        for i, (bar, count, z_score) in enumerate(zip(bars, frame_counts, z_scores_list)):
            if count > 0:
                label_text = f'n={count}'
                if z_score >= 1.5:
                    label_text += f'\nz={z_score:.1f}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       label_text, ha='center', va='bottom', fontsize=7, alpha=0.8)

        # Formatting
        ax.set_xlabel('음소 (Phoneme)', fontsize=12, fontweight='bold')
        ax.set_ylabel('MAR 값 (Mouth Aspect Ratio)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(phoneme_names, rotation=0, ha='center', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Set y-axis limit considering expected ranges
        max_val = max(max(measured_mar) if measured_mar else 0, max(expected_p90s) if expected_p90s else 0)
        ax.set_ylim(0, max_val * 1.2 + 0.1 if max_val > 0 else 1.0)

        # Legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='lightgray', alpha=0.3, edgecolor='gray', label='기대 범위 (P10-P90)'),
            Line2D([0], [0], color='blue', linestyle='--', linewidth=1.5, label='기대 평균'),
            Patch(facecolor='#4caf50', alpha=0.7, label='정상 (|z|<1.5)'),
            Patch(facecolor='#fbc02d', alpha=0.7, label='경미 (1.5≤|z|<2)'),
            Patch(facecolor='#ff6f00', alpha=0.7, label='높음 (2≤|z|<3)'),
            Patch(facecolor='#d32f2f', alpha=0.7, label='심각 (|z|≥3)'),
            Patch(facecolor='#9e9e9e', alpha=0.7, label='데이터 없음'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7)

        # Title with interval info
        if suspicious_intervals and len(suspicious_intervals) > 0:
            interval_str = ', '.join([f"{i['start']:.1f}~{i['end']:.1f}s"
                                      for i in suspicious_intervals[:3]])
            if title is None:
                title = f'MAR 편차 분석 (의심 구간: {interval_str})'
        else:
            if title is None:
                title = 'MAR 편차 분석 (전체 영상)'

        ax.set_title(title, fontsize=14, weight='bold', pad=15)

        # Stats text box - count abnormal phonemes
        total_frames = sum(frame_counts)
        abnormal_count = sum(1 for z in z_scores_list if z >= 1.5)
        stats_text = f"분석 프레임: {total_frames}\n이상 음소: {abnormal_count}/14\n(음소별 baseline 비교)"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig

    def plot_phoneme_attention_30fps(
        self,
        features: Dict[str, Any],
        suspicious_intervals: List[Dict],
        phoneme_analysis: Dict[str, Any],
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        top_n: int = 14
    ) -> plt.Figure:
        """
        Plot phoneme attention based on frame counts in suspicious intervals.

        Instead of using model attention weights, shows phoneme distribution
        within suspicious intervals.

        Args:
            features: 30fps feature dict
            suspicious_intervals: List of suspicious interval dicts
            phoneme_analysis: Original PIA phoneme analysis (for attention weights if available)
            ax: Matplotlib axes
            title: Plot title
            top_n: Number of phonemes to show

        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Extract 30fps data
        phoneme_labels_30fps = features.get('phoneme_labels_30fps', np.array([]))
        timestamps_30fps = features.get('timestamps_30fps', np.array([]))

        if len(phoneme_labels_30fps) == 0:
            ax.text(0.5, 0.5, 'No phoneme data available', ha='center', va='center',
                   fontsize=14, transform=ax.transAxes)
            ax.set_title(title or 'Phoneme Attention (No Data)', fontsize=14, weight='bold')
            return fig

        # Get suspicious interval mask
        if suspicious_intervals and len(suspicious_intervals) > 0:
            interval_mask = np.zeros(len(timestamps_30fps), dtype=bool)
            for interval in suspicious_intervals:
                start = interval.get('start', 0)
                end = interval.get('end', timestamps_30fps[-1] if len(timestamps_30fps) > 0 else 0)
                interval_mask |= (timestamps_30fps >= start) & (timestamps_30fps <= end)
        else:
            interval_mask = np.ones(len(timestamps_30fps), dtype=bool)

        phoneme_interval = phoneme_labels_30fps[interval_mask]

        # Count phoneme occurrences
        phoneme_vocab = get_phoneme_vocab()
        skip_tokens = {'<pad>', '<PAD>', '<unk>', '<UNK>', '', 'sil', 'sp', 'spn'}

        phoneme_counts = {}
        total_valid = 0
        for phoneme in phoneme_vocab:
            count = int(np.sum(phoneme_interval == phoneme))
            if count > 0:
                phoneme_counts[phoneme] = count
                total_valid += count

        if total_valid == 0:
            ax.text(0.5, 0.5, 'No valid phonemes in intervals', ha='center', va='center',
                   fontsize=14, transform=ax.transAxes)
            ax.set_title(title or 'Phoneme Attention (No Phonemes)', fontsize=14, weight='bold')
            return fig

        # Get PIA model attention weights if available
        model_attention = {}
        if phoneme_analysis and 'phoneme_scores' in phoneme_analysis:
            for p in phoneme_analysis['phoneme_scores']:
                phoneme_mfa = p.get('phoneme_mfa', p.get('phoneme', ''))
                model_attention[phoneme_mfa] = p.get('score', 0)

        # Prepare data: combine frame count (exposure) with model attention
        phoneme_data = []
        for phoneme, count in phoneme_counts.items():
            exposure = count / total_valid  # Frame proportion
            attention = model_attention.get(phoneme, exposure)  # Use model attention if available
            combined_score = 0.5 * exposure + 0.5 * attention  # Blend both metrics

            phoneme_data.append({
                'phoneme': phoneme,
                'phoneme_kr': phoneme_to_korean(phoneme),
                'count': count,
                'exposure': exposure,
                'attention': attention,
                'score': combined_score
            })

        # Sort by combined score
        sorted_data = sorted(phoneme_data, key=lambda x: x['score'], reverse=True)[:top_n]

        # Prepare for plotting
        phonemes = [f"{p['phoneme_kr']}\n({p['phoneme']})" for p in sorted_data]
        scores = [p['score'] for p in sorted_data]
        counts = [p['count'] for p in sorted_data]

        # Color by score level
        bar_colors = []
        for p in sorted_data:
            if p['score'] > 0.15:
                bar_colors.append(self.colors['high'])
            elif p['score'] > 0.08:
                bar_colors.append(self.colors['medium'])
            else:
                bar_colors.append(self.colors['low'])

        # Create horizontal bar chart
        y_pos = np.arange(len(phonemes))
        bars = ax.barh(y_pos, scores, color=bar_colors, edgecolor='black', linewidth=0.5)

        # Add count annotations
        for i, (bar, score, count) in enumerate(zip(bars, scores, counts)):
            # Score label
            ax.text(score + 0.005, i, f'{score:.4f}', va='center', fontsize=9, weight='bold')
            # Count label inside bar
            if bar.get_width() > 0.03:
                ax.text(bar.get_width() * 0.5, i, f'n={count}', va='center', ha='center',
                       fontsize=8, color='white', weight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(phonemes)
        ax.set_xlabel('Combined Score (Exposure + Attention)', fontsize=12, weight='bold')
        ax.set_ylabel('Phoneme (음소)', fontsize=12, weight='bold')
        ax.set_xlim(0, max(scores) * 1.2 if scores else 1.0)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['high'], edgecolor='black', label='High (>0.15)'),
            Patch(facecolor=self.colors['medium'], edgecolor='black', label='Medium (0.08-0.15)'),
            Patch(facecolor=self.colors['low'], edgecolor='black', label='Low (<0.08)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        # Title
        if title is None:
            title = f'음소별 Attention 분포 (의심 구간 {len(suspicious_intervals)}개)'
        ax.set_title(title, fontsize=14, weight='bold', pad=15)

        plt.tight_layout()
        return fig
