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
            'E': 'ㅔ', 'EU': 'ㅡ', 'I': 'ㅣ', 'M': 'ㅁ',
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

        # Sort by score (descending) and take top N
        sorted_data = sorted(phoneme_data, key=lambda x: x['score'], reverse=True)[:top_n]

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

        # Get phoneme labels
        phoneme_labels = [
            f"{p['phoneme']} ({p['phoneme_mfa']})"
            for p in result['phoneme_analysis']['phoneme_scores']
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
        Plot geometry (MAR) analysis with abnormality detection.

        Args:
            result: XAI result dictionary
            ax: Matplotlib axes (if None, creates new figure)
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Extract geometry data
        geom_analysis = result['geometry_analysis']
        mean_mar = geom_analysis['mean_mar']
        std_mar = geom_analysis['std_mar']
        abnormal_phonemes = geom_analysis.get('abnormal_phonemes', [])

        # Create summary text box
        summary_text = f"""
        MAR 통계:
        - Mean: {mean_mar:.4f}
        - Std: {std_mar:.4f}
        - Abnormal: {len(abnormal_phonemes)}/14
        """

        ax.text(
            0.5, 0.7, summary_text.strip(),
            ha='center', va='center',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            transform=ax.transAxes
        )

        # List abnormal phonemes
        if abnormal_phonemes:
            abnormal_text = "이상 음소:\n" + "\n".join([
                f"- {p['phoneme']} ({p['phoneme_mfa']}): {p['measured_mar']:.4f} "
                f"(예상: {p['expected_range'][0]:.2f}-{p['expected_range'][1]:.2f})"
                for p in abnormal_phonemes[:5]  # Show top 5
            ])

            ax.text(
                0.5, 0.3, abnormal_text,
                ha='center', va='top',
                fontsize=10,
                color='red',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
                transform=ax.transAxes
            )

        ax.axis('off')

        # Set title
        if title is None:
            title = 'Geometry (MAR) 분석'
        ax.set_title(title, fontsize=14, weight='bold', pad=15)

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
        detection = result['detection']
        pred_label = detection['prediction_label']
        confidence = detection['confidence'] * 100

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
        summary = result['summary']['overall']
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

        # 3. Temporal Heatmap Comparison
        fig3, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig3.suptitle('시간별 Attention 비교 (4 Cases)', fontsize=16, weight='bold')

        for idx, (case_name, result) in enumerate(results.items()):
            ax = axes[idx // 2, idx % 2]
            self.plot_temporal_heatmap(result, ax=ax, title=f'{case_name.upper()}')

        plt.tight_layout()
        figures['temporal_heatmap'] = fig3

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
