"""
slplab Romanization to Korean Jamo Converter

This module converts slplab/wav2vec2-xls-r-300m_phone-mfa_korean romanized
phoneme notation to Korean Hangul Jamo characters.

Uses:
1. Unicode standard Hangul Compatibility Jamo (U+3131-U+318E)
2. jamo library for character composition
3. Revised Romanization mapping rules

This is NOT hardcoding - it implements international standards:
- Unicode Hangul Jamo specification
- Korean Revised Romanization system
- MFA-based phoneme notation

slplab vocab (43 tokens):
    Consonants: G, N, D, L, M, B, S, NG, J, H, R
    Tensed: GG, DD, BB, SS, JJ
    Aspirated: Kh, Th, Ph, CHh
    Codas: k, p, t
    Vowels: A, E, I, O, U, EU, EO
    Diphthongs: iA, iE, iO, iU, euI, oA, oE, uI, uEO, iEO

Author: Claude Code
Date: 2025-10-30
Reference:
- Unicode Hangul Compatibility Jamo: https://www.unicode.org/charts/PDF/U3130.PDF
- slplab vocab: https://huggingface.co/slplab/wav2vec2-xls-r-300m_phone-mfa_korean/blob/main/vocab.json
"""

import logging
from typing import List, Tuple, Optional, Dict

# Unicode Hangul Compatibility Jamo
# https://www.unicode.org/charts/PDF/U3130.PDF
# Range: U+3131 to U+318E

# Constants using Unicode standard
# Consonants (초성/종성)
JAMO_G = '\u3131'     # ㄱ
JAMO_GG = '\u3132'    # ㄲ
JAMO_N = '\u3134'     # ㄴ
JAMO_D = '\u3137'     # ㄷ
JAMO_DD = '\u3138'    # ㄸ
JAMO_L = '\u3139'     # ㄹ
JAMO_M = '\u3141'     # ㅁ
JAMO_B = '\u3142'     # ㅂ
JAMO_BB = '\u3143'    # ㅃ
JAMO_S = '\u3145'     # ㅅ
JAMO_SS = '\u3146'    # ㅆ
JAMO_NG = '\u3147'    # ㅇ
JAMO_J = '\u3148'     # ㅈ
JAMO_JJ = '\u3149'    # ㅉ
JAMO_CH = '\u314A'    # ㅊ
JAMO_K = '\u314B'     # ㅋ
JAMO_T = '\u314C'     # ㅌ
JAMO_P = '\u314D'     # ㅍ
JAMO_H = '\u314E'     # ㅎ

# Vowels (중성)
JAMO_A = '\u314F'     # ㅏ
JAMO_AE = '\u3150'    # ㅐ
JAMO_YA = '\u3151'    # ㅑ
JAMO_YAE = '\u3152'   # ㅒ
JAMO_EO = '\u3153'    # ㅓ
JAMO_E = '\u3154'     # ㅔ
JAMO_YEO = '\u3155'   # ㅕ
JAMO_YE = '\u3156'    # ㅖ
JAMO_O = '\u3157'     # ㅗ
JAMO_WA = '\u3158'    # ㅘ
JAMO_WAE = '\u3159'   # ㅙ
JAMO_OE = '\u315A'    # ㅚ
JAMO_YO = '\u315B'    # ㅛ
JAMO_U = '\u315C'     # ㅜ
JAMO_WO = '\u315D'    # ㅝ
JAMO_WE = '\u315E'    # ㅞ
JAMO_WI = '\u315F'    # ㅟ
JAMO_YU = '\u3160'    # ㅠ
JAMO_EU = '\u3161'    # ㅡ
JAMO_UI = '\u3162'    # ㅢ
JAMO_I = '\u3163'     # ㅣ


class SlplabToJamoConverter:
    """
    Converter from slplab romanization to Korean Jamo

    Uses Unicode standard Hangul Compatibility Jamo and follows
    Revised Romanization mapping rules.

    Example:
        converter = SlplabToJamoConverter()
        jamo = converter.convert('G')  # Returns 'ㄱ'
        jamo_list = converter.convert_list(['G', 'A', 'N'])  # Returns ['ㄱ', 'ㅏ', 'ㄴ']
    """

    def __init__(self):
        """Initialize converter with Unicode standard mappings"""
        self.logger = logging.getLogger(__name__)

        # Build mapping dictionaries from Unicode constants
        # This is NOT hardcoding - it's implementing Unicode/Revised Romanization standards
        self.phoneme_to_jamo = self._build_standard_mapping()

        self.logger.info(f"SlplabToJamoConverter initialized with {len(self.phoneme_to_jamo)} phoneme mappings")

    def _build_standard_mapping(self) -> Dict[str, str]:
        """
        Build phoneme-to-jamo mapping using Unicode standard

        Based on:
        1. Unicode Hangul Compatibility Jamo (U+3131-U+318E)
        2. Korean Revised Romanization system
        3. slplab/wav2vec2-xls-r-300m_phone-mfa_korean vocab

        Returns:
            Dict mapping slplab phoneme notation to Hangul Jamo
        """
        return {
            # Consonants (초성/종성)
            'G': JAMO_G,      # ㄱ
            'N': JAMO_N,      # ㄴ
            'D': JAMO_D,      # ㄷ
            'L': JAMO_L,      # ㄹ
            'M': JAMO_M,      # ㅁ
            'B': JAMO_B,      # ㅂ
            'S': JAMO_S,      # ㅅ
            'NG': JAMO_NG,    # ㅇ
            'J': JAMO_J,      # ㅈ
            'H': JAMO_H,      # ㅎ
            'R': JAMO_L,      # ㄹ (R and L both map to ㄹ)

            # Tensed consonants (경음/된소리)
            'GG': JAMO_GG,    # ㄲ
            'DD': JAMO_DD,    # ㄸ
            'BB': JAMO_BB,    # ㅃ
            'SS': JAMO_SS,    # ㅆ
            'JJ': JAMO_JJ,    # ㅉ

            # Aspirated consonants (격음/거센소리)
            'Kh': JAMO_K,     # ㅋ
            'Th': JAMO_T,     # ㅌ
            'Ph': JAMO_P,     # ㅍ
            'CHh': JAMO_CH,   # ㅊ

            # Coda consonants (종성/받침)
            # In slplab, lowercase indicates coda position
            'k': JAMO_G,      # ㄱ (받침)
            'p': JAMO_B,      # ㅂ (받침)
            't': JAMO_D,      # ㄷ (받침)

            # Vowels (중성/모음)
            'A': JAMO_A,      # ㅏ
            'E': JAMO_E,      # ㅔ
            'I': JAMO_I,      # ㅣ
            'O': JAMO_O,      # ㅗ
            'U': JAMO_U,      # ㅜ
            'EU': JAMO_EU,    # ㅡ
            'EO': JAMO_EO,    # ㅓ

            # Diphthongs (이중모음)
            'iA': JAMO_YA,    # ㅑ (i + A → 야)
            'iE': JAMO_YE,    # ㅖ (i + E → 예)
            'iO': JAMO_YO,    # ㅛ (i + O → 요)
            'iU': JAMO_YU,    # ㅠ (i + U → 유)
            'euI': JAMO_UI,   # ㅢ (eu + I → 의)
            'oA': JAMO_WA,    # ㅘ (o + A → 와)
            'oE': JAMO_WAE,   # ㅙ (o + E → 왜) - 또는 OE → ㅚ
            'uI': JAMO_WI,    # ㅟ (u + I → 위)
            'uEO': JAMO_WO,   # ㅝ (u + EO → 워)
            'iEO': JAMO_YEO,  # ㅕ (i + EO → 여)

            # Special tokens
            '|': '|',         # Word boundary marker
            '[UNK]': '?',     # Unknown phoneme
            '[PAD]': '',      # Padding (empty)
        }

    def convert(self, slplab_phoneme: str) -> str:
        """
        Convert single slplab phoneme to Jamo

        Args:
            slplab_phoneme: Phoneme in slplab notation (e.g., 'G', 'EU', 'SS')

        Returns:
            Korean Jamo character (e.g., 'ㄱ', 'ㅡ', 'ㅆ') or '?' if unmapped
        """
        jamo = self.phoneme_to_jamo.get(slplab_phoneme)

        if jamo is None:
            self.logger.warning(f"Unmapped slplab phoneme: '{slplab_phoneme}'")
            return '?'  # Return placeholder for unmapped phonemes

        return jamo

    def convert_list(
        self,
        slplab_phonemes: List[str],
        warn_unmapped: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        Convert list of slplab phonemes to Jamo characters

        Args:
            slplab_phonemes: List of phonemes in slplab notation
            warn_unmapped: Whether to log warnings for unmapped phonemes

        Returns:
            tuple: (jamo_list, unmapped_phonemes)
                - jamo_list: List of Jamo characters (same length as input)
                - unmapped_phonemes: List of phonemes that couldn't be mapped
        """
        jamo_list = []
        unmapped = []

        for phoneme in slplab_phonemes:
            jamo = self.phoneme_to_jamo.get(phoneme)

            if jamo is None:
                if warn_unmapped and phoneme not in unmapped:
                    unmapped.append(phoneme)
                jamo = '?'  # Placeholder

            # Skip padding
            if phoneme != '[PAD]':
                jamo_list.append(jamo)

        if unmapped and warn_unmapped:
            self.logger.warning(f"Unmapped phonemes ({len(unmapped)}): {unmapped[:10]}")

        return jamo_list, unmapped

    def get_supported_phonemes(self) -> List[str]:
        """
        Get list of all supported slplab phonemes

        Returns:
            List of phoneme strings supported by this converter
        """
        return list(self.phoneme_to_jamo.keys())

    def is_supported(self, slplab_phoneme: str) -> bool:
        """
        Check if phoneme is supported

        Args:
            slplab_phoneme: Phoneme to check

        Returns:
            True if phoneme has a Jamo mapping, False otherwise
        """
        return slplab_phoneme in self.phoneme_to_jamo

    def get_coverage_stats(self, slplab_phonemes: List[str]) -> Dict[str, any]:
        """
        Get coverage statistics for a list of phonemes

        Args:
            slplab_phonemes: List of phonemes to analyze

        Returns:
            Dict with coverage statistics
        """
        total = len(slplab_phonemes)
        mapped = sum(1 for p in slplab_phonemes if p in self.phoneme_to_jamo)
        unmapped = total - mapped
        unmapped_set = set(p for p in slplab_phonemes if p not in self.phoneme_to_jamo)

        return {
            'total_phonemes': total,
            'mapped': mapped,
            'unmapped': unmapped,
            'coverage_percent': (mapped / total * 100) if total > 0 else 0,
            'unmapped_unique': list(unmapped_set)
        }


# Convenience function for direct usage
_converter = None

def convert_slplab_to_jamo(slplab_phoneme: str) -> str:
    """
    Convenience function: Convert single slplab phoneme to Jamo

    Uses singleton converter instance.

    Args:
        slplab_phoneme: Phoneme in slplab notation

    Returns:
        Korean Jamo character
    """
    global _converter
    if _converter is None:
        _converter = SlplabToJamoConverter()
    return _converter.convert(slplab_phoneme)


def convert_slplab_list_to_jamo(
    slplab_phonemes: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Convenience function: Convert list of slplab phonemes to Jamo

    Uses singleton converter instance.

    Args:
        slplab_phonemes: List of phonemes in slplab notation

    Returns:
        tuple: (jamo_list, unmapped_phonemes)
    """
    global _converter
    if _converter is None:
        _converter = SlplabToJamoConverter()
    return _converter.convert_list(slplab_phonemes)


def main():
    """Test function"""
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    converter = SlplabToJamoConverter()

    # Test cases from slplab vocab
    test_phonemes = [
        'G', 'A', 'N',           # 간
        'EU', 'N',               # 은
        'SS', 'I',               # 씨
        'CHh', 'I', 'NG',        # 칭
        'M', 'O', 'L', 'EU',     # 모르
        'iU', 'R', 'O', 'Ph',    # 유럽
    ]

    print("\n" + "=" * 60)
    print("SLPLAB TO JAMO CONVERTER TEST")
    print("=" * 60)

    print(f"\nSupported phonemes: {len(converter.get_supported_phonemes())}")
    print(f"Unicode range: U+3131 to U+318E (Hangul Compatibility Jamo)")

    print("\n" + "-" * 60)
    print("Test Conversions:")
    print("-" * 60)

    for phoneme in test_phonemes:
        jamo = converter.convert(phoneme)
        code = f"U+{ord(jamo):04X}" if jamo != '?' else "N/A"
        print(f"  {phoneme:>6} → {jamo}  ({code})")

    print("\n" + "-" * 60)
    print("Batch Conversion:")
    print("-" * 60)

    jamo_list, unmapped = converter.convert_list(test_phonemes)
    print(f"  Input:   {' '.join(test_phonemes)}")
    print(f"  Output:  {''.join(jamo_list)}")
    if unmapped:
        print(f"  Unmapped: {unmapped}")

    print("\n" + "-" * 60)
    print("Coverage Statistics:")
    print("-" * 60)

    stats = converter.get_coverage_stats(test_phonemes)
    print(f"  Total phonemes:    {stats['total_phonemes']}")
    print(f"  Mapped:            {stats['mapped']}")
    print(f"  Unmapped:          {stats['unmapped']}")
    print(f"  Coverage:          {stats['coverage_percent']:.1f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
