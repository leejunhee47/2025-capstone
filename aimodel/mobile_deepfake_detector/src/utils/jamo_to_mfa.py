"""
Jamo to MFA phoneme mapping for Korean

MFA (Montreal Forced Alignment) uses romanized Korean phonemes.
This module converts Jamo characters to MFA format.
"""

# Jamo to MFA mapping based on Korean phonetics
JAMO_TO_MFA = {
    # Initial consonants (초성)
    'ㄱ': 'G',    # ㄱ
    'ㄲ': 'GG',   # ㄲ
    'ㄴ': 'N',    # ㄴ
    'ㄷ': 'D',    # ㄷ
    'ㄸ': 'DD',   # ㄸ
    'ㄹ': 'R',    # ㄹ (initial) / L (final)
    'ㅁ': 'M',    # ㅁ
    'ㅂ': 'B',    # ㅂ
    'ㅃ': 'BB',   # ㅃ
    'ㅅ': 'S',    # ㅅ
    'ㅆ': 'SS',   # ㅆ
    'ㅇ': '',     # ㅇ (silent when initial)
    'ㅈ': 'J',    # ㅈ
    'ㅉ': 'JJ',   # ㅉ
    'ㅊ': 'CHh',  # ㅊ
    'ㅋ': 'Kh',   # ㅋ
    'ㅌ': 'Th',   # ㅌ
    'ㅍ': 'Ph',   # ㅍ
    'ㅎ': 'H',    # ㅎ

    # Vowels (중성)
    'ㅏ': 'A',    # ㅏ
    'ㅐ': 'E',    # ㅐ
    'ㅑ': 'iA',   # ㅑ
    'ㅒ': 'iE',   # ㅒ
    'ㅓ': 'EO',   # ㅓ
    'ㅔ': 'E',    # ㅔ
    'ㅕ': 'iEO',  # ㅕ
    'ㅖ': 'iE',   # ㅖ
    'ㅗ': 'O',    # ㅗ
    'ㅘ': 'oA',   # ㅘ
    'ㅙ': 'oE',   # ㅙ
    'ㅚ': 'oE',   # ㅚ
    'ㅛ': 'iO',   # ㅛ
    'ㅜ': 'U',    # ㅜ
    'ㅝ': 'uEO',  # ㅝ
    'ㅞ': 'uEO',  # ㅞ (MFA vocabulary does not include uE)
    'ㅟ': 'uI',   # ㅟ
    'ㅠ': 'iU',   # ㅠ
    'ㅡ': 'EU',   # ㅡ
    'ㅢ': 'euI',  # ㅢ
    'ㅣ': 'I',    # ㅣ

    # Final consonants (종성)
    # In final position, some consonants change
    # Using simplified mapping - may need refinement
}

# Additional mapping for final consonants
FINAL_CONSONANT_MAP = {
    'ㄱ': 'k',    # ㄱ final
    'ㄲ': 'k',    # ㄲ final
    'ㄳ': 'k',    # ㄳ
    'ㄴ': 'N',    # ㄴ
    'ㄵ': 'N',    # ㄵ
    'ㄶ': 'N',    # ㄶ
    'ㄷ': 't',    # ㄷ final
    'ㄹ': 'L',    # ㄹ final
    'ㄺ': 'k',    # ㄺ
    'ㄻ': 'M',    # ㄻ
    'ㄼ': 'L',    # ㄼ
    'ㄽ': 'L',    # ㄽ
    'ㄾ': 'L',    # ㄾ
    'ㄿ': 'p',    # ㄿ
    'ㅀ': 'L',    # ㅀ
    'ㅁ': 'M',    # ㅁ
    'ㅂ': 'p',    # ㅂ final
    'ㅄ': 'p',    # ㅄ
    'ㅅ': 't',    # ㅅ final
    'ㅆ': 't',    # ㅆ final
    'ㅇ': 'NG',   # ㅇ final
    'ㅈ': 't',    # ㅈ final
    'ㅊ': 't',    # ㅊ final
    'ㅋ': 'k',    # ㅋ final
    'ㅌ': 't',    # ㅌ final
    'ㅍ': 'p',    # ㅍ final
    'ㅎ': 't',    # ㅎ final
}


def jamo_to_mfa(jamo_list):
    """
    Convert list of Jamo characters to MFA phonemes

    Args:
        jamo_list: List of Jamo characters

    Returns:
        List of MFA phoneme symbols
    """
    mfa_phonemes = []

    for i, jamo in enumerate(jamo_list):
        # Check if it's a vowel or consonant
        if jamo in 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ':
            # Vowel
            if jamo in JAMO_TO_MFA:
                mfa = JAMO_TO_MFA[jamo]
                if mfa:  # Skip empty mappings
                    mfa_phonemes.append(mfa)
        else:
            # Consonant - check position
            # Simple heuristic: if next is vowel, it's initial; if prev is vowel, it's final
            is_final = False
            if i > 0 and jamo_list[i-1] in 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ':
                # Previous was vowel, this might be final consonant
                if i == len(jamo_list) - 1:
                    is_final = True
                elif i < len(jamo_list) - 1 and jamo_list[i+1] not in 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ':
                    is_final = True

            if is_final and jamo in FINAL_CONSONANT_MAP:
                mfa = FINAL_CONSONANT_MAP[jamo]
                if mfa:
                    mfa_phonemes.append(mfa)
            elif jamo in JAMO_TO_MFA:
                mfa = JAMO_TO_MFA[jamo]
                if mfa:  # Skip empty mappings (like initial ㅇ)
                    mfa_phonemes.append(mfa)

    return mfa_phonemes


def test_conversion():
    """Test Jamo to MFA conversion"""
    # Test cases
    test_cases = [
        (['ㅎ', 'ㅘ'], ['H', 'oA']),  # 화
        (['ㄷ', 'ㅏ', 'ㅂ'], ['D', 'A', 'p']),  # 답
        (['ㅎ', 'ㅏ'], ['H', 'A']),  # 하
        (['ㅇ', 'ㅕ'], ['iEO']),  # 여 (ㅇ is silent when initial)
        (['ㅇ', 'ㅡ', 'ㅇ'], ['EU', 'NG']),  # 응
        (['ㅎ', 'ㅏ', 'ㄴ'], ['H', 'A', 'N']),  # 한
    ]

    print("Testing Jamo to MFA conversion:")
    for jamo, expected in test_cases:
        result = jamo_to_mfa(jamo)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {''.join(jamo):5} -> {result} (expected: {expected})")


# MFA to Jamo reverse mapping (for Korean display in XAI visualizations)
MFA_TO_JAMO = {
    # Vowels (모음) - primary mappings
    'A': 'ㅏ',
    'E': 'ㅔ',    # ㅐ, ㅔ both map to E
    'iA': 'ㅑ',
    'iE': 'ㅖ',   # ㅒ, ㅖ both map to iE
    'EO': 'ㅓ',
    'iEO': 'ㅕ',
    'O': 'ㅗ',
    'oA': 'ㅘ',
    'oE': 'ㅙ',   # ㅙ, ㅚ both map to oE
    'iO': 'ㅛ',
    'U': 'ㅜ',
    'uEO': 'ㅝ',  # ㅝ, ㅞ both map to uEO
    'uI': 'ㅟ',
    'iU': 'ㅠ',
    'EU': 'ㅡ',
    'euI': 'ㅢ',
    'I': 'ㅣ',

    # Consonants (자음) - initial
    'G': 'ㄱ',
    'GG': 'ㄲ',
    'N': 'ㄴ',
    'D': 'ㄷ',
    'DD': 'ㄸ',
    'R': 'ㄹ',
    'L': 'ㄹ',
    'M': 'ㅁ',
    'B': 'ㅂ',
    'BB': 'ㅃ',
    'S': 'ㅅ',
    'SS': 'ㅆ',
    'J': 'ㅈ',
    'JJ': 'ㅉ',
    'CHh': 'ㅊ',
    'CH': 'ㅊ',
    'Kh': 'ㅋ',
    'K': 'ㅋ',
    'Th': 'ㅌ',
    'T': 'ㅌ',
    'Ph': 'ㅍ',
    'P': 'ㅍ',
    'H': 'ㅎ',
    'NG': 'ㅇ',

    # Final consonants (종성)
    'k': 'ㄱ',
    't': 'ㄷ',
    'p': 'ㅂ',
}


def mfa_to_korean(mfa_phonemes: list) -> str:
    """
    Convert MFA phoneme list to Korean Jamo string.

    Args:
        mfa_phonemes: List of MFA phoneme strings (e.g., ['A', 'M', 'U', 'T'])

    Returns:
        Korean Jamo string (e.g., 'ㅏㅁㅜㅌ')
    """
    result = []
    skip_tokens = {'<pad>', '<PAD>', '<unk>', '<UNK>', '', 'sil', 'sp', 'spn'}

    for phoneme in mfa_phonemes:
        if phoneme in skip_tokens:
            continue

        # Direct lookup
        jamo = MFA_TO_JAMO.get(phoneme)
        if jamo:
            result.append(jamo)
            continue

        # Case-insensitive lookup
        for key, value in MFA_TO_JAMO.items():
            if key.lower() == phoneme.lower():
                result.append(value)
                break

    return ''.join(result)


def get_interval_transcription(phoneme_intervals: list, start_time: float, end_time: float) -> str:
    """
    Get Korean transcription for a specific time interval.

    Args:
        phoneme_intervals: List of dicts with 'start', 'end', 'phoneme' keys
        start_time: Start of the interval in seconds
        end_time: End of the interval in seconds

    Returns:
        Korean Jamo string for phonemes in the interval
    """
    relevant_phonemes = []

    for interval in phoneme_intervals:
        ph_start = interval.get('start', 0)
        ph_end = interval.get('end', 0)
        phoneme = interval.get('phoneme', '')

        # Include if there's any overlap with the target interval
        if ph_end > start_time and ph_start < end_time:
            relevant_phonemes.append(phoneme)

    return mfa_to_korean(relevant_phonemes)


def test_mfa_to_korean():
    """Test MFA to Korean conversion"""
    test_cases = [
        (['A', 'M', 'U', 'T'], 'ㅏㅁㅜㅌ'),
        (['H', 'A', 'N', 'G', 'EU', 'k'], 'ㅎㅏㄴㄱㅡㄱ'),  # 한국
        (['A', '<pad>', 'E', '<pad>'], 'ㅏㅔ'),
        (['G', 'A', 'N', 'J', 'A', 'NG'], 'ㄱㅏㄴㅈㅏㅇ'),  # 간장
    ]

    print("\nTesting MFA to Korean conversion:")
    for mfa, expected in test_cases:
        result = mfa_to_korean(mfa)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {mfa} -> '{result}' (expected: '{expected}')")


if __name__ == "__main__":
    test_conversion()
    test_mfa_to_korean()