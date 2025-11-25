"""
Korean Phoneme Configuration for PIA-style Deepfake Detection

This module defines phoneme filtering sets for Korean language deepfake detection,
adapted from the PIA (Phoneme-Temporal and Identity-Dynamic Analysis) paper.

PIA's approach: Select phonemes with high visual salience and distinct articulatory
features to maximize lip-sync mismatch detection in deepfake videos.

Korean MFA Phoneme Format:
- Bilabial: M (ㅁ), B (ㅂ), BB (ㅃ), Ph (ㅍ)
- Vowels: A (ㅏ), E (ㅐ/ㅔ), I (ㅣ), O (ㅗ), U (ㅜ), EU (ㅡ), EO (ㅓ)
- Compounds: iA (ㅑ), iO (ㅛ), iU (ㅠ), etc.
"""

# ============================================================================
# Phoneme Selection Strategy
# ============================================================================

# KEEP_PHONEMES: High-priority phonemes with distinct visual characteristics
# Selected based on:
#   1. Visual salience (easily visible lip/mouth movements)
#   2. Articulatory distinctness (clear, measurable geometric changes)
#   3. Coverage across phonetic categories (bilabial, open vowels, rounded vowels)
#
# Total: 14 phonemes (matching PIA's English phoneme count for fair comparison)
# IMPORTANT: Use sorted() order for compatibility with preprocessed data and trained model
KEEP_PHONEMES_KOREAN = {
    # ---- Priority 1: Bilabial Consonants (4 phonemes) ----
    # Highest reliability - require complete lip closure
    "M",   # ㅁ - Bilabial nasal (lips completely closed, nasal airflow)
    "B",   # ㅂ - Bilabial stop (lips closed, airflow blocked)
    "BB",  # ㅃ - Tense bilabial (tight closure, glottis tensed)
    "Ph",  # ㅍ - Aspirated bilabial (forceful release after closure)
    # MAR Range: 0.10-0.30 (very low - lips closed)

    # ---- Priority 2: Open Vowels (3 phonemes) ----
    # Wide mouth opening - high visual salience
    "A",   # ㅏ - Low front vowel (wide mouth opening, low tongue position)
    "E",   # ㅐ/ㅔ - Mid front vowel (moderate opening, front tongue)
    "iA",  # ㅑ - Compound vowel with glide (dynamic mouth movement)
    # MAR Range: 0.60-0.90 (very high - mouth open)

    # ---- Priority 3: Rounded Vowels (4 phonemes) ----
    # Visible lip protrusion/rounding
    "O",   # ㅗ - Mid back rounded (visible lip rounding)
    "U",   # ㅜ - High back rounded (tight lip rounding, protrusion)
    "iO",  # ㅛ - Compound with glide (dynamic rounding)
    "iU",  # ㅠ - Compound with glide (strong protrusion)
    # MAR Range: 0.20-0.50 (moderate - lips rounded)

    # ---- Priority 4: Close Vowels (2 phonemes) ----
    # Moderate visibility
    "I",   # ㅣ - High front unrounded (lips stretched horizontally)
    "EU",  # ㅡ - High central unrounded (neutral lip position)
    # MAR Range: 0.30-0.60 (moderate)

    # ---- Bonus: Palatal Consonant (1 phoneme) ----
    "CHh"  # ㅊ - Palatal affricate with noticeable lip rounding (similar to English /ʃ/)
}

# IGNORED_PHONEMES: Low-confidence phonemes to exclude from analysis
# Includes silence markers, padding tokens, and very short transitions
IGNORED_PHONEMES_KOREAN = {
    # Silence and pause markers
    "<sil>", "<SIL>",  # Silence tokens from ASR
    "sp", "<sp>",      # Short pause markers
    "spn",             # Spoken noise

    # Empty/whitespace
    "", " ",

    # Special tokens
    "<PAD>",   # Padding token
    "<UNK>",   # Unknown phoneme

    # Optional: very short transitions (unreliable timestamps)
    "<short>",
    "<pause>"
}

# CLOSURE_PHONEMES: Bilabial phonemes requiring complete lip closure
# Used for detecting lip-sync mismatches (closure_score < CLOSURE_THRESHOLD during bilabial = mismatch)
CLOSURE_PHONEMES_KOREAN = {
    "M",   # ㅁ - Complete closure (nasal airflow through nose)
    "B",   # ㅂ - Complete closure (stop consonant, airflow blocked)
    "BB",  # ㅃ - Tight closure (tense version)
    "Ph"   # ㅍ - Complete closure (aspirated release)
}

# Closure detection threshold
# If closure_score < 0.3 during a closure phoneme, flag as potential mismatch
CLOSURE_THRESHOLD = 0.3

# ============================================================================
# PIA Model Hyperparameters
# ============================================================================

# Number of frames to collect per phoneme class (PIA uses 5)
# Each phoneme gets up to 5 representative frames across the video
IMAGES_PER_PHON = 5

# Image size for lip crop ResNet input (PIA uses 112x112)
IMAGE_SIZE = (112, 112)

# ============================================================================
# Phoneme Display Mapping (MFA → Korean)
# ============================================================================

# Map MFA phoneme codes to Korean characters for user-friendly display
PHONEME_TO_KOREAN = {
    # Existing KEEP_PHONEMES (14)
    'A': 'ㅏ', 'E': 'ㅔ', 'I': 'ㅣ', 'O': 'ㅗ', 'U': 'ㅜ',
    'EU': 'ㅡ', 'EO': 'ㅓ',
    'M': 'ㅁ', 'B': 'ㅂ', 'BB': 'ㅃ', 'Ph': 'ㅍ',
    'CHh': 'ㅊ',
    'iA': 'ㅑ', 'iO': 'ㅛ', 'iU': 'ㅠ',

    # Additional phonemes (for dynamic selection)
    'G': 'ㄱ', 'GG': 'ㄲ', 'K': 'ㅋ', 'Kh': 'ㅋ',
    'S': 'ㅅ', 'SS': 'ㅆ',
    'N': 'ㄴ', 'NG': 'ㅇ',
    'L': 'ㄹ', 'R': 'ㄹ',
    'D': 'ㄷ', 'DD': 'ㄸ', 'T': 'ㅌ', 'Th': 'ㅌ',
    'J': 'ㅈ', 'JJ': 'ㅉ',
    'H': 'ㅎ',
    'oA': 'ㅘ', 'oE': 'ㅙ', 'uEO': 'ㅝ', 'uI': 'ㅟ', 'euI': 'ㅢ'
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_phoneme_vocab() -> list:
    """
    Get list of kept phonemes for consistent indexing.

    Returns:
        List of 14 Korean phonemes in sorted alphabetical order (A, B, BB, CHh, E, EU, I, M, O, Ph, U, iA, iO, iU)
        This order MUST match the order used during preprocessing and training.
    """
    return sorted(list(KEEP_PHONEMES_KOREAN))


def is_kept_phoneme(phoneme: str) -> bool:
    """
    Check if phoneme should be kept for analysis.

    Args:
        phoneme: Korean MFA phoneme (e.g., "M", "A", "iO")

    Returns:
        True if phoneme is in KEEP_PHONEMES_KOREAN
    """
    return phoneme in KEEP_PHONEMES_KOREAN


def is_ignored_phoneme(phoneme: str) -> bool:
    """
    Check if phoneme should be ignored.

    Args:
        phoneme: Phoneme or special token

    Returns:
        True if phoneme is in IGNORED_PHONEMES_KOREAN
    """
    return phoneme in IGNORED_PHONEMES_KOREAN


def is_closure_phoneme(phoneme: str) -> bool:
    """
    Check if phoneme requires lip closure (bilabial).

    Args:
        phoneme: Korean MFA phoneme

    Returns:
        True if phoneme requires complete lip closure
    """
    return phoneme in CLOSURE_PHONEMES_KOREAN


def detect_mismatch(phoneme: str, closure_score: float) -> bool:
    """
    Detect potential lip-sync mismatch.

    PIA's heuristic: If a bilabial phoneme is spoken (requiring lip closure)
    but the lip closure score is low, this indicates a potential deepfake.

    Args:
        phoneme: Korean MFA phoneme
        closure_score: Lip closure metric (0=open, 1=closed)

    Returns:
        True if mismatch detected (bilabial phoneme with open lips)
    """
    return is_closure_phoneme(phoneme) and closure_score < CLOSURE_THRESHOLD


# ============================================================================
# Phoneme Category Mappings
# ============================================================================

# Map phonemes to articulatory categories for analysis
PHONEME_CATEGORIES = {
    "bilabial": {"M", "B", "BB", "Ph"},
    "open_vowel": {"A", "E", "iA"},
    "rounded_vowel": {"O", "U", "iO", "iU"},
    "close_vowel": {"I", "EU"},
    "palatal": {"CHh"}
}


def get_phoneme_category(phoneme: str) -> str:
    """
    Get articulatory category for a phoneme.

    Args:
        phoneme: Korean MFA phoneme

    Returns:
        Category name or "unknown"
    """
    for category, phonemes in PHONEME_CATEGORIES.items():
        if phoneme in phonemes:
            return category
    return "unknown"


# ============================================================================
# Expected MAR (Mouth Aspect Ratio) Ranges for Validation
# ============================================================================

# Reference ranges from phoneme_classifier.py analysis
EXPECTED_MAR_RANGES = {
    "M": (0.10, 0.30),   # Bilabial - lips closed
    "B": (0.10, 0.30),
    "BB": (0.10, 0.30),
    "Ph": (0.10, 0.30),

    "A": (0.60, 0.90),   # Open vowels - mouth open
    "E": (0.50, 0.80),
    "iA": (0.55, 0.85),

    "O": (0.20, 0.50),   # Rounded vowels - lips rounded
    "U": (0.20, 0.50),
    "iO": (0.25, 0.55),
    "iU": (0.25, 0.55),

    "I": (0.30, 0.60),   # Close vowels - moderate opening
    "EU": (0.30, 0.60),

    "CHh": (0.25, 0.55)  # Palatal - moderate rounding
}


def validate_mar_range(phoneme: str, mar_value: float) -> bool:
    """
    Validate if MAR value is within expected range for phoneme.

    Args:
        phoneme: Korean MFA phoneme
        mar_value: Measured mouth aspect ratio

    Returns:
        True if MAR is within expected range (±20% tolerance)
    """
    if phoneme not in EXPECTED_MAR_RANGES:
        return True  # Unknown phoneme, skip validation

    min_mar, max_mar = EXPECTED_MAR_RANGES[phoneme]
    tolerance = 0.2
    return (min_mar - tolerance) <= mar_value <= (max_mar + tolerance)


def phoneme_to_korean(phoneme: str) -> str:
    """
    Convert MFA phoneme code to Korean character for user-friendly display.

    Args:
        phoneme: MFA phoneme code (e.g., "M", "A", "iO", "G")

    Returns:
        Korean character (e.g., "ㅁ", "ㅏ", "ㅛ", "ㄱ") or original if not in mapping
    """
    return PHONEME_TO_KOREAN.get(phoneme, phoneme)
