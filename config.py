# Configuration for Enhanced Translation Assistant

# Supported languages - EU languages + major world languages
# Format: 'language_code': 'Language Name'
SUPPORTED_LANGUAGES = {
    # EU Languages
    'bul': 'Bulgarian',
    'hrv': 'Croatian',
    'ces': 'Czech',
    'dan': 'Danish',
    'nld': 'Dutch',
    'eng': 'English',
    'est': 'Estonian',
    'fin': 'Finnish',
    'fra': 'French',
    'deu': 'German',
    'ell': 'Greek',
    'hun': 'Hungarian',
    'gle': 'Irish',
    'ita': 'Italian',
    'lav': 'Latvian',
    'lit': 'Lithuanian',
    'mlt': 'Maltese',
    'pol': 'Polish',
    'por': 'Portuguese',
    'ron': 'Romanian',
    'slk': 'Slovak',
    'slv': 'Slovenian',
    'spa': 'Spanish',
    'swe': 'Swedish',
    
    # Major World Languages
    'ara': 'Arabic',
    'zho': 'Chinese (Simplified)',
    'zht': 'Chinese (Traditional)',
    'jpn': 'Japanese',
    'kor': 'Korean',
    'rus': 'Russian',
    'tur': 'Turkish',
    'hin': 'Hindi',
    'ben': 'Bengali',
    'vie': 'Vietnamese',
    'tha': 'Thai',
    'afr': 'Afrikaans',
    'heb': 'Hebrew',
    'ukr': 'Ukrainian',
    'nor': 'Norwegian',
}

# OpenAI Models
OPENAI_MODELS = [
    'gpt-4o',
    'gpt-4-turbo',
]

# Default values
DEFAULT_SOURCE_LANGUAGE = 'eng'
DEFAULT_TARGET_LANGUAGE = 'tur'
DEFAULT_MODEL = 'gpt-4o'

# Translation settings
ACCEPTANCE_THRESHOLD = 95  # % - bypass segments at this match or higher
MATCH_THRESHOLD = 70       # % - use fuzzy match for TM context
CHAT_HISTORY_LENGTH = 5    # segments to include in history for consistency

# API Settings
OPENAI_API_BASE = "https://api.openai.com/v1"

# App name
APP_NAME = "Enhanced Translation Assistant"

# UI Settings
LAYOUT = "wide"
THEME = "light"

# Batch processing
DEFAULT_BATCH_SIZE = 20
MAX_BATCH_SIZE = 50

# Cost calculation
TOKENS_PER_SEGMENT = 100
GPT_4O_INPUT_PRICE = 0.00025   # per token
GPT_4O_OUTPUT_PRICE = 0.001    # per token
CONTEXT_DISCOUNT = 0.5         # 50% discount for fuzzy match segments
