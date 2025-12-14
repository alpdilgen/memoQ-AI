# config.py
import os

# Application Settings
APP_NAME = "Enhanced Translation Assistant"
VERSION = "1.0.0"

# Language Settings
SUPPORTED_LANGUAGES = {
    "bg": "Bulgarian",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "ro": "Romanian",
    "tr": "Turkish",
    "cs": "Czech",
    "pl": "Polish",
    "nl": "Dutch",
    "pt": "Portuguese"
}

# Processing Settings
BATCH_SIZE = 20  # Segments per AI request
DEFAULT_MATCH_THRESHOLD = 70       # Minimum for TM context (sent to LLM)
DEFAULT_ACCEPTANCE_THRESHOLD = 95  # Minimum for direct TM usage (bypass LLM)
DEFAULT_CHAT_HISTORY = 5           # Number of previous batches to include as context
DEFAULT_TEMPERATURE = 0.1
MAX_TOKENS = 4000
TIMEOUT_SECONDS = 45
MAX_RETRIES = 3

# AI Model Settings
OPENAI_MODELS = ["gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini", "o3", "o4-mini"]
ANTHROPIC_MODELS = ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"]

# File Paths
RESOURCES_DIR = "resources"
PROMPT_TEMPLATE_PATH = os.path.join(RESOURCES_DIR, "prompt Template.txt")