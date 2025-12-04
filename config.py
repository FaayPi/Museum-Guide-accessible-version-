"""
Application Configuration with Environment Support.

Supports multiple environments: development, production, testing
Configuration is loaded from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv
from typing import Literal

# Load environment variables from .env file
load_dotenv()

# ==================== ENVIRONMENT ====================
Environment = Literal['development', 'production', 'testing']
ENVIRONMENT: Environment = os.getenv('ENVIRONMENT', 'development')  # type: ignore

# ==================== API KEYS ====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Validate required API keys
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# OpenAI Models
VISION_MODEL = "gpt-4o-mini"  # gpt-4o-mini: 5-8x faster than gpt-4o, still excellent for artwork analysis
CHAT_MODEL = "gpt-4o-mini"
TTS_MODEL = "tts-1"  # or "tts-1-hd" for higher quality
TTS_VOICE = "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer

# ==================== APP SETTINGS ====================
APP_TITLE = "Museum Audio Guide"
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "webp"]

# Server settings (for production deployment)
HOST = os.getenv('HOST', '127.0.0.1')
PORT = int(os.getenv('PORT', 7860))

# ==================== AUDIO SETTINGS ====================
AUDIO_SAMPLE_RATE = 16000
AUDIO_FORMAT = "wav"

# ==================== PERFORMANCE SETTINGS ====================
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 4))
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 30))

# ==================== FEATURE FLAGS ====================
ENABLE_RAG = os.getenv('ENABLE_RAG', 'true').lower() == 'true'
ENABLE_CACHE = os.getenv('ENABLE_CACHE', 'true').lower() == 'true'
ENABLE_METRICS = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'

# ==================== RATE LIMITING ====================
MAX_REQUESTS_PER_MINUTE = int(os.getenv('MAX_REQUESTS_PER_MINUTE', 60))
MAX_REQUESTS_PER_HOUR = int(os.getenv('MAX_REQUESTS_PER_HOUR', 1000))

# ==================== LOGGING ====================
LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG' if ENVIRONMENT == 'development' else 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'logs/app.log')

# ==================== MONITORING ====================
ENABLE_MONITORING = os.getenv('ENABLE_MONITORING', 'false').lower() == 'true'
METRICS_PORT = int(os.getenv('METRICS_PORT', 9090))

# Prompts
VISION_PROMPT = """Describe this artwork in 2-3 sentences covering:
1. Main subject and colors
2. Style and mood

Be concise and engaging."""  # ⚡ OPTIMIZED: Shorter prompt = faster response

METADATA_PROMPT = """Analyze this artwork carefully and extract metadata. Return ONLY valid JSON in this exact format:

{
  "artist": "Artist name (or 'Unknown' if cannot identify)",
  "title": "Artwork title (or 'Unknown' if cannot identify)",
  "year": "Year created (or 'Unknown' if cannot identify)",
  "period": "Art period/movement (e.g., Renaissance, Baroque, Impressionism, Surrealism, Contemporary, etc.)",
  "confidence": "high/medium/low - your confidence level in the identification"
}

IMPORTANT:
1. Try your BEST to identify the artwork - check style, technique, composition, subject matter
2. Even if you're not 100% certain, make an educated guess based on:
   - Art style and technique
   - Historical period indicators
   - Subject matter and composition
   - Color palette and brushwork
3. For "period", ALWAYS provide an art movement/period based on visual analysis, never leave it as "Unknown"
4. If you recognize the specific artwork, provide accurate details
5. If unsure about artist/title, focus on accurate period identification from visual style
6. Return ONLY the JSON, no other text

Examples of periods: Renaissance, Baroque, Rococo, Neoclassicism, Romanticism, Realism, Impressionism, Post-Impressionism, Expressionism, Cubism, Surrealism, Abstract Expressionism, Pop Art, Contemporary Art, etc."""
