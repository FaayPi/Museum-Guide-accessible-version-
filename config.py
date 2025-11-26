import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI Models
VISION_MODEL = "gpt-4o"  # gpt-4o hat Vision-Capabilities
CHAT_MODEL = "gpt-4o"
TTS_MODEL = "tts-1"  # or "tts-1-hd" for higher quality
TTS_VOICE = "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer

# App Settings
APP_TITLE = "Museum Audio Guide"
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "webp"]

# Audio Settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_FORMAT = "wav"

# Prompts
VISION_PROMPT = """Analyze this artwork in detail and provide a comprehensive description in English. Include:

**What you see:**
- Main subjects, figures, objects, and scene
- Visual elements and composition
- Colors, lighting, and atmosphere

**Artistic analysis:**
- Art style and technique
- Brushwork and medium (if visible)
- Composition and perspective
- Mood and emotional impact

**Historical context (if recognizable):**
- If you recognize this specific artwork, identify:
  * Artist name
  * Title of the work
  * Year created
  * Art period/movement
- If not certain, describe the style characteristics that indicate the likely period

Provide a thorough yet accessible description suitable for both art enthusiasts and general audiences. Write in a engaging, descriptive style that helps someone visualize the artwork."""

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
