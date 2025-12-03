"""Vision API functions using OpenAI GPT-4 Vision with robust error handling"""

import base64
import json
from openai import OpenAI
from PIL import Image
import io
import config
from src.core.error_handler import retry_on_failure, APIError, logger


def encode_image(image_file):
    """Encode image to base64 string"""
    if isinstance(image_file, bytes):
        return base64.b64encode(image_file).decode('utf-8')
    else:
        # If it's a file-like object
        return base64.b64encode(image_file.read()).decode('utf-8')


@retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
def generate_description(image_file):
    """
    Analyze artwork image and return detailed description with robust error handling

    Args:
        image_file: Image file (bytes or file-like object)

    Returns:
        str: Detailed description of the artwork

    Raises:
        APIError: If API call fails after retries
    """
    try:
        if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your_openai_api_key_here":
            raise APIError("OpenAI API key not configured")

        client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=30.0)

        # Encode image
        base64_image = encode_image(image_file)

        if not base64_image:
            raise APIError("Failed to encode image")

        # Call OpenAI Vision API
        logger.info("Calling Vision API for artwork description")
        response = client.chat.completions.create(
            model=config.VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": config.VISION_PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100,  # ⚡ OPTIMIZED: Reduced from 150 for faster response
            temperature=0.7
        )

        description = response.choices[0].message.content

        if not description or len(description.strip()) == 0:
            raise APIError("Empty response from Vision API")

        logger.info(f"Vision API returned description ({len(description)} chars)")
        return description.strip()

    except APIError:
        raise  # Re-raise API errors for retry logic
    except Exception as e:
        logger.error(f"Vision API Error: {str(e)}")
        raise APIError(f"Vision API call failed: {str(e)}")


@retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
def extract_metadata(image_file):
    """
    Extract metadata (artist, title, year, period) from artwork with robust error handling

    Args:
        image_file: Image file (bytes or file-like object)

    Returns:
        dict: Metadata dictionary with keys: artist, title, year, period, confidence

    Raises:
        APIError: If API call fails after retries
    """
    try:
        if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your_openai_api_key_here":
            raise APIError("OpenAI API key not configured")

        client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=30.0)

        # Encode image
        base64_image = encode_image(image_file)

        if not base64_image:
            raise APIError("Failed to encode image")

        # Call OpenAI Vision API
        logger.info("Calling Vision API for metadata extraction")
        response = client.chat.completions.create(
            model=config.VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": config.METADATA_PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=60,  # ⚡ OPTIMIZED: Reduced from 100 for faster response (JSON only)
            temperature=0.3  # Lower temperature for more consistent metadata
        )

        # Parse JSON response
        metadata_text = response.choices[0].message.content

        if not metadata_text or len(metadata_text.strip()) == 0:
            raise APIError("Empty response from Vision API")

        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            if "```json" in metadata_text:
                metadata_text = metadata_text.split("```json")[1].split("```")[0]
            elif "```" in metadata_text:
                metadata_text = metadata_text.split("```")[1].split("```")[0]

            metadata = json.loads(metadata_text.strip())

            # Validate required fields
            required_fields = ['artist', 'title', 'year', 'period']
            for field in required_fields:
                if field not in metadata:
                    metadata[field] = 'Unknown'

            if 'confidence' not in metadata:
                metadata['confidence'] = 'low'

            logger.info(f"Metadata extracted: {metadata.get('artist')} - {metadata.get('title')}")
            return metadata

        except json.JSONDecodeError as je:
            logger.warning(f"JSON parsing failed: {je}. Using fallback metadata.")
            # Graceful degradation - return fallback metadata
            return {
                "artist": "Unknown",
                "title": "Unknown",
                "year": "Unknown",
                "period": "Contemporary",  # Default to Contemporary if can't determine
                "confidence": "low"
            }

    except APIError:
        raise  # Re-raise API errors for retry logic
    except Exception as e:
        logger.error(f"Metadata extraction error: {str(e)}")
        raise APIError(f"Metadata extraction failed: {str(e)}")
