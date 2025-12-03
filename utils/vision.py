"""Vision API functions using OpenAI GPT-4 Vision"""

import base64
import json
from openai import OpenAI
from PIL import Image
import io
import config


def encode_image(image_file):
    """Encode image to base64 string"""
    if isinstance(image_file, bytes):
        return base64.b64encode(image_file).decode('utf-8')
    else:
        # If it's a file-like object
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_artwork(image_file):
    """
    Analyze artwork image and return detailed description
    
    Args:
        image_file: Image file (bytes or file-like object)
    
    Returns:
        str: Detailed description of the artwork
    """
    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Encode image
        base64_image = encode_image(image_file)
        
        # Call OpenAI Vision API
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
            max_tokens=100  # ⚡ OPTIMIZED: Reduced from 150 for faster response
        )

        description = response.choices[0].message.content
        return description
        
    except Exception as e:
        print(f"Vision API Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def get_metadata(image_file):
    """
    Extract metadata (artist, title, year, period) from artwork
    
    Args:
        image_file: Image file (bytes or file-like object)
    
    Returns:
        dict: Metadata dictionary with keys: artist, title, year, period, confidence
    """
    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Encode image
        base64_image = encode_image(image_file)
        
        # Call OpenAI Vision API
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
            max_tokens=60  # ⚡ OPTIMIZED: Reduced from 100 for faster response (JSON only)
        )
        
        # Parse JSON response
        metadata_text = response.choices[0].message.content
        
        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            if "```json" in metadata_text:
                metadata_text = metadata_text.split("```json")[1].split("```")[0]
            elif "```" in metadata_text:
                metadata_text = metadata_text.split("```")[1].split("```")[0]
            
            metadata = json.loads(metadata_text.strip())
        except:
            # Fallback if JSON parsing fails
            metadata = {
                "artist": "Unknown",
                "title": "Unknown",
                "year": "Unknown",
                "period": "Unknown",
                "confidence": "low"
            }
        
        return metadata
        
    except Exception as e:
        print(f"Metadata extraction error: {str(e)}")
        return None  # Return None on error
