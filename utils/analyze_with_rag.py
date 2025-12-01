"""
Artwork analysis with RAG fallback

This module provides artwork analysis that uses OpenAI Vision first,
and falls back to the RAG database if OpenAI cannot identify the artwork.
"""

from utils.vision import analyze_artwork, get_metadata
from utils.rag_database_openai import ArtworkRAGOpenAI
import io
from PIL import Image


# Global RAG instance (lazy loaded)
_rag_instance = None


def get_rag_instance():
    """Get or create RAG instance"""
    global _rag_instance
    if _rag_instance is None:
        try:
            print("Initializing RAG database...")
            _rag_instance = ArtworkRAGOpenAI()
            print("✓ RAG database ready")
        except Exception as e:
            print(f"ERROR: Could not initialize RAG database: {e}")
            return None
    return _rag_instance


def analyze_artwork_with_rag_fallback(image_input):
    """
    Analyze artwork with RAG priority

    First checks RAG database for exact match. If no exact match found (similarity < 0.90),
    falls back to OpenAI Vision for general artwork analysis.

    Args:
        image_input: PIL Image or bytes

    Returns:
        tuple: (description, metadata, from_rag: bool)
    """
    # Convert PIL Image to bytes if needed
    if isinstance(image_input, Image.Image):
        img_byte_arr = io.BytesIO()
        image_input.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()
    else:
        image_bytes = image_input

    # STEP 1: Try RAG database FIRST for exact match
    print("\n=== Checking RAG database for exact match ===")
    rag = get_rag_instance()
    if rag:
        rag_result = rag.search_exact_match(image_bytes)

        if rag_result:
            print("✓ Found exact match in RAG database!")
            print(f"  Artist: {rag_result['metadata']['artist']}")
            print(f"  Title: {rag_result['metadata']['title']}")
            print(f"  Similarity: {rag_result['similarity_score']:.3f}")
            # Use RAG data - this is from our Special Exhibition
            return (
                rag_result['description'],  # Full description from RAG
                rag_result['metadata'],  # Metadata from RAG
                True  # from_rag flag
            )
        else:
            print("No exact match found in RAG database (similarity < 0.90)")

    # STEP 2: Fallback to OpenAI Vision if no RAG match
    print("\n=== Falling back to OpenAI Vision ===")
    description = analyze_artwork(image_bytes)
    metadata = get_metadata(image_bytes)

    # Return OpenAI Vision results
    return (description, metadata, False)


def format_metadata_text(metadata, from_rag=False):
    """Format metadata into display text"""
    if not metadata:
        return "Could not extract metadata"

    source = " (from Special Exhibition Database)" if from_rag else ""

    text = f"""**Artist:** {metadata.get('artist', 'Unknown')}{source}
**Title:** {metadata.get('title', 'Unknown')}
**Year:** {metadata.get('year', 'Unknown')}
**Period:** {metadata.get('period', 'Unknown')}
**Confidence:** {metadata.get('confidence', 'low')}
"""
    return text
