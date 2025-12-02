"""
Artwork analysis with RAG fallback

This module provides artwork analysis that uses OpenAI Vision first,
and falls back to the RAG database if OpenAI cannot identify the artwork.
"""

from utils.vision import analyze_artwork, get_metadata
from utils.rag_database_openai import ArtworkRAGOpenAI
from utils.image_similarity import ImageSimilarityIndex
import io
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


# Global instances (lazy loaded)
_rag_instance = None
_similarity_index = None


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


def get_similarity_index():
    """Get or create similarity index instance"""
    global _similarity_index
    if _similarity_index is None:
        try:
            print("Loading image similarity index...")
            _similarity_index = ImageSimilarityIndex()
            if _similarity_index.hash_index:
                print(f"✓ Similarity index ready ({len(_similarity_index.hash_index)} images)")
            else:
                print("⚠️  Similarity index is empty - will fall back to RAG search")
        except Exception as e:
            print(f"ERROR: Could not load similarity index: {e}")
            return None
    return _similarity_index


def analyze_artwork_with_rag_fallback(image_input):
    """
    Analyze artwork with multi-tier fallback strategy:
    1. Perceptual hash match (~0.25s) - FAST!
    2. RAG semantic search (~5-8s with gpt-4o-mini) - Accurate
    3. OpenAI Vision fallback (~2-3s with gpt-4o-mini + parallel) - General

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

    # TIER 1: Try perceptual hash match FIRST (lightning fast!)
    print("\n=== TIER 1: Checking perceptual hash similarity ===")
    similarity_index = get_similarity_index()
    if similarity_index and similarity_index.hash_index:
        hash_match = similarity_index.find_match(image_bytes, threshold=10)

        if hash_match:
            print("✓✓✓ FAST MATCH FOUND via perceptual hashing!")
            print(f"  Artist: {hash_match['metadata']['artist']}")
            print(f"  Title: {hash_match['metadata']['title']}")
            print(f"  Distance: {hash_match['distance']:.1f}")

            # Return the matched data (already includes full description!)
            return (
                hash_match['description'],  # Full description from index
                hash_match['metadata'],  # Metadata from index
                True  # from_hash_match (Special Exhibition)
            )
        else:
            print("No perceptual hash match - trying RAG semantic search...")

    # TIER 2: Try RAG database semantic search
    print("\n=== TIER 2: Checking RAG database for exact match ===")
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
            print("No exact match found in RAG database (similarity < 0.85)")

    # TIER 3: Fallback to OpenAI Vision (with parallel execution)
    print("\n=== TIER 3: Falling back to OpenAI Vision ===")
    print("⚡ Running description and metadata extraction in parallel...")

    # Run both Vision calls in parallel for faster processing
    with ThreadPoolExecutor(max_workers=2) as executor:
        desc_future = executor.submit(analyze_artwork, image_bytes)
        meta_future = executor.submit(get_metadata, image_bytes)

        description = desc_future.result()
        metadata = meta_future.result()

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
