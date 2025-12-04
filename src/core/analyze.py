"""
Artwork analysis with RAG fallback

This module provides artwork analysis that uses OpenAI Vision first,
and falls back to the RAG database if OpenAI cannot identify the artwork.
"""

from src.services.vision import generate_description, extract_metadata
from src.services.rag_database import ArtworkRAGOpenAI
from src.services.image_similarity import ImageSimilarityIndex
import io
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time
import numpy as np


# Global instances (lazy loaded)
_rag_instance = None
_similarity_index = None


def get_rag_instance():
    """Get or create RAG instance"""
    global _rag_instance
    if _rag_instance is None:
        try:
            _rag_instance = ArtworkRAGOpenAI()
        except Exception as e:
            print(f"ERROR: RAG initialization failed - {e}")
            return None
    return _rag_instance


def get_similarity_index():
    """Get or create similarity index instance"""
    global _similarity_index
    if _similarity_index is None:
        try:
            _similarity_index = ImageSimilarityIndex()
        except Exception as e:
            print(f"ERROR: Similarity index failed - {e}")
            return None
    return _similarity_index


def is_likely_generic_artwork(image_bytes):
    """
    Fast pre-check to detect generic/unknown artworks

    This helps skip expensive RAG searches for images that are unlikely
    to be in our database (photos, simple sketches, modern digital art, etc.)

    Returns:
        bool: True if likely generic (skip RAG), False if might be known artwork
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize for faster analysis
        img.thumbnail((200, 200))
        img_array = np.array(img)

        # Quick heuristics (all checks are very fast, <50ms total)

        # 1. Check image complexity (simple images unlikely to be museum art)
        # Calculate edge density using simple gradient
        gray = np.mean(img_array, axis=2)
        grad_x = np.abs(np.diff(gray, axis=1))
        grad_y = np.abs(np.diff(gray, axis=0))
        edge_density = (np.mean(grad_x) + np.mean(grad_y)) / 2

        # Very low complexity = likely simple photo/sketch
        if edge_density < 5:
            return True

        # 2. Check color variance (museum paintings usually have rich color)
        color_std = np.std(img_array, axis=(0, 1))
        avg_color_variance = np.mean(color_std)

        # Very low color variance = likely plain photo/document
        if avg_color_variance < 15:
            return True

        # 3. Check if image is mostly one color (screenshots, graphics, etc.)
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
        total_pixels = img_array.shape[0] * img_array.shape[1]
        color_diversity = unique_colors / total_pixels

        if color_diversity < 0.1:
            return True

        return False

    except Exception as e:
        return False  # On error, default to doing RAG search


def analyze_artwork(image_input):
    """
    ⚡ OPTIMIZED: Analyze artwork with multi-tier fallback strategy:
    1. Perceptual hash match (~0.25s) - FAST!
    2. Fast pre-check (~0.05s) - Skip RAG for generic images
    3. RAG semantic search with TIMEOUT (max 4s) - Accurate
    4. OpenAI Vision fallback (~2-3s with gpt-4o-mini + parallel) - General

    Args:
        image_input: PIL Image or bytes

    Returns:
        tuple: (description, metadata, from_rag: bool)
    """
    overall_start = time.time()

    # Convert PIL Image to bytes if needed
    if isinstance(image_input, Image.Image):
        img_byte_arr = io.BytesIO()
        image_input.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()
    else:
        image_bytes = image_input

    # TIER 1: Try perceptual hash match FIRST (lightning fast!)
    print("\n=== TIER 1: Checking perceptual hash similarity ===")
    tier1_start = time.time()
    similarity_index = get_similarity_index()
    if similarity_index and similarity_index.hash_index:
        hash_match = similarity_index.find_match(image_bytes, threshold=10)

        if hash_match:
            tier1_time = time.time() - tier1_start
            print(f"✓✓✓ FAST MATCH FOUND via perceptual hashing! ({tier1_time:.2f}s)")
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
            tier1_time = time.time() - tier1_start
            print(f"No perceptual hash match ({tier1_time:.2f}s)")

    # TIER 1.5: ⚡ NEW - Fast pre-check to skip RAG for generic images
    print("\n=== TIER 1.5: Fast pre-check for generic artwork ===")
    precheck_start = time.time()
    if is_likely_generic_artwork(image_bytes):
        precheck_time = time.time() - precheck_start
        print(f"⚡⚡⚡ FAST PATH: Generic artwork detected ({precheck_time:.2f}s) - skipping RAG, going to Vision")

        # Skip RAG, go straight to Vision
        print("\n=== TIER 3: OpenAI Vision (fast path) ===")
        print("⚡ Running description and metadata extraction in parallel...")

        with ThreadPoolExecutor(max_workers=2) as executor:
            desc_future = executor.submit(generate_description, image_bytes)
            meta_future = executor.submit(extract_metadata, image_bytes)

            description = desc_future.result()
            metadata = meta_future.result()

        total_time = time.time() - overall_start
        print(f"⏱️  Total time (fast path): {total_time:.2f}s")
        return (description, metadata, False)

    precheck_time = time.time() - precheck_start
    print(f"Pre-check passed ({precheck_time:.2f}s) - might be known artwork, checking RAG")

    # TIER 2: ⚡ OPTIMIZED - Try RAG database semantic search with TIMEOUT
    print("\n=== TIER 2: Checking RAG database with timeout ===")
    tier2_start = time.time()
    rag = get_rag_instance()

    if rag:
        RAG_TIMEOUT = 2.5  # ⚡ AGGRESSIVE: Maximum 2.5 seconds for RAG search (ultra-fast fallback)

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                rag_future = executor.submit(rag.search_exact_match, image_bytes)

                try:
                    # Wait for result with timeout
                    rag_result = rag_future.result(timeout=RAG_TIMEOUT)

                    if rag_result:
                        tier2_time = time.time() - tier2_start
                        print(f"✓ Found exact match in RAG database! ({tier2_time:.2f}s)")
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
                        tier2_time = time.time() - tier2_start
                        print(f"No exact match found in RAG database ({tier2_time:.2f}s, similarity < 0.85)")

                except TimeoutError:
                    tier2_time = time.time() - tier2_start
                    print(f"⚡ RAG search timeout ({tier2_time:.2f}s) - falling back to Vision API")

        except Exception as e:
            tier2_time = time.time() - tier2_start
            print(f"⚠️  RAG search error ({tier2_time:.2f}s): {e}")

    # TIER 3: Fallback to OpenAI Vision (with parallel execution)
    print("\n=== TIER 3: Falling back to OpenAI Vision ===")
    tier3_start = time.time()
    print("⚡ Running description and metadata extraction in parallel...")

    # Run both Vision calls in parallel for faster processing
    with ThreadPoolExecutor(max_workers=2) as executor:
        desc_future = executor.submit(generate_description, image_bytes)
        meta_future = executor.submit(extract_metadata, image_bytes)

        description = desc_future.result()
        metadata = meta_future.result()

    tier3_time = time.time() - tier3_start
    total_time = time.time() - overall_start
    print(f"⏱️  Vision API time: {tier3_time:.2f}s")
    print(f"⏱️  Total time: {total_time:.2f}s")

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
