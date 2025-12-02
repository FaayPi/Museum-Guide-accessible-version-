# Performance Optimization Roadmap
## Target: 3-10 Second Latency

This document outlines concrete options to achieve the 3-10 second target for image analysis.

---

## 🎯 Recommended Implementation Path

### Phase 1: Quick Win - Use GPT-4o-mini for RAG Search
**Effort:** 2 hours | **Impact:** 15-20s savings | **Result:** 5-8s total ✅

**Change in [utils/rag_database_openai.py:245](utils/rag_database_openai.py#L245):**

```python
def search_exact_match(self, query_image_bytes):
    print("\n=== RAG Fallback: Searching for exact match ===")

    # OPTIMIZATION: Use GPT-4o-mini instead of GPT-4o for faster RAG search
    print("Analyzing query image with mini model...")
    query_description = self._analyze_with_mini_model(query_image_bytes)  # NEW

    if not query_description:
        print("Failed to analyze query image")
        return None

    # ... rest stays the same ...

def _analyze_with_mini_model(self, image_bytes):
    """Fast artwork analysis using GPT-4o-mini for RAG matching"""
    import base64

    b64_image = base64.b64encode(image_bytes).decode('utf-8')

    try:
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",  # 5-8x faster than gpt-4o
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                    },
                    {
                        "type": "text",
                        "text": "Describe this artwork in 2-3 sentences focusing on: subject, colors, style, mood."
                    }
                ]
            }],
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with mini model: {e}")
        return None
```

**Expected Performance:**
- RAG search: 5-8 seconds (was 20-30s)
- TTS parallel: 6-7 seconds
- **Total: 11-15 seconds** (better, but still above target)

---

### Phase 2: High Impact - Add Perceptual Hashing Layer
**Effort:** 4-6 hours | **Impact:** 20-25s savings for matches | **Result:** 0.5-2s for known artworks ✅✅

**Install dependency:**
```bash
pip install imagehash
echo "imagehash>=4.3.1" >> requirements_gradio.txt
```

**Create [utils/image_similarity.py](utils/image_similarity.py):**

```python
"""Fast image similarity search using perceptual hashing"""

import imagehash
from PIL import Image
import io
from pathlib import Path
import json

class ImageSimilarityIndex:
    """Lightning-fast image matching using perceptual hashing"""

    def __init__(self, index_file="image_hash_index.json"):
        self.index_file = index_file
        self.hash_index = self._load_index()

    def _load_index(self):
        """Load hash index from disk"""
        if Path(self.index_file).exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Save hash index to disk"""
        with open(self.index_file, 'w') as f:
            json.dump(self.hash_index, f)

    def add_image(self, image_path, metadata):
        """Add image to similarity index"""
        img = Image.open(image_path)

        # Generate multiple hashes for robustness
        phash = str(imagehash.phash(img, hash_size=16))
        dhash = str(imagehash.dhash(img, hash_size=16))

        image_id = Path(image_path).stem

        self.hash_index[image_id] = {
            'phash': phash,
            'dhash': dhash,
            'metadata': metadata,
            'filepath': str(image_path)
        }

        self._save_index()
        print(f"✓ Added to similarity index: {image_id}")

    def find_match(self, image_bytes, threshold=10):
        """
        Find matching image using perceptual hashing

        Args:
            image_bytes: Query image as bytes
            threshold: Hamming distance threshold (lower = stricter)

        Returns:
            dict or None: Matched image metadata if found
        """
        img = Image.open(io.BytesIO(image_bytes))

        # Generate query hashes
        query_phash = imagehash.phash(img, hash_size=16)
        query_dhash = imagehash.dhash(img, hash_size=16)

        best_match = None
        best_distance = float('inf')

        # Compare against all indexed images
        for image_id, data in self.hash_index.items():
            stored_phash = imagehash.hex_to_hash(data['phash'])
            stored_dhash = imagehash.hex_to_hash(data['dhash'])

            # Calculate combined distance
            phash_dist = query_phash - stored_phash
            dhash_dist = query_dhash - stored_dhash
            combined_dist = (phash_dist + dhash_dist) / 2

            if combined_dist < best_distance:
                best_distance = combined_dist
                best_match = {
                    'image_id': image_id,
                    'metadata': data['metadata'],
                    'distance': combined_dist,
                    'filepath': data['filepath']
                }

        # Check if match is close enough
        if best_match and best_match['distance'] < threshold:
            print(f"✓ Found similar image: {best_match['image_id']} (distance: {best_match['distance']:.1f})")
            return best_match
        else:
            print(f"No similar image found (best distance: {best_distance:.1f} > threshold: {threshold})")
            return None

def index_all_artworks():
    """Index all RAG database artworks for fast similarity search"""
    from utils.rag_database_openai import ArtworkRAGOpenAI

    print("\n=== Building Image Similarity Index ===")

    similarity_index = ImageSimilarityIndex()
    rag = ArtworkRAGOpenAI()

    # Get all indexed artworks from Pinecone
    stats = rag.get_collection_stats()
    print(f"Indexing {stats['total_artworks']} artworks...")

    rag_folder = Path("RAG_database")
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []

    for ext in image_extensions:
        image_files.extend(rag_folder.glob(f'*{ext}'))
        image_files.extend(rag_folder.glob(f'*{ext.upper()}'))

    for img_path in image_files:
        # Get metadata from RAG
        # (In practice, you'd fetch this from Pinecone)
        metadata = {
            'filename': img_path.name,
            'artist': 'Fee Pieper',
            'title': img_path.stem
        }

        similarity_index.add_image(str(img_path), metadata)

    print(f"\n✓ Similarity index built: {len(similarity_index.hash_index)} images")
    return similarity_index

if __name__ == "__main__":
    index_all_artworks()
```

**Modify [utils/analyze_with_rag.py](utils/analyze_with_rag.py#L32):**

```python
from utils.vision import analyze_artwork, get_metadata
from utils.rag_database_openai import ArtworkRAGOpenAI
from utils.image_similarity import ImageSimilarityIndex  # NEW
import io
from PIL import Image

# Global instances
_rag_instance = None
_similarity_index = None  # NEW

def get_similarity_index():
    """Get or create similarity index instance"""
    global _similarity_index
    if _similarity_index is None:
        try:
            print("Loading image similarity index...")
            _similarity_index = ImageSimilarityIndex()
            print(f"✓ Similarity index ready ({len(_similarity_index.hash_index)} images)")
        except Exception as e:
            print(f"ERROR: Could not load similarity index: {e}")
            return None
    return _similarity_index

def analyze_artwork_with_rag_fallback(image_input):
    """
    Analyze artwork with multi-tier fallback:
    1. Fast perceptual hash match (~0.5s)
    2. RAG semantic search (~20s)
    3. OpenAI Vision fallback (~15s)
    """
    # Convert PIL Image to bytes if needed
    if isinstance(image_input, Image.Image):
        img_byte_arr = io.BytesIO()
        image_input.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()
    else:
        image_bytes = image_input

    # TIER 1: Try fast perceptual hash match FIRST
    print("\n=== TIER 1: Checking perceptual hash similarity ===")
    similarity_index = get_similarity_index()
    if similarity_index:
        hash_match = similarity_index.find_match(image_bytes, threshold=10)

        if hash_match:
            print("✓✓✓ FAST MATCH FOUND via perceptual hashing!")
            print(f"  Distance: {hash_match['distance']:.1f}")

            # Get full data from RAG database
            rag = get_rag_instance()
            if rag:
                # Use the matched filepath to get full description
                # In practice, store full metadata in similarity index
                # For now, fall through to RAG for full data
                print("  Fetching full description from RAG...")

            # Return the matched metadata
            # (In production, store full description in similarity index)
            return (
                f"This is {hash_match['metadata']['title']} by {hash_match['metadata']['artist']}.",
                hash_match['metadata'],
                True  # from_hash_match
            )

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
            return (
                rag_result['description'],
                rag_result['metadata'],
                True  # from_rag
            )
        else:
            print("No exact match found in RAG database")

    # TIER 3: Fallback to OpenAI Vision
    print("\n=== TIER 3: Falling back to OpenAI Vision ===")
    description = analyze_artwork(image_bytes)
    metadata = get_metadata(image_bytes)

    return (description, metadata, False)
```

**Expected Performance:**
- Known artwork (hash match): **0.5-2 seconds** ✅✅✅
- Unknown artwork (Vision fallback): 15-20 seconds
- **Average: 3-5 seconds** for Special Exhibition artworks ✅

---

### Phase 3: Polish - Enhance Similarity Index
**Effort:** 2-3 hours | **Impact:** Better accuracy

**Improvements:**

1. **Store full descriptions in similarity index** (no RAG lookup needed)
2. **Add multiple hash types** (average-hash, wavelet-hash)
3. **Implement fuzzy matching** for slightly rotated/cropped images
4. **Pre-load index at app startup** alongside RAG database

---

## 📊 Performance Comparison

| Strategy | Known Artwork | Unknown Artwork | Complexity | Cost Impact |
|----------|---------------|-----------------|------------|-------------|
| **Current** | 20-30s | 20-30s | - | High |
| **Phase 1 (Mini)** | 5-8s | 11-15s | Low | Medium (-60% cost) |
| **Phase 2 (Hash)** | 0.5-2s | 15-20s | Medium | Low (-90% for matches) |
| **Phase 1+2** | 0.5-2s | 8-12s | Medium | Very Low |

---

## 💰 Cost Analysis

Current costs per query:
- GPT-4o Vision analysis: ~$0.01-0.02
- Text embedding: ~$0.0001
- TTS: ~$0.01

**Phase 1 savings:**
- GPT-4o-mini: ~$0.001-0.002 (5-10x cheaper)
- **Savings: ~$0.009-0.018 per query** (60-90% cost reduction)

**Phase 2 savings:**
- Perceptual hash: $0 (local computation)
- **Savings: ~$0.02-0.03 per known artwork match** (95% cost reduction)

For 1000 queries with 80% known artworks:
- Current: $20-30
- Phase 1: $8-12 (60% savings)
- Phase 2: $4-6 (80% savings)

---

## 🚀 Implementation Priority

1. **Start with Phase 1** (GPT-4o-mini)
   - Quick to implement
   - Immediate 3x speedup
   - Significant cost savings
   - Tests well with current architecture

2. **Then add Phase 2** (Perceptual hashing)
   - Maximum impact for Special Exhibition use case
   - Near-instant responses for target artworks
   - Minimal API costs

3. **Monitor and optimize**
   - Track query patterns
   - Adjust thresholds based on real usage
   - Consider Phase 3 enhancements if needed

---

## 📝 Next Steps

To implement Phase 1 (Quick Win):

```bash
# 1. Test mini model performance
python -c "from utils.rag_database_openai import ArtworkRAGOpenAI;
rag = ArtworkRAGOpenAI();
# Add _analyze_with_mini_model method
"

# 2. Update search_exact_match to use mini model

# 3. Test performance
python test_performance.py

# 4. Deploy if results look good
```

To implement Phase 2 (Perceptual Hashing):

```bash
# 1. Install dependencies
pip install imagehash

# 2. Create utils/image_similarity.py

# 3. Build initial index
python -c "from utils.image_similarity import index_all_artworks; index_all_artworks()"

# 4. Update analyze_with_rag.py to check hash first

# 5. Test end-to-end
python test_performance.py
```

---

**Recommendation:** Implement Phase 1 first (2 hours work, immediate 3x improvement), then Phase 2 if you need sub-3-second responses for known artworks.

**Last Updated:** 2025-12-02
