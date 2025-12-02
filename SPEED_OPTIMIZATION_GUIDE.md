# Speed Optimization Guide for Museum Guide App

## Current Performance Bottlenecks

### Timing Analysis (per image analysis):
1. **RAG Database Initialization**: ~2 seconds (first time only)
2. **Image Analysis (OpenAI Vision)**: 2-5 seconds
3. **RAG Search**: 1-3 seconds (if checking RAG)
4. **TTS Generation (Metadata)**: 2-3 seconds
5. **TTS Generation (Description)**: 3-5 seconds

**Total**: ~10-18 seconds per image

---

## Optimization Strategies

### ⚡ **1. Pre-initialize RAG Database at Startup**
**Impact**: High (saves 2 seconds on first request)
**Complexity**: Easy
**Implementation**:

```python
# Add to app_gradio.py after imports
from utils.analyze_with_rag import get_rag_instance

# Pre-load RAG database
print("🔄 Pre-loading RAG database...")
get_rag_instance()
print("✅ RAG database ready")
```

---

### ⚡ **2. Parallel TTS Generation**
**Impact**: High (saves 2-4 seconds)
**Complexity**: Easy
**Implementation**:

```python
import concurrent.futures

# In analyze_image() function:
def analyze_image(image):
    # ... existing code up to TTS generation ...

    # Generate unique session ID
    session_id = str(uuid.uuid4())[:8]

    # Prepare metadata text
    source_text = " from our Special Exhibition" if from_rag else ""
    metadata_text = f"""
Artist: {metadata.get('artist', 'Unknown')}{source_text}.
Title: {metadata.get('title', 'Unknown')}.
Year: {metadata.get('year', 'Unknown')}.
Period: {metadata.get('period', 'Unknown')}.
"""

    # Generate both audio files in PARALLEL
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_metadata = executor.submit(text_to_speech, metadata_text, 60)
        future_description = executor.submit(text_to_speech, description, 60)

        # Wait for both to complete
        metadata_audio = future_metadata.result()
        description_audio = future_description.result()

    # Save audio files
    description_audio_path = None
    if description_audio:
        audio_path = OUTPUT_DIR / f"description_{session_id}.mp3"
        with open(audio_path, "wb") as f:
            f.write(description_audio)
        description_audio_path = str(audio_path)

    metadata_audio_path = None
    if metadata_audio:
        audio_path = OUTPUT_DIR / f"metadata_{session_id}.mp3"
        with open(audio_path, "wb") as f:
            f.write(metadata_audio)
        metadata_audio_path = str(audio_path)

    return description, metadata, description_audio_path, metadata_audio_path, status_message
```

---

### ⚡ **3. Image Resolution Reduction**
**Impact**: Medium (saves 0.5-1 second)
**Complexity**: Easy
**Implementation**:

```python
def analyze_image(image):
    if image is None:
        return None, None, None, None, "Please upload an image first."

    try:
        # Resize large images before processing
        max_size = 1024
        if max(image.size) > max_size:
            print(f"Resizing image from {image.size} to fit {max_size}px")
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()

        # Continue with existing code...
```

---

### ⚡ **4. Image Hash Caching**
**Impact**: Very High for repeated images (near-instant)
**Complexity**: Moderate
**Implementation**:

```python
import hashlib

# Global cache dictionary
image_analysis_cache = {}

def analyze_image(image):
    if image is None:
        return None, None, None, None, "Please upload an image first."

    try:
        # Generate image hash for caching
        img_bytes = image.tobytes()
        img_hash = hashlib.md5(img_bytes).hexdigest()

        # Check cache
        if img_hash in image_analysis_cache:
            print(f"✓ Using cached result for image {img_hash[:8]}")
            cached = image_analysis_cache[img_hash]
            return (
                cached['description'],
                cached['metadata'],
                cached['description_audio_path'],
                cached['metadata_audio_path'],
                cached['status'] + " (from cache)"
            )

        # ... existing analysis code ...

        # Store result in cache before returning
        image_analysis_cache[img_hash] = {
            'description': description,
            'metadata': metadata,
            'description_audio_path': description_audio_path,
            'metadata_audio_path': metadata_audio_path,
            'status': status_message
        }

        return description, metadata, description_audio_path, metadata_audio_path, status_message
```

---

### ⚡ **5. Pre-generate Audio for RAG Database Images**
**Impact**: Very High for RAG images (instant playback)
**Complexity**: Moderate
**Implementation**:

Create a new script `pregenerate_rag_audio.py`:

```python
"""Pre-generate audio files for all RAG database artworks"""

from pathlib import Path
from PIL import Image
import hashlib
from utils.analyze_with_rag import analyze_artwork_with_rag_fallback
from utils.audio import text_to_speech

RAG_AUDIO_CACHE_DIR = Path("rag_audio_cache")
RAG_AUDIO_CACHE_DIR.mkdir(exist_ok=True)

def pregenerate_all_rag_audio():
    """Pre-generate audio for all RAG database images"""
    rag_folder = Path("RAG_database")

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(rag_folder.glob(f'*{ext}'))
        image_files.extend(rag_folder.glob(f'*{ext.upper()}'))

    print(f"Pre-generating audio for {len(image_files)} RAG images...")

    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")

        # Load image
        img = Image.open(img_path)

        # Generate hash
        img_hash = hashlib.md5(img.tobytes()).hexdigest()

        # Check if already cached
        desc_cache = RAG_AUDIO_CACHE_DIR / f"description_{img_hash}.mp3"
        meta_cache = RAG_AUDIO_CACHE_DIR / f"metadata_{img_hash}.mp3"

        if desc_cache.exists() and meta_cache.exists():
            print(f"  ✓ Already cached")
            continue

        # Analyze artwork
        description, metadata, from_rag = analyze_artwork_with_rag_fallback(img)

        if not from_rag:
            print(f"  ⚠️  Not in RAG database, skipping")
            continue

        # Generate audio
        print(f"  Generating audio...")
        desc_audio = text_to_speech(description, timeout=60)

        metadata_text = f"""
Artist: {metadata.get('artist', 'Unknown')} from our Special Exhibition.
Title: {metadata.get('title', 'Unknown')}.
Year: {metadata.get('year', 'Unknown')}.
Period: {metadata.get('period', 'Unknown')}.
"""
        meta_audio = text_to_speech(metadata_text, timeout=60)

        # Save to cache
        if desc_audio:
            with open(desc_cache, "wb") as f:
                f.write(desc_audio)
            print(f"  ✓ Description audio cached")

        if meta_audio:
            with open(meta_cache, "wb") as f:
                f.write(meta_audio)
            print(f"  ✓ Metadata audio cached")

    print(f"\n✓ Pre-generation complete!")

if __name__ == "__main__":
    pregenerate_all_rag_audio()
```

Then modify `analyze_image()` to use pre-generated audio:

```python
def analyze_image(image):
    # ... existing code ...

    # Check if pre-generated audio exists (for RAG images)
    img_hash = hashlib.md5(image.tobytes()).hexdigest()
    rag_audio_dir = Path("rag_audio_cache")

    desc_cache = rag_audio_dir / f"description_{img_hash}.mp3"
    meta_cache = rag_audio_dir / f"metadata_{img_hash}.mp3"

    if from_rag and desc_cache.exists() and meta_cache.exists():
        print("✓ Using pre-generated audio from cache")
        return (
            description,
            metadata,
            str(desc_cache),
            str(meta_cache),
            status_message + " (instant audio)"
        )

    # Otherwise generate audio as usual...
```

---

### ⚡ **6. Progress Indicators**
**Impact**: UX improvement (no speed gain, but feels faster)
**Complexity**: Easy
**Implementation**:

```python
def analyze_image(image):
    # Use Gradio's progress indicator
    import gradio as gr

    with gr.Progress() as progress:
        progress(0, desc="Checking RAG database...")
        # RAG check

        progress(0.3, desc="Analyzing artwork with AI...")
        # Vision API

        progress(0.6, desc="Generating audio description...")
        # TTS

        progress(1.0, desc="Complete!")
```

---

## Recommended Implementation Order

### Phase 1: Quick Wins (< 30 minutes)
1. ✅ Pre-initialize RAG database
2. ✅ Parallel TTS generation
3. ✅ Image resolution reduction

**Expected improvement**: 3-5 seconds faster

### Phase 2: Caching (1 hour)
4. ✅ Image hash caching
5. ✅ Pre-generate RAG audio

**Expected improvement**: Near-instant for repeated/RAG images

### Phase 3: Polish (optional)
6. ✅ Progress indicators
7. ✅ Error handling improvements

---

## Performance Targets

### Current:
- First image: ~12-18 seconds
- Subsequent images: ~10-15 seconds
- RAG images: ~10-15 seconds

### After Phase 1:
- First image: ~8-12 seconds
- Subsequent images: ~6-10 seconds
- RAG images: ~6-10 seconds

### After Phase 2:
- First image: ~8-12 seconds
- Repeated images: <1 second (cache)
- RAG images: <1 second (pre-generated audio)

---

## Testing Performance

Add timing measurements:

```python
import time

def analyze_image(image):
    start_time = time.time()

    # ... existing code ...

    elapsed = time.time() - start_time
    print(f"⏱️  Total analysis time: {elapsed:.2f} seconds")

    return ...
```
