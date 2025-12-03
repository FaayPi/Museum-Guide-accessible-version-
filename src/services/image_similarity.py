"""Fast image similarity search using perceptual hashing"""

import imagehash
from PIL import Image
import io
from pathlib import Path
import json

class ImageSimilarityIndex:
    """Lightning-fast image matching using perceptual hashing"""

    def __init__(self, index_file="data/image_hash_index.json"):
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
            json.dump(self.hash_index, f, indent=2)

    def add_image(self, image_path, metadata, description):
        """Add image to similarity index with full metadata and description"""
        img = Image.open(image_path)

        # Generate multiple hashes for robustness
        phash = str(imagehash.phash(img, hash_size=16))
        dhash = str(imagehash.dhash(img, hash_size=16))
        ahash = str(imagehash.average_hash(img, hash_size=16))

        image_id = Path(image_path).stem

        self.hash_index[image_id] = {
            'phash': phash,
            'dhash': dhash,
            'ahash': ahash,
            'metadata': metadata,
            'description': description,  # Store full description!
            'filepath': str(image_path)
        }

        self._save_index()
        print(f"✓ Added to similarity index: {image_id}")

    def find_match(self, image_bytes, threshold=10):
        """
        Find matching image using perceptual hashing

        Args:
            image_bytes: Query image as bytes
            threshold: Hamming distance threshold (lower = stricter, default 10)

        Returns:
            dict or None: Matched image with metadata and description if found
        """
        img = Image.open(io.BytesIO(image_bytes))

        # Generate query hashes
        query_phash = imagehash.phash(img, hash_size=16)
        query_dhash = imagehash.dhash(img, hash_size=16)
        query_ahash = imagehash.average_hash(img, hash_size=16)

        best_match = None
        best_distance = float('inf')

        # Compare against all indexed images
        for image_id, data in self.hash_index.items():
            try:
                stored_phash = imagehash.hex_to_hash(data['phash'])
                stored_dhash = imagehash.hex_to_hash(data['dhash'])
                stored_ahash = imagehash.hex_to_hash(data['ahash'])

                # Calculate combined distance (average of all hash types)
                phash_dist = query_phash - stored_phash
                dhash_dist = query_dhash - stored_dhash
                ahash_dist = query_ahash - stored_ahash
                combined_dist = (phash_dist + dhash_dist + ahash_dist) / 3

                if combined_dist < best_distance:
                    best_distance = combined_dist
                    best_match = {
                        'image_id': image_id,
                        'metadata': data['metadata'],
                        'description': data['description'],  # Full description!
                        'distance': combined_dist,
                        'filepath': data['filepath']
                    }
            except Exception as e:
                print(f"Warning: Error comparing with {image_id}: {e}")
                continue

        # Check if match is close enough
        if best_match and best_match['distance'] < threshold:
            print(f"✓ Found similar image: {best_match['image_id']} (distance: {best_match['distance']:.1f})")
            return best_match
        else:
            if best_match:
                print(f"No similar image found (best distance: {best_distance:.1f} > threshold: {threshold})")
            else:
                print("No similar images in index")
            return None

def build_similarity_index_from_rag():
    """Build similarity index from existing RAG database"""
    from utils.rag_database_openai import ArtworkRAGOpenAI

    print("\n" + "="*70)
    print("BUILDING IMAGE SIMILARITY INDEX FROM RAG DATABASE")
    print("="*70)

    try:
        similarity_index = ImageSimilarityIndex()
        rag = ArtworkRAGOpenAI()

        # Get stats
        stats = rag.get_collection_stats()
        print(f"\nRAG database contains {stats['total_artworks']} artworks")

        # Find all images in RAG_database folder
        rag_folder = Path("RAG_database")
        if not rag_folder.exists():
            print(f"ERROR: RAG_database folder not found")
            return None

        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []

        for ext in image_extensions:
            image_files.extend(rag_folder.glob(f'*{ext}'))
            image_files.extend(rag_folder.glob(f'*{ext.upper()}'))

        print(f"Found {len(image_files)} images to index")

        # For each image, fetch its metadata and description from Pinecone
        for img_path in image_files:
            try:
                # Generate hash for this image to look it up in Pinecone
                img_hash = rag._get_image_hash(str(img_path))
                doc_id = f"artwork_{img_hash}"

                # Fetch from Pinecone
                result = rag.index.fetch(ids=[doc_id])

                if result['vectors'] and doc_id in result['vectors']:
                    vector_data = result['vectors'][doc_id]
                    metadata = vector_data['metadata']
                    description = metadata.get('description', '')

                    # Add to similarity index
                    similarity_index.add_image(str(img_path), metadata, description)
                else:
                    print(f"  Skipping {img_path.name} - not found in RAG database")

            except Exception as e:
                print(f"  Error indexing {img_path.name}: {e}")
                continue

        print(f"\n✓ Similarity index built: {len(similarity_index.hash_index)} images")
        print(f"✓ Index saved to: {similarity_index.index_file}")
        print("="*70 + "\n")

        return similarity_index

    except Exception as e:
        print(f"ERROR building similarity index: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    build_similarity_index_from_rag()
