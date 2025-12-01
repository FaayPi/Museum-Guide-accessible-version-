"""RAG Vector Database for Museum Artworks"""

import os
import chromadb
from chromadb.config import Settings
from PIL import Image
import io
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from utils.vision import analyze_artwork, get_metadata
import hashlib


class ArtworkRAG:
    """Vector database for artwork images using ChromaDB and CLIP embeddings"""

    def __init__(self, database_path="RAG_database", persist_directory="./chroma_db"):
        self.database_path = database_path
        self.persist_directory = persist_directory

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="museum_artworks",
            metadata={"description": "Museum artwork images with descriptions"}
        )

        # Initialize CLIP model for image embeddings
        print("Loading CLIP model for image embeddings...")
        self.clip_model = SentenceTransformer('clip-ViT-B-32')
        print("CLIP model loaded successfully!")

    def _get_image_hash(self, image_path):
        """Generate a unique hash for an image file"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _load_and_preprocess_image(self, image_path):
        """Load and preprocess image for embedding"""
        img = Image.open(image_path).convert('RGB')
        # Resize to reasonable size if too large
        max_size = 512
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return img

    def _generate_image_embedding(self, image):
        """Generate CLIP embedding for an image"""
        embedding = self.clip_model.encode(image, convert_to_numpy=True)
        return embedding.tolist()

    def _generate_title_from_description(self, description):
        """Generate a fitting title based on the artwork description"""
        from openai import OpenAI
        import config

        try:
            client = OpenAI(api_key=config.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": f"""Based on this artwork description, generate a short, poetic title (2-4 words) that captures the essence of the work. Return ONLY the title, nothing else.

Description: {description[:300]}

Title:"""
                }],
                max_tokens=20,
                temperature=0.7
            )
            title = response.choices[0].message.content.strip().strip('"').strip("'")
            return title
        except Exception as e:
            print(f"Error generating title: {e}")
            # Fallback to filename-based title
            return "Untitled"

    def index_artwork(self, image_path):
        """
        Index a single artwork image with metadata and description

        Args:
            image_path: Path to the artwork image

        Returns:
            dict: Indexed artwork information
        """
        print(f"\n=== Indexing artwork: {image_path} ===")

        # Generate unique ID based on file hash
        image_hash = self._get_image_hash(image_path)
        doc_id = f"artwork_{image_hash}"

        # Check if already indexed
        existing = self.collection.get(ids=[doc_id])
        if existing['ids']:
            print(f"Artwork already indexed: {Path(image_path).name}")
            return existing['metadatas'][0]

        # Load image
        img = self._load_and_preprocess_image(image_path)

        # Convert to bytes for vision API
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()

        # Get description from OpenAI Vision
        print("Analyzing artwork with GPT-4 Vision...")
        description = analyze_artwork(image_bytes)

        if not description:
            print(f"ERROR: Failed to analyze artwork {image_path}")
            return None

        # Generate title from description
        print("Generating title...")
        title = self._generate_title_from_description(description)

        # Generate CLIP embedding
        print("Generating CLIP embedding...")
        embedding = self._generate_image_embedding(img)

        # Prepare metadata - Fixed values for RAG database artworks
        artwork_metadata = {
            "filename": Path(image_path).name,
            "filepath": str(image_path),
            "artist": "Fee Pieper",  # Fixed artist
            "title": title,  # Generated from description
            "year": "2000",  # Fixed year
            "period": "Contemporary Art",  # Fixed period
            "confidence": "high",  # High confidence since we know the artist
            "description": description[:500]  # Store first 500 chars in metadata
        }

        # Add to ChromaDB collection
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[description],  # Full description as document
            metadatas=[artwork_metadata]
        )

        print(f"✓ Successfully indexed: {artwork_metadata['filename']}")
        print(f"  Artist: {artwork_metadata['artist']}")
        print(f"  Title: {artwork_metadata['title']}")
        print(f"  Year: {artwork_metadata['year']}")

        return artwork_metadata

    def index_all_artworks(self):
        """Index all artworks in the RAG_database folder"""
        database_path = Path(self.database_path)

        if not database_path.exists():
            print(f"ERROR: Database path does not exist: {self.database_path}")
            return []

        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(database_path.glob(f'*{ext}'))
            image_files.extend(database_path.glob(f'*{ext.upper()}'))

        print(f"\n=== Found {len(image_files)} images to index ===")

        indexed_artworks = []
        for i, image_path in enumerate(image_files, 1):
            print(f"\nProcessing {i}/{len(image_files)}: {image_path.name}")
            try:
                artwork_info = self.index_artwork(str(image_path))
                if artwork_info:
                    indexed_artworks.append(artwork_info)
            except Exception as e:
                print(f"ERROR indexing {image_path.name}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n=== Indexing complete! {len(indexed_artworks)} artworks indexed ===")
        return indexed_artworks

    def search_similar_artworks(self, query_image, n_results=3):
        """
        Search for similar artworks using image similarity

        Args:
            query_image: PIL Image or bytes
            n_results: Number of similar artworks to return

        Returns:
            list: Similar artworks with metadata
        """
        # Convert bytes to PIL Image if needed
        if isinstance(query_image, bytes):
            query_image = Image.open(io.BytesIO(query_image)).convert('RGB')

        # Generate embedding for query image
        query_embedding = self._generate_image_embedding(query_image)

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # Format results
        similar_artworks = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                artwork = {
                    'id': results['ids'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'description': results['documents'][0][i],
                    'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
                }
                similar_artworks.append(artwork)

        return similar_artworks

    def get_artwork_by_filename(self, filename):
        """Retrieve artwork information by filename"""
        results = self.collection.get(
            where={"filename": filename}
        )

        if results['ids']:
            return {
                'id': results['ids'][0],
                'metadata': results['metadatas'][0],
                'description': results['documents'][0]
            }
        return None

    def get_collection_stats(self):
        """Get statistics about the indexed collection"""
        count = self.collection.count()
        return {
            'total_artworks': count,
            'collection_name': self.collection.name
        }


def initialize_rag_database():
    """Initialize and index all artworks in the RAG database"""
    print("\n" + "="*60)
    print("INITIALIZING RAG VECTOR DATABASE")
    print("="*60)

    rag = ArtworkRAG()

    # Check if already indexed
    stats = rag.get_collection_stats()
    if stats['total_artworks'] > 0:
        print(f"\nDatabase already initialized with {stats['total_artworks']} artworks")
        print("To re-index, delete the ./chroma_db folder")
        return rag

    # Index all artworks
    indexed = rag.index_all_artworks()

    print("\n" + "="*60)
    print(f"RAG DATABASE READY: {len(indexed)} artworks indexed")
    print("="*60 + "\n")

    return rag


if __name__ == "__main__":
    # Test the RAG database
    rag = initialize_rag_database()

    # Show stats
    stats = rag.get_collection_stats()
    print(f"\nDatabase Statistics:")
    print(f"  Total artworks: {stats['total_artworks']}")
