"""RAG Vector Database for Museum Artworks using Pinecone"""

import os
from pinecone import Pinecone, ServerlessSpec
from PIL import Image
import io
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from utils.vision import analyze_artwork
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ArtworkRAGPinecone:
    """Vector database for artwork images using Pinecone and CLIP embeddings"""

    def __init__(self, database_path="RAG_database", index_name="museum-artworks"):
        self.database_path = database_path
        self.index_name = index_name

        # Get Pinecone API key from environment
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables. Please add it to .env file.")

        # Initialize Pinecone
        print("Initializing Pinecone...")
        self.pc = Pinecone(api_key=api_key)

        # Initialize CLIP model for image embeddings
        print("Loading CLIP model for image embeddings...")
        self.clip_model = SentenceTransformer('clip-ViT-B-32')
        self.embedding_dim = 512  # CLIP ViT-B-32 dimension
        print("CLIP model loaded successfully!")

        # Create or connect to index
        self._setup_index()

    def _setup_index(self):
        """Create Pinecone index if it doesn't exist"""
        existing_indexes = [index.name for index in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"✓ Index '{self.index_name}' created")
        else:
            print(f"✓ Using existing index: {self.index_name}")

        # Connect to index
        self.index = self.pc.Index(self.index_name)

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
        try:
            existing = self.index.fetch(ids=[doc_id])
            if existing['vectors'] and doc_id in existing['vectors']:
                print(f"Artwork already indexed: {Path(image_path).name}")
                return existing['vectors'][doc_id]['metadata']
        except:
            pass  # Not indexed yet

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
            "description": description  # Full description
        }

        # Upsert to Pinecone
        self.index.upsert(
            vectors=[{
                "id": doc_id,
                "values": embedding,
                "metadata": artwork_metadata
            }]
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

        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=n_results,
            include_metadata=True
        )

        # Format results
        similar_artworks = []
        for match in results['matches']:
            artwork = {
                'id': match['id'],
                'metadata': match['metadata'],
                'similarity_score': match['score']
            }
            similar_artworks.append(artwork)

        return similar_artworks

    def get_collection_stats(self):
        """Get statistics about the indexed collection"""
        stats = self.index.describe_index_stats()
        return {
            'total_artworks': stats['total_vector_count'],
            'index_name': self.index_name,
            'dimension': stats['dimension']
        }


def initialize_rag_database():
    """Initialize and index all artworks in the RAG database"""
    print("\n" + "="*60)
    print("INITIALIZING RAG VECTOR DATABASE (PINECONE)")
    print("="*60)

    try:
        rag = ArtworkRAGPinecone()
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease add your Pinecone API key to the .env file:")
        print("PINECONE_API_KEY=your_api_key_here")
        return None

    # Check if already indexed
    stats = rag.get_collection_stats()
    if stats['total_artworks'] > 0:
        print(f"\n✓ Database already initialized with {stats['total_artworks']} artworks")
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

    if rag:
        # Show stats
        stats = rag.get_collection_stats()
        print(f"\nDatabase Statistics:")
        print(f"  Total artworks: {stats['total_artworks']}")
        print(f"  Index name: {stats['index_name']}")
        print(f"  Embedding dimension: {stats['dimension']}")
