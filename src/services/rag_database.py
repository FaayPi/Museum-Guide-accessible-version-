"""RAG Vector Database for Museum Artworks using Pinecone and OpenAI Embeddings"""

import os
from pinecone import Pinecone, ServerlessSpec
from PIL import Image
import io
import base64
from pathlib import Path
from openai import OpenAI
from src.services.vision import generate_description
import hashlib
from dotenv import load_dotenv
import config

# Load environment variables
load_dotenv()


class ArtworkRAGOpenAI:
    """Vector database for artwork images using Pinecone and OpenAI vision + text embeddings"""

    def __init__(self, database_path="data/RAG_database", index_name="museum-artworks"):
        self.database_path = database_path
        self.index_name = index_name

        # Get Pinecone API key from environment
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables. Please add it to .env file.")

        # Initialize Pinecone
        print("Initializing Pinecone...")
        self.pc = Pinecone(api_key=api_key)

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

        # OpenAI text-embedding-3-large has 3072 dimensions
        self.embedding_dim = 1536  # text-embedding-3-small
        print("✓ OpenAI client initialized")

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
        """Load and preprocess image"""
        img = Image.open(image_path).convert('RGB')
        # Resize to reasonable size if too large
        max_size = 512
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return img

    def _generate_text_embedding(self, text):
        """Generate OpenAI text embedding"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000]  # Limit text length
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def _generate_title_from_description(self, description):
        """Generate a fitting title based on the artwork description"""
        try:
            response = self.openai_client.chat.completions.create(
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
        description = generate_description(image_bytes)

        if not description:
            print(f"ERROR: Failed to analyze artwork {image_path}")
            return None

        # Generate title from description
        print("Generating title...")
        title = self._generate_title_from_description(description)

        # Generate text embedding from description
        print("Generating embedding from description...")
        embedding = self._generate_text_embedding(description)

        if not embedding:
            print(f"ERROR: Failed to generate embedding for {image_path}")
            return None

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

    def search_exact_match(self, query_image_bytes):
        """
        Search for exact image match in the database

        This is used as a fallback when OpenAI Vision cannot identify the artwork.
        It analyzes the query image and searches for an exact match in the RAG database.

        Args:
            query_image_bytes: Image bytes to search for

        Returns:
            dict or None: Exact matched artwork with full description, or None if not found
        """
        print("\n=== RAG Fallback: Searching for exact match ===")

        # Get description of query image using FAST mini model for RAG matching
        print("Analyzing query image with gpt-4o-mini (fast RAG search)...")
        query_description = self._analyze_with_mini_model(query_image_bytes)

        if not query_description:
            print("Failed to analyze query image")
            return None

        # Generate embedding for query description
        query_embedding = self._generate_text_embedding(query_description)

        if not query_embedding:
            print("Failed to generate embedding")
            return None

        # Search in Pinecone with high similarity threshold for exact matches
        results = self.index.query(
            vector=query_embedding,
            top_k=1,
            include_metadata=True
        )

        # Check if we have a very high similarity match (>=85 indicates likely exact match)
        if results['matches'] and len(results['matches']) > 0:
            top_match = results['matches'][0]
            similarity = top_match['score']

            print(f"Top match similarity: {similarity:.3f}")

            # Threshold for exact match - adjusted to 0.85 based on testing
            if similarity >= 0.85:
                print(f"✓ Exact match found: {top_match['metadata']['title']}")
                print(f"  Artist: {top_match['metadata']['artist']}")
                print(f"  Year: {top_match['metadata']['year']}")
                return {
                    'metadata': top_match['metadata'],
                    'description': top_match['metadata']['description'],  # Full description from metadata
                    'similarity_score': similarity,
                    'is_exact_match': True
                }
            else:
                print(f"No exact match found (similarity: {similarity:.3f} < 0.85)")

        return None

    def _analyze_with_mini_model(self, image_bytes):
        """
        Fast artwork analysis using GPT-4o-mini for RAG matching.
        This is 5-8x faster than full analysis and sufficient for similarity search.

        Args:
            image_bytes: Image data as bytes

        Returns:
            str: Brief description of the artwork
        """
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
                            "text": "Describe this artwork in 2-3 sentences focusing on: subject, colors, style, mood. Be concise but specific."
                        }
                    ]
                }],
                max_tokens=150  # Short description for fast matching
            )
            description = response.choices[0].message.content
            print(f"✓ Quick description generated ({len(description)} chars)")
            return description
        except Exception as e:
            print(f"Error with mini model: {e}")
            return None

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
    print("INITIALIZING RAG VECTOR DATABASE (PINECONE + OPENAI)")
    print("="*60)

    try:
        rag = ArtworkRAGOpenAI()
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
