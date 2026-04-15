#!/usr/bin/env python3
"""
Re-index all artworks in RAG database with updated description limit (2000 chars)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from src.services.rag_database import ArtworkRAGOpenAI

def main():
    print("\n" + "="*70)
    print("RE-INDEXING RAG DATABASE WITH UPDATED DESCRIPTION LIMIT (2000 chars)")
    print("="*70)
    
    try:
        # Initialize RAG database
        rag = ArtworkRAGOpenAI()
        
        # Get current stats
        stats = rag.get_collection_stats()
        print(f"\nCurrent database statistics:")
        print(f"  Total artworks: {stats['total_artworks']}")
        print(f"  Index name: {stats['index_name']}")
        print(f"  Embedding dimension: {stats['dimension']}")
        
        # Re-index all artworks
        print("\n" + "="*70)
        print("Starting re-indexing process...")
        print("="*70)
        
        indexed_artworks = rag.index_all_artworks()
        
        print("\n" + "="*70)
        print(f"RE-INDEXING COMPLETE!")
        print(f"Successfully indexed {len(indexed_artworks)} artworks")
        print("="*70)
        
        # Show final stats
        final_stats = rag.get_collection_stats()
        print(f"\nFinal database statistics:")
        print(f"  Total artworks: {final_stats['total_artworks']}")
        print(f"  All descriptions now have up to 2000 characters")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

