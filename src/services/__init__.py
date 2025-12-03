"""
External Services Module

Integrations with external services:
    - vision: OpenAI Vision API for image analysis
    - audio: OpenAI TTS for audio generation
    - chat: OpenAI Chat API for conversational Q&A
    - rag_database: Pinecone RAG database integration
    - image_similarity: Perceptual hashing for image matching

Usage:
    from src.services.vision import generate_description, extract_metadata
    from src.services.audio import text_to_speech
    from src.services.chat import chat_with_artwork
    from src.services.rag_database import ArtworkRAGOpenAI
    from src.services.image_similarity import ImageSimilarityIndex
"""
