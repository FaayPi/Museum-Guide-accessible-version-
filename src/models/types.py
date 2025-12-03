"""
Type definitions for the Museum Guide App.

This module provides type hints and data structures for type safety and code clarity.
Following PEP 484 (Type Hints) and PEP 526 (Variable Annotations).
"""

from typing import Dict, List, Optional, Tuple, Union, Any, TypedDict
from dataclasses import dataclass
from pathlib import Path
from enum import Enum


# ==================== ENUMS ====================

class ConfidenceLevel(str, Enum):
    """Confidence levels for metadata extraction."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ImpactLevel(str, Enum):
    """Impact levels for AI limitations."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class MessageRole(str, Enum):
    """Chat message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ==================== TYPE ALIASES ====================

ImageData = Union[bytes, Path, Any]  # PIL Image, bytes, or Path
ChatHistoryList = List[Dict[str, str]]  # List of {"role": str, "content": str}
MetadataDict = Dict[str, str]  # Artwork metadata dictionary


# ==================== TYPED DICTIONARIES ====================

class ArtworkMetadata(TypedDict, total=False):
    """
    Structured metadata for an artwork.

    Attributes:
        artist: Artist name or "Unknown"
        title: Artwork title or "Unknown"
        year: Year created or "Unknown"
        period: Art period/movement
        confidence: Confidence level of identification
    """
    artist: str
    title: str
    year: str
    period: str
    confidence: str


class ChatMessage(TypedDict):
    """
    Single chat message in conversation.

    Attributes:
        role: Message role (user, assistant, system)
        content: Message content
    """
    role: str
    content: str


class AnalysisResult(TypedDict):
    """
    Complete artwork analysis result.

    Attributes:
        description: Textual description of artwork
        metadata: Structured metadata
        from_rag: Whether data came from RAG database
    """
    description: str
    metadata: ArtworkMetadata
    from_rag: bool


class CacheEntry(TypedDict):
    """
    Cached analysis entry.

    Attributes:
        description: Artwork description
        metadata: Artwork metadata
        description_audio_path: Path to description audio file
        metadata_audio_path: Path to metadata audio file
        status: Status message
    """
    description: str
    metadata: ArtworkMetadata
    description_audio_path: Optional[str]
    metadata_audio_path: Optional[str]
    status: str


# ==================== DATACLASSES ====================

@dataclass(frozen=True)
class APIConfig:
    """
    Configuration for API calls.

    Attributes:
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff multiplier
    """
    timeout: float
    max_retries: int
    backoff_factor: float


@dataclass(frozen=True)
class TokenLimits:
    """
    Token limits for API calls.

    Attributes:
        description: Max tokens for description generation
        metadata: Max tokens for metadata extraction
        chat: Max tokens for chat responses
    """
    description: int
    metadata: int
    chat: int


@dataclass(frozen=True)
class OptimizationConfig:
    """
    Performance optimization configuration.

    Attributes:
        enable_parallel: Enable parallel API calls
        enable_caching: Enable result caching
        enable_rag: Enable RAG database lookup
        enable_precheck: Enable fast pre-check for generic images
        rag_timeout: Timeout for RAG search in seconds
        max_image_size: Maximum image dimension in pixels
    """
    enable_parallel: bool = True
    enable_caching: bool = True
    enable_rag: bool = True
    enable_precheck: bool = True
    rag_timeout: float = 2.5
    max_image_size: int = 384


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for a single operation.

    Attributes:
        operation: Operation name
        execution_time: Execution time in seconds
        success: Whether operation succeeded
        tokens_used: Number of tokens consumed
        error_message: Error message if failed
    """
    operation: str
    execution_time: float
    success: bool
    tokens_used: int
    error_message: str = ""


@dataclass
class RAGSearchResult:
    """
    Result from RAG database search.

    Attributes:
        description: Artwork description
        metadata: Artwork metadata
        similarity_score: Similarity score (0-1)
        source: Data source identifier
    """
    description: str
    metadata: ArtworkMetadata
    similarity_score: float
    source: str


@dataclass
class HashMatchResult:
    """
    Result from perceptual hash matching.

    Attributes:
        description: Artwork description
        metadata: Artwork metadata
        distance: Hash distance (lower = more similar)
        image_id: Matched image identifier
    """
    description: str
    metadata: ArtworkMetadata
    distance: float
    image_id: str


# ==================== CONSTANTS ====================

# API Configuration
DEFAULT_API_CONFIG = APIConfig(
    timeout=30.0,
    max_retries=3,
    backoff_factor=2.0
)

CHAT_API_CONFIG = APIConfig(
    timeout=20.0,
    max_retries=3,
    backoff_factor=2.0
)

# Token Limits (Optimized)
OPTIMIZED_TOKEN_LIMITS = TokenLimits(
    description=100,
    metadata=60,
    chat=150
)

# Default Optimization Configuration
DEFAULT_OPTIMIZATION_CONFIG = OptimizationConfig()

# Chat History Limits
MAX_CHAT_HISTORY_MESSAGES = 6  # Last 6 messages (3 exchanges)

# RAG Search Thresholds
RAG_SIMILARITY_THRESHOLD = 0.85  # Minimum similarity for match
HASH_DISTANCE_THRESHOLD = 10  # Maximum hash distance for match

# Generic Artwork Detection Thresholds
GENERIC_ARTWORK_THRESHOLDS = {
    'edge_density_min': 5,      # Minimum edge density
    'color_variance_min': 15,    # Minimum color variance
    'color_diversity_min': 0.1   # Minimum color diversity ratio
}

# Image Processing
MAX_IMAGE_DIMENSION = 384  # Pixels (optimized for speed)
IMAGE_QUALITY = 85  # JPEG quality (0-100)

# TTS Optimization
MAX_TTS_SENTENCES = 3  # Maximum sentences for TTS

# Error Messages
ERROR_MESSAGES = {
    'no_api_key': "OpenAI API key not configured",
    'empty_response': "Empty response from API",
    'invalid_image': "Invalid or corrupt image",
    'timeout': "Operation timed out",
    'encoding_failed': "Failed to encode image",
    'parsing_failed': "Failed to parse response"
}


# ==================== TYPE GUARDS ====================

def is_valid_metadata(data: Any) -> bool:
    """
    Type guard to check if data is valid ArtworkMetadata.

    Args:
        data: Data to validate

    Returns:
        True if data has all required metadata fields
    """
    if not isinstance(data, dict):
        return False

    required_fields = {'artist', 'title', 'year', 'period'}
    return all(field in data for field in required_fields)


def is_valid_chat_history(data: Any) -> bool:
    """
    Type guard to check if data is valid chat history.

    Args:
        data: Data to validate

    Returns:
        True if data is a list of valid chat messages
    """
    if not isinstance(data, list):
        return False

    return all(
        isinstance(msg, dict) and
        'role' in msg and
        'content' in msg
        for msg in data
    )


# ==================== HELPER FUNCTIONS ====================

def create_chat_message(role: MessageRole, content: str) -> ChatMessage:
    """
    Create a properly formatted chat message.

    Args:
        role: Message role
        content: Message content

    Returns:
        Formatted chat message
    """
    return ChatMessage(role=role.value, content=content)


def create_default_metadata(confidence: ConfidenceLevel = ConfidenceLevel.LOW) -> ArtworkMetadata:
    """
    Create default/fallback metadata.

    Args:
        confidence: Confidence level

    Returns:
        Default metadata with "Unknown" values
    """
    return ArtworkMetadata(
        artist="Unknown",
        title="Unknown",
        year="Unknown",
        period="Contemporary",
        confidence=confidence.value
    )
