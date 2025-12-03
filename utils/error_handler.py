"""
Centralized Error Handling and Validation System

Provides robust error handling, retry logic, and validation
for all API calls and data processing operations.
"""

import time
import functools
from typing import Callable, Any, Optional, Dict
import logging
from PIL import Image
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Base exception for pipeline errors"""
    pass


class ValidationError(PipelineError):
    """Raised when input validation fails"""
    pass


class APIError(PipelineError):
    """Raised when API calls fail"""
    pass


class ProcessingError(PipelineError):
    """Raised when data processing fails"""
    pass


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {e}")
            
            raise APIError(f"{func.__name__} failed after {max_retries} attempts") from last_exception
        
        return wrapper
    return decorator


def validate_image(image_data: Any, max_size_mb: int = 10) -> Image.Image:
    """
    Validate and sanitize image input
    
    Args:
        image_data: PIL Image or bytes
        max_size_mb: Maximum allowed file size in MB
    
    Returns:
        Validated PIL Image
    
    Raises:
        ValidationError: If image is invalid
    """
    try:
        # Convert to PIL Image if bytes
        if isinstance(image_data, bytes):
            if len(image_data) > max_size_mb * 1024 * 1024:
                raise ValidationError(f"Image size exceeds {max_size_mb}MB limit")
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            raise ValidationError(f"Invalid image type: {type(image_data)}")
        
        # Verify it's a valid image
        image.verify()
        
        # Reload image after verify (verify closes the file)
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        
        # Check dimensions
        if image.size[0] < 50 or image.size[1] < 50:
            raise ValidationError(f"Image too small: {image.size}. Minimum 50x50 pixels required")
        
        if image.size[0] > 10000 or image.size[1] > 10000:
            raise ValidationError(f"Image too large: {image.size}. Maximum 10000x10000 pixels")
        
        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if image.mode not in ('RGB', 'L'):
            logger.info(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        logger.info(f"Image validated: {image.size}, mode={image.mode}")
        return image
        
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Image validation failed: {str(e)}")


def validate_text(text: Optional[str], field_name: str = "text", 
                 min_length: int = 1, max_length: int = 10000) -> str:
    """
    Validate and sanitize text input
    
    Args:
        text: Input text
        field_name: Name of the field for error messages
        min_length: Minimum allowed length
        max_length: Maximum allowed length
    
    Returns:
        Sanitized text
    
    Raises:
        ValidationError: If text is invalid
    """
    if text is None:
        raise ValidationError(f"{field_name} cannot be None")
    
    if not isinstance(text, str):
        raise ValidationError(f"{field_name} must be a string, got {type(text)}")
    
    # Strip whitespace
    text = text.strip()
    
    if len(text) < min_length:
        raise ValidationError(f"{field_name} too short: {len(text)} chars (minimum {min_length})")
    
    if len(text) > max_length:
        raise ValidationError(f"{field_name} too long: {len(text)} chars (maximum {max_length})")
    
    # Remove null bytes and other problematic characters
    text = text.replace('\x00', '')
    
    return text


def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize metadata dictionary
    
    Args:
        metadata: Metadata dictionary
    
    Returns:
        Validated metadata
    
    Raises:
        ValidationError: If metadata is invalid
    """
    if not isinstance(metadata, dict):
        raise ValidationError(f"Metadata must be a dictionary, got {type(metadata)}")
    
    required_fields = ['artist', 'title', 'year', 'period']
    validated = {}
    
    for field in required_fields:
        value = metadata.get(field, 'Unknown')
        if value is None or value == '':
            value = 'Unknown'
        validated[field] = str(value).strip()[:200]  # Limit length
    
    # Optional confidence field
    if 'confidence' in metadata:
        confidence = metadata['confidence']
        if confidence not in ['high', 'medium', 'low']:
            validated['confidence'] = 'low'
        else:
            validated['confidence'] = confidence
    else:
        validated['confidence'] = 'low'
    
    logger.info(f"Metadata validated: {validated}")
    return validated


def safe_api_call(func: Callable, *args, **kwargs) -> Optional[Any]:
    """
    Safely execute an API call with error handling
    
    Args:
        func: Function to call
        *args, **kwargs: Arguments to pass to function
    
    Returns:
        Function result or None on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"API call {func.__name__} failed: {e}")
        return None


def handle_pipeline_error(error: Exception, context: str = "") -> Dict[str, Any]:
    """
    Handle pipeline errors and return user-friendly error information
    
    Args:
        error: Exception that occurred
        context: Additional context about where error occurred
    
    Returns:
        Error information dictionary
    """
    error_type = type(error).__name__
    error_message = str(error)
    
    logger.error(f"Pipeline error in {context}: {error_type} - {error_message}")
    
    # Determine user-friendly message
    if isinstance(error, ValidationError):
        user_message = f"Input validation failed: {error_message}"
        recoverable = False
    elif isinstance(error, APIError):
        user_message = "Unable to connect to AI services. Please check your internet connection and try again."
        recoverable = True
    elif isinstance(error, ProcessingError):
        user_message = "Error processing your request. Please try again with a different image."
        recoverable = True
    else:
        user_message = "An unexpected error occurred. Please try again."
        recoverable = True
    
    return {
        'error_type': error_type,
        'error_message': error_message,
        'user_message': user_message,
        'recoverable': recoverable,
        'context': context
    }


class ProgressTracker:
    """Track progress through the pipeline for better error context"""
    
    def __init__(self):
        self.steps = []
        self.current_step = None
        self.start_time = time.time()
    
    def start_step(self, step_name: str):
        """Start tracking a new step"""
        self.current_step = {
            'name': step_name,
            'start_time': time.time(),
            'status': 'in_progress'
        }
        logger.info(f"Starting step: {step_name}")
    
    def complete_step(self, success: bool = True, error: Optional[str] = None):
        """Mark current step as complete"""
        if self.current_step:
            self.current_step['end_time'] = time.time()
            self.current_step['duration'] = self.current_step['end_time'] - self.current_step['start_time']
            self.current_step['status'] = 'success' if success else 'failed'
            self.current_step['error'] = error
            self.steps.append(self.current_step)
            
            logger.info(
                f"Completed step: {self.current_step['name']} "
                f"({self.current_step['duration']:.2f}s) - {self.current_step['status']}"
            )
            self.current_step = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline execution summary"""
        total_time = time.time() - self.start_time
        return {
            'total_time': total_time,
            'steps_completed': len([s for s in self.steps if s['status'] == 'success']),
            'steps_failed': len([s for s in self.steps if s['status'] == 'failed']),
            'steps': self.steps
        }
