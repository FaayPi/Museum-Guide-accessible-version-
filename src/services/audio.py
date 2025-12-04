"""Audio functions for TTS and STT using OpenAI"""

from openai import OpenAI
import config
import io


def text_to_speech(text, timeout=30):
    """
    Convert text to speech using OpenAI TTS
    
    Args:
        text (str): Text to convert to speech
        timeout (int): Timeout in seconds (default: 30)
    
    Returns:
        bytes: Audio data in MP3 format, or None on error
    """
    try:
        if not text or len(text.strip()) == 0:
            return None

        client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=timeout)

        response = client.audio.speech.create(
            model=config.TTS_MODEL,
            voice=config.TTS_VOICE,
            input=text
        )

        # Return audio bytes
        audio_bytes = response.content
        return audio_bytes

    except Exception as e:
        print(f"ERROR: TTS failed - {str(e)}")
        return None


def speech_to_text(audio_file, language="en"):
    """
    Convert speech to text using OpenAI Whisper
    
    Args:
        audio_file: Audio file (bytes or file-like object)
        language: Language code (default: "en" for English)
    
    Returns:
        str: Transcribed text
    """
    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)

        # If audio_file is bytes, convert to file-like object
        if isinstance(audio_file, bytes):
            audio_file = io.BytesIO(audio_file)
            audio_file.name = "audio.wav"  # Whisper needs a filename

        # Call Whisper API
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language
        )

        return transcript.text

    except Exception as e:
        print(f"ERROR: Speech recognition failed - {str(e)}")
        return None
