"""
Museum Audio Guide - Gradio App
Main application with Audio-Guide and Visual-Guide modes
"""

import gradio as gr
import io
from pathlib import Path
import sys
import uuid
import time
import concurrent.futures
from PIL import Image

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import backend functions
from utils.vision import analyze_artwork, get_metadata
from utils.audio import text_to_speech, speech_to_text
from utils.chat import chat_with_artwork
from utils.analyze_with_rag import analyze_artwork_with_rag_fallback, format_metadata_text, get_rag_instance
import config

# Create audio_outputs directory
OUTPUT_DIR = Path("audio_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ⚡ SPEED OPTIMIZATION: Pre-initialize RAG database at startup
print("🔄 Pre-loading RAG database...")
startup_time = time.time()
get_rag_instance()
print(f"✅ RAG database ready ({time.time() - startup_time:.2f}s)")

# ⚡ OPTIMIZATION: Image analysis cache for repeated images
import hashlib
image_analysis_cache = {}

# Global session state (will be replaced by Gradio's state management)
session_data = {
    'image': None,
    'description': None,
    'metadata': None,
    'chat_history': [],
    'description_audio_path': None,
    'metadata_audio_path': None
}


def reset_session():
    """Reset session to initial state"""
    return {
        'image': None,
        'description': None,
        'metadata': None,
        'chat_history': [],
        'description_audio_path': None,
        'metadata_audio_path': None
    }


def analyze_image(image):
    """
    Analyze image and generate audio outputs with RAG fallback
    Returns: (description, metadata, description_audio, metadata_audio, status_message)

    ⚡ OPTIMIZED with:
    - Image resolution reduction (faster API calls)
    - Parallel TTS generation (2-4 seconds faster)
    - Performance timing
    """
    if image is None:
        return None, None, None, None, "Please upload an image first."

    try:
        start_time = time.time()

        # ⚡ OPTIMIZATION 1: Resize large images before processing - aggressive for speed
        max_size = 768  # Smaller = faster upload & processing
        if max(image.size) > max_size:
            print(f"⚡ Resizing image from {image.size} to fit {max_size}px")
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # ⚡ OPTIMIZATION 2: Check cache for repeated images
        img_bytes_for_hash = image.tobytes()
        img_hash = hashlib.md5(img_bytes_for_hash).hexdigest()

        if img_hash in image_analysis_cache:
            print(f"✓ Using cached result for image {img_hash[:8]}")
            cached = image_analysis_cache[img_hash]
            cache_time = time.time() - start_time
            print(f"⚡ Cache retrieval time: {cache_time:.2f}s")
            return (
                cached['description'],
                cached['metadata'],
                cached['description_audio_path'],
                cached['metadata_audio_path'],
                cached['status'] + " (from cache)"
            )

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()

        # Step 1: Analyze artwork with RAG fallback
        print("\n=== Starting artwork analysis with RAG fallback ===")
        description, metadata, from_rag = analyze_artwork_with_rag_fallback(image)

        if not description:
            return None, None, None, None, "Unable to analyze the artwork. Please try again."

        if not metadata:
            return None, None, None, None, "Unable to retrieve artwork information. Please check your connection and try again."

        # Add indicator if data came from RAG
        if from_rag:
            print("✓ Using data from RAG database (Special Exhibition)")
            status_message = "Analysis complete! This artwork is from our Special Exhibition."
        else:
            print("✓ Using data from OpenAI Vision")
            status_message = "Analysis complete! Audio ready to play."

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

        # ⚡ OPTIMIZATION 2: Generate both audio files in PARALLEL
        print("⚡ Generating audio files in parallel...")
        tts_start = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both TTS tasks simultaneously
            future_description = executor.submit(text_to_speech, description, 60)
            future_metadata = executor.submit(text_to_speech, metadata_text, 60)

            # Wait for both to complete
            description_audio = future_description.result()
            metadata_audio = future_metadata.result()

        print(f"⚡ TTS generation completed in {time.time() - tts_start:.2f}s (parallel)")

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

        # Performance timing
        total_time = time.time() - start_time
        print(f"⏱️  Total analysis time: {total_time:.2f}s")

        # ⚡ Store result in cache for future use
        image_analysis_cache[img_hash] = {
            'description': description,
            'metadata': metadata,
            'description_audio_path': description_audio_path,
            'metadata_audio_path': metadata_audio_path,
            'status': status_message
        }
        print(f"✓ Result cached for image {img_hash[:8]}")

        return description, metadata, description_audio_path, metadata_audio_path, status_message

    except Exception as e:
        print(f"ERROR in analyze_image: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, f"An error occurred while processing the artwork: {str(e)}"


def process_voice_question(audio_file, description, metadata, chat_history):
    """
    Process voice question and return audio answer
    Returns: (chat_history, answer_audio)
    """
    if audio_file is None:
        return chat_history, None

    if description is None or metadata is None:
        return chat_history, None

    try:
        # Read audio file
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()

        # Transcribe question
        question = speech_to_text(audio_bytes, language=None)
        if not question:
            return chat_history, None

        # Convert old-style chat history to new format for chat_with_artwork
        old_format_history = []
        for user_msg, bot_msg in chat_history:
            if user_msg:
                old_format_history.append({"role": "user", "content": user_msg})
            if bot_msg:
                old_format_history.append({"role": "assistant", "content": bot_msg})

        # Get answer from chatbot
        answer = chat_with_artwork(
            question=question,
            artwork_description=description,
            metadata=metadata,
            chat_history=old_format_history
        )

        # Update chat history (Gradio Chatbot format: list of tuples)
        chat_history.append((question, answer))

        # Generate audio answer
        answer_audio = text_to_speech(answer, timeout=60)
        answer_audio_path = None

        if answer_audio:
            answer_id = str(uuid.uuid4())[:8]
            audio_path = OUTPUT_DIR / f"answer_{answer_id}.mp3"
            with open(audio_path, "wb") as f:
                f.write(answer_audio)
            answer_audio_path = str(audio_path)

        return chat_history, answer_audio_path

    except Exception as e:
        print(f"Error processing voice question: {e}")
        return chat_history, None


def process_text_question(question, description, metadata, chat_history):
    """
    Process text question and return answer
    Returns: (updated_chat_history, clear_input)
    """
    if not question or question.strip() == "":
        return chat_history, ""

    if description is None or metadata is None:
        chat_history.append((question, "Please analyze an artwork first."))
        return chat_history, ""

    try:
        # Convert chat history from Gradio format to old format for chat_with_artwork
        old_format_history = []
        for user_msg, bot_msg in chat_history:
            if user_msg:
                old_format_history.append({"role": "user", "content": user_msg})
            if bot_msg:
                old_format_history.append({"role": "assistant", "content": bot_msg})

        # Get answer from chatbot
        answer = chat_with_artwork(
            question=question,
            artwork_description=description,
            metadata=metadata,
            chat_history=old_format_history
        )

        # Update chat history (Gradio Chatbot format: list of tuples)
        chat_history.append((question, answer))

        return chat_history, ""  # Return empty string to clear input

    except Exception as e:
        print(f"Error processing text question: {e}")
        chat_history.append((question, f"Error: {str(e)}"))
        return chat_history, ""


def create_audio_guide_interface():
    """Create Audio-Guide interface (accessible mode)"""

    with gr.Blocks(title="Audio-Guide (Accessible Mode)") as demo:

        gr.Markdown("# Audio-Guide (Accessible Mode)")
        gr.Markdown("For blind and visually impaired visitors - Fully accessible with audio feedback")

        # State variables
        description_state = gr.State(None)
        metadata_state = gr.State(None)
        chat_history_state = gr.State([])

        with gr.Tab("Step 1: Upload & Analyze") as tab1:
            gr.Markdown("## Upload Artwork Photo")
            gr.Markdown("Upload a photo of the artwork. It will be automatically analyzed and audio will be generated.")

            image_input = gr.Image(label="Upload Artwork Image", type="pil")
            analyze_btn = gr.Button("Analyze Artwork", variant="primary", size="lg")
            status_output = gr.Textbox(label="Status", interactive=False)

            # Combined audio output (metadata + description)
            gr.Markdown("### Artwork Information")
            combined_audio_output = gr.Audio(
                label="Audio Guide (Metadata followed by Description)",
                autoplay=True,
                waveform_options=gr.WaveformOptions(
                    waveform_color="#3b82f6",
                    waveform_progress_color="#1e40af",
                    show_recording_waveform=False,
                    skip_length=5
                )
            )

        with gr.Tab("Step 2: Ask Questions (Voice)") as tab2:
            gr.Markdown("## Voice Q&A")
            gr.Markdown("Record your question and get an audio answer.")

            audio_question = gr.Audio(label="Record Your Question", sources=["microphone"], type="filepath")
            process_voice_btn = gr.Button("Get Answer", variant="primary", size="lg")

            answer_audio_output = gr.Audio(
                label="Audio Answer",
                autoplay=True,
                waveform_options=gr.WaveformOptions(
                    waveform_color="#3b82f6",
                    waveform_progress_color="#1e40af",
                    show_recording_waveform=False,
                    skip_length=5
                )
            )

        # Button click handlers
        def analyze_and_update(image):
            try:
                print("DEBUG: Starting analyze_and_update...")
                desc, meta, desc_audio, meta_audio, status = analyze_image(image)
                print(f"DEBUG: analyze_image returned - desc: {bool(desc)}, meta: {bool(meta)}")
                print(f"DEBUG: Audio paths - desc: {desc_audio}, meta: {meta_audio}")

                # Combine both audio files into one
                combined_audio_path = None
                if meta_audio and desc_audio:
                    try:
                        from pydub import AudioSegment
                        import uuid

                        print("DEBUG: Combining audio files...")
                        # Load both audio files
                        metadata_audio = AudioSegment.from_mp3(meta_audio)
                        description_audio = AudioSegment.from_mp3(desc_audio)

                        # Add 1 second silence between them
                        silence = AudioSegment.silent(duration=1000)  # 1000ms = 1s

                        # Combine: metadata + silence + description
                        combined = metadata_audio + silence + description_audio

                        # Save combined audio
                        session_id = str(uuid.uuid4())[:8]
                        combined_path = OUTPUT_DIR / f"combined_{session_id}.mp3"
                        combined.export(combined_path, format="mp3")
                        combined_audio_path = str(combined_path)
                        print(f"DEBUG: Combined audio saved to {combined_audio_path}")

                    except Exception as e:
                        print(f"ERROR combining audio: {e}")
                        # Fallback to just metadata audio if combining fails
                        combined_audio_path = meta_audio

                return desc, meta, combined_audio_path, status
            except Exception as e:
                print(f"ERROR in analyze_and_update: {e}")
                import traceback
                traceback.print_exc()
                return None, None, None, f"Error: {str(e)}"

        # Analyze and play combined audio
        analyze_btn.click(
            fn=analyze_and_update,
            inputs=[image_input],
            outputs=[
                description_state,
                metadata_state,
                combined_audio_output,
                status_output
            ]
        )

        process_voice_btn.click(
            fn=process_voice_question,
            inputs=[audio_question, description_state, metadata_state, chat_history_state],
            outputs=[chat_history_state, answer_audio_output]
        )

        # Tab switching handlers - stop audio when switching tabs
        tab1.select(
            fn=lambda: None,  # Clear audio in tab2
            inputs=[],
            outputs=[answer_audio_output]
        )

        tab2.select(
            fn=lambda: None,  # Clear audio in tab1
            inputs=[],
            outputs=[combined_audio_output]
        )

    return demo


def create_visual_guide_interface():
    """Create Visual-Guide interface (standard mode)"""

    with gr.Blocks() as demo:

        gr.Markdown("# Visual-Guide")
        gr.Markdown("Classic visual interface with text and audio")

        # State variables
        description_state = gr.State(None)
        metadata_state = gr.State(None)
        chat_history_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=2):
                # Image upload and display
                image_input = gr.Image(label="Upload Artwork Image", type="pil")
                analyze_btn = gr.Button("Analyze Artwork", variant="primary")

                # Description section
                gr.Markdown("### Artwork Description")
                description_text = gr.Markdown("")
                description_audio = gr.Audio(label="Audio Description")

            with gr.Column(scale=1):
                # Metadata section
                gr.Markdown("### Metadata")
                artist_output = gr.Textbox(label="Artist", interactive=False)
                title_output = gr.Textbox(label="Title", interactive=False)
                year_output = gr.Textbox(label="Year", interactive=False)
                period_output = gr.Textbox(label="Period", interactive=False)

        # Chat section
        gr.Markdown("---")
        gr.Markdown("### Chatbot")

        with gr.Row():
            question_input = gr.Textbox(
                label="Ask a question about the artwork",
                placeholder="e.g., What colors are used? What techniques did the artist use?",
                scale=4
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        # Button handlers
        def analyze_and_display(image):
            desc, meta, desc_audio, meta_audio, status = analyze_image(image)

            if meta:
                return (
                    desc, meta, desc_audio,
                    desc if desc else "",
                    meta.get('artist', 'Unknown'),
                    meta.get('title', 'Unknown'),
                    meta.get('year', 'Unknown'),
                    meta.get('period', 'Unknown')
                )
            else:
                return desc, meta, desc_audio, "", "", "", "", ""

        analyze_btn.click(
            fn=analyze_and_display,
            inputs=[image_input],
            outputs=[
                description_state,
                metadata_state,
                description_audio,
                description_text,
                artist_output,
                title_output,
                year_output,
                period_output
            ]
        )

        send_btn.click(
            fn=process_text_question,
            inputs=[question_input, description_state, metadata_state, chat_history_state],
            outputs=[chat_history_state, question_input]
        )

        # Also allow pressing Enter to send
        question_input.submit(
            fn=process_text_question,
            inputs=[question_input, description_state, metadata_state, chat_history_state],
            outputs=[chat_history_state, question_input]
        )

    return demo


def create_home_interface():
    """Create home page with mode selection"""

    with gr.Blocks() as demo:
        gr.Markdown("# Museum Audio Guide")
        gr.Markdown("---")
        gr.Markdown("## Welcome! Choose your guide mode:")

        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### Audio-Guide

                **For blind and visually impaired visitors**
                - Audio description of artwork
                - Audio metadata
                - Voice-based Q&A chat
                - Fully accessible
                """)
                audio_btn = gr.Button("Start Audio-Guide", variant="primary", size="lg")

            with gr.Column():
                gr.Markdown("""
                ### Visual-Guide

                **Classic visual interface**
                - Text + Audio description
                - Visual metadata display
                - Text-based chat
                - Rich visual experience
                """)
                visual_btn = gr.Button("Start Visual-Guide", variant="secondary", size="lg")

        # Navigation will be handled by TabbedInterface
        gr.Markdown("---")
        gr.Markdown("Use the tabs above to switch between modes")

    return demo


def main():
    """Main application logic"""

    # Check if API key is set
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your_openai_api_key_here":
        print("ERROR: OpenAI API Key not set! Please set it in the .env file.")
        # Create a simple error interface
        with gr.Blocks() as error_demo:
            gr.Markdown("# Error")
            gr.Markdown("OpenAI API Key not set! Please set it in the .env file.")
        error_demo.launch()
        return

    # Create tabbed interface with both modes
    demo = gr.TabbedInterface(
        [create_home_interface(), create_audio_guide_interface(), create_visual_guide_interface()],
        ["Home", "Audio-Guide", "Visual-Guide"],
        title="Museum Audio Guide"
    )

    # Launch the app
    demo.launch(share=False)


if __name__ == "__main__":
    main()
