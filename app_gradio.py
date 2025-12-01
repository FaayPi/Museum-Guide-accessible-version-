"""
Museum Audio Guide - Gradio App
Main application with Audio-Guide and Visual-Guide modes
"""

import gradio as gr
import io
from pathlib import Path
import sys
import uuid

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import backend functions
from utils.vision import analyze_artwork, get_metadata
from utils.audio import text_to_speech, speech_to_text
from utils.chat import chat_with_artwork
import config

# Create audio_outputs directory
OUTPUT_DIR = Path("audio_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

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
    Analyze image and generate audio outputs
    Returns: (description, metadata, description_audio, metadata_audio, status_message)
    """
    if image is None:
        return None, None, None, None, "Please upload an image first."

    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()

        # Step 1: Analyze artwork
        description = analyze_artwork(image_bytes)

        if not description:
            return None, None, None, None, "Unable to analyze the artwork. Please try again."

        # Step 2: Extract metadata
        metadata = get_metadata(image_bytes)
        if not metadata:
            return None, None, None, None, "Unable to retrieve artwork information. Please check your connection and try again."

        # Generate unique session ID
        session_id = str(uuid.uuid4())[:8]

        # Step 3: Generate description audio
        description_audio = text_to_speech(description, timeout=60)
        description_audio_path = None
        if description_audio:
            audio_path = OUTPUT_DIR / f"description_{session_id}.mp3"
            with open(audio_path, "wb") as f:
                f.write(description_audio)
            description_audio_path = str(audio_path)

        # Step 4: Generate metadata audio
        metadata_text = f"""
Artist: {metadata.get('artist', 'Unknown')}.
Title: {metadata.get('title', 'Unknown')}.
Year: {metadata.get('year', 'Unknown')}.
Period: {metadata.get('period', 'Unknown')}.
"""
        metadata_audio = text_to_speech(metadata_text, timeout=60)
        metadata_audio_path = None
        if metadata_audio:
            audio_path = OUTPUT_DIR / f"metadata_{session_id}.mp3"
            with open(audio_path, "wb") as f:
                f.write(metadata_audio)
            metadata_audio_path = str(audio_path)

        return description, metadata, description_audio_path, metadata_audio_path, "Analysis complete! Audio ready to play."

    except Exception as e:
        print(f"ERROR in analyze_image: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, f"An error occurred while processing the artwork: {str(e)}"


def process_voice_question(audio_file, description, metadata, chat_history):
    """
    Process voice question and return audio answer
    Returns: (chat_history, answer_audio, status)
    """
    if audio_file is None:
        return chat_history, None, "Please record a question first."

    if description is None or metadata is None:
        return chat_history, None, "Please analyze an artwork first."

    try:
        # Read audio file
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()

        # Transcribe question
        question = speech_to_text(audio_bytes, language=None)
        if not question:
            return chat_history, None, "Unable to transcribe audio. Please try again."

        # Get answer from chatbot
        answer = chat_with_artwork(
            question=question,
            artwork_description=description,
            metadata=metadata,
            chat_history=chat_history
        )

        # Update chat history
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})

        # Generate audio answer
        answer_audio = text_to_speech(answer, timeout=60)
        answer_audio_path = None

        if answer_audio:
            answer_id = str(uuid.uuid4())[:8]
            audio_path = OUTPUT_DIR / f"answer_{answer_id}.mp3"
            with open(audio_path, "wb") as f:
                f.write(answer_audio)
            answer_audio_path = str(audio_path)

        return chat_history, answer_audio_path, f"Question: {question}\n\nAnswer: {answer}"

    except Exception as e:
        print(f"Error processing voice question: {e}")
        return chat_history, None, f"Error: {str(e)}"


def process_text_question(question, description, metadata, chat_history):
    """
    Process text question and return answer
    Returns: (updated_chat_history, answer)
    """
    if not question or question.strip() == "":
        return chat_history, ""

    if description is None or metadata is None:
        return chat_history, "Please analyze an artwork first."

    try:
        # Get answer from chatbot
        answer = chat_with_artwork(
            question=question,
            artwork_description=description,
            metadata=metadata,
            chat_history=chat_history
        )

        # Update chat history
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})

        return chat_history, answer

    except Exception as e:
        print(f"Error processing text question: {e}")
        return chat_history, f"Error: {str(e)}"


def format_chat_history(chat_history):
    """Format chat history for display"""
    if not chat_history:
        return ""

    formatted = []
    for msg in chat_history:
        if msg['role'] == 'user':
            formatted.append(f"**You:** {msg['content']}")
        else:
            formatted.append(f"**Assistant:** {msg['content']}")

    return "\n\n---\n\n".join(formatted)


def create_audio_guide_interface():
    """Create Audio-Guide interface (accessible mode)"""

    with gr.Blocks(title="Audio-Guide (Accessible Mode)") as demo:

        gr.Markdown("# Audio-Guide (Accessible Mode)")
        gr.Markdown("For blind and visually impaired visitors - Fully accessible with audio feedback")

        # State variables
        description_state = gr.State(None)
        metadata_state = gr.State(None)
        chat_history_state = gr.State([])

        with gr.Tab("Step 1: Upload & Analyze"):
            gr.Markdown("## Upload Artwork Photo")
            gr.Markdown("Upload a photo of the artwork. It will be automatically analyzed and audio will be generated.")

            image_input = gr.Image(label="Upload Artwork Image", type="pil")
            analyze_btn = gr.Button("Analyze Artwork", variant="primary", size="lg")
            status_output = gr.Textbox(label="Status", interactive=False)

            # Outputs - Metadata plays first
            gr.Markdown("### Artwork Information")
            metadata_audio_output = gr.Audio(
                label="Metadata Audio (plays first)",
                autoplay=True,
                waveform_options=gr.WaveformOptions(
                    waveform_color="#3b82f6",
                    waveform_progress_color="#1e40af",
                    show_recording_waveform=False,
                    skip_length=5
                )
            )

            gr.Markdown("---")
            gr.Markdown("### Description")
            description_audio_output = gr.Audio(
                label="Description Audio (plays after metadata)",
                autoplay=True,
                waveform_options=gr.WaveformOptions(
                    waveform_color="#3b82f6",
                    waveform_progress_color="#1e40af",
                    show_recording_waveform=False,
                    skip_length=5
                )
            )

        with gr.Tab("Step 2: Ask Questions (Voice)"):
            gr.Markdown("## Voice Q&A")
            gr.Markdown("Record your question and get an audio answer")

            audio_question = gr.Audio(label="Record Your Question", sources=["microphone"], type="filepath")
            process_voice_btn = gr.Button("Get Answer", variant="primary", size="lg")

            voice_status = gr.Textbox(label="Question & Answer", interactive=False, lines=5)
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

                # Return metadata audio immediately, description audio later
                print("DEBUG: Returning values - metadata audio will autoplay...")
                return desc, meta, meta_audio, None, status, desc_audio, meta_audio
            except Exception as e:
                print(f"ERROR in analyze_and_update: {e}")
                import traceback
                traceback.print_exc()
                return None, None, None, None, f"Error: {str(e)}", None, None

        def load_description_audio(desc_audio_path, meta_audio_path):
            """Load description audio after metadata audio finishes"""
            try:
                print(f"DEBUG: load_description_audio called - desc: {desc_audio_path}, meta: {meta_audio_path}")
                import time
                if meta_audio_path and desc_audio_path:
                    # Calculate metadata audio duration
                    try:
                        from pydub import AudioSegment
                        print(f"DEBUG: Loading metadata audio to calculate duration...")
                        audio = AudioSegment.from_mp3(meta_audio_path)
                        duration_seconds = len(audio) / 1000.0
                        print(f"DEBUG: Metadata duration: {duration_seconds:.2f}s, waiting...")
                        # Wait for metadata to finish plus 1 second buffer
                        time.sleep(duration_seconds + 1)
                    except Exception as e:
                        # Fallback: wait 3 seconds if we can't get duration (metadata is shorter)
                        print(f"DEBUG: Could not calculate duration ({e}), waiting 3s...")
                        time.sleep(3)

                # Return description audio path (will autoplay because of autoplay=True)
                print(f"DEBUG: Returning description audio path for autoplay...")
                return desc_audio_path
            except Exception as e:
                print(f"ERROR in load_description_audio: {e}")
                import traceback
                traceback.print_exc()
                return None

        # State to store audio paths temporarily
        description_audio_path_state = gr.State(None)
        metadata_audio_path_state = gr.State(None)

        # First: Analyze and play metadata audio, then description audio
        analyze_btn.click(
            fn=analyze_and_update,
            inputs=[image_input],
            outputs=[
                description_state,
                metadata_state,
                metadata_audio_output,
                description_audio_output,
                status_output,
                description_audio_path_state,
                metadata_audio_path_state
            ]
        ).then(
            fn=load_description_audio,
            inputs=[description_audio_path_state, metadata_audio_path_state],
            outputs=[description_audio_output]
        )

        process_voice_btn.click(
            fn=process_voice_question,
            inputs=[audio_question, description_state, metadata_state, chat_history_state],
            outputs=[chat_history_state, answer_audio_output, voice_status]
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
                status_output = gr.Textbox(label="Status", interactive=False)

                # Description section
                gr.Markdown("### Artwork Description")
                description_text = gr.Textbox(label="Description", lines=5, interactive=False)
                description_audio = gr.Audio(label="Audio Description")

            with gr.Column(scale=1):
                # Metadata section
                gr.Markdown("### Metadata")
                artist_output = gr.Textbox(label="Artist", interactive=False)
                title_output = gr.Textbox(label="Title", interactive=False)
                year_output = gr.Textbox(label="Year", interactive=False)
                period_output = gr.Textbox(label="Period", interactive=False)
                confidence_output = gr.Textbox(label="Confidence", interactive=False)

        # Chat section
        gr.Markdown("---")
        gr.Markdown("### Chatbot")

        chat_display = gr.Markdown(label="Chat History")

        with gr.Row():
            question_input = gr.Textbox(
                label="Ask a question about the artwork",
                placeholder="e.g., What colors are used? What techniques did the artist use?",
                scale=4
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        answer_output = gr.Textbox(label="Answer", lines=3, interactive=False)

        # Button handlers
        def analyze_and_display(image):
            desc, meta, desc_audio, meta_audio, status = analyze_image(image)

            if meta:
                return (
                    desc, meta, desc_audio, status,
                    desc if desc else "",
                    meta.get('artist', 'Unknown'),
                    meta.get('title', 'Unknown'),
                    meta.get('year', 'Unknown'),
                    meta.get('period', 'Unknown'),
                    meta.get('confidence', 'unknown')
                )
            else:
                return desc, meta, desc_audio, status, "", "", "", "", "", ""

        analyze_btn.click(
            fn=analyze_and_display,
            inputs=[image_input],
            outputs=[
                description_state,
                metadata_state,
                description_audio,
                status_output,
                description_text,
                artist_output,
                title_output,
                year_output,
                period_output,
                confidence_output
            ]
        )

        def handle_question(question, desc, meta, chat_hist):
            new_hist, answer = process_text_question(question, desc, meta, chat_hist)
            formatted_chat = format_chat_history(new_hist)
            return new_hist, formatted_chat, answer, ""

        send_btn.click(
            fn=handle_question,
            inputs=[question_input, description_state, metadata_state, chat_history_state],
            outputs=[chat_history_state, chat_display, answer_output, question_input]
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
