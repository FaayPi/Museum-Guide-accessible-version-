"""
Museum Audio Guide - Streamlit App
Main application with Audio-Guide and Visual-Guide modes
"""

import streamlit as st
from PIL import Image
import io
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import backend functions
from utils.vision import analyze_artwork, get_metadata
from utils.audio import text_to_speech, speech_to_text
from utils.chat import chat_with_artwork
import config

# Import audio recorder (optional)
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False
    print("Warning: streamlit-audio-recorder not installed. Live recording disabled.")

# Create audio_outputs directory
OUTPUT_DIR = Path("audio_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Page config
st.set_page_config(
    page_title="Museum Audio Guide",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'image' not in st.session_state:
    st.session_state.image = None
if 'description' not in st.session_state:
    st.session_state.description = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'description_audio_path' not in st.session_state:
    st.session_state.description_audio_path = None
if 'metadata_audio_path' not in st.session_state:
    st.session_state.metadata_audio_path = None


def reset_session():
    """Reset session to initial state"""
    st.session_state.mode = None
    st.session_state.image = None
    st.session_state.description = None
    st.session_state.metadata = None
    st.session_state.chat_history = []
    st.session_state.description_audio_path = None
    st.session_state.metadata_audio_path = None


def show_home_page():
    """Display home page with mode selection"""
    st.title("Museum Audio Guide")
    st.markdown("---")
    
    st.markdown("""
    ## Welcome! Choose your guide mode:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Audio-Guide
        
        **For blind and visually impaired visitors**
        - Audio description of artwork
        - Audio metadata
        - Voice-based Q&A chat
        - Fully accessible
        """)
        if st.button("Start Audio-Guide", type="primary"):
            st.session_state.mode = "audio"
            st.rerun()
    
    with col2:
        st.markdown("""
        ### Visual-Guide
        
        **Classic visual interface**
        - Text + Audio description
        - Visual metadata display
        - Text-based chat
        - Rich visual experience
        """)
        if st.button("Start Visual-Guide", type="secondary"):
            st.session_state.mode = "visual"
            st.rerun()


def analyze_image(image_bytes):
    """Analyze image and generate audio outputs"""
    
    st.write(f"DEBUG: Starting analysis with image size: {len(image_bytes)} bytes")
    
    try:
        # Step 1: Analyze artwork
        with st.spinner("Analyzing artwork..."):
            st.write("DEBUG: Calling analyze_artwork()...")
            description = analyze_artwork(image_bytes)
            
            st.write(f"DEBUG: Description result: {description[:100] if description else 'None'}...")
            
            if not description:
                st.error("Failed to analyze artwork. Please check:")
                st.error("- Your API key is valid")
                st.error("- You have internet connection")
                st.error("- OpenAI API is accessible")
                st.info("Check the terminal/console for detailed error messages")
                return False
            
            st.session_state.description = description
            st.write(f"DEBUG: Description saved to session_state ({len(description)} chars)")
        
        # Step 2: Extract metadata
        with st.spinner("Extracting metadata..."):
            st.write("DEBUG: Calling get_metadata()...")
            metadata = get_metadata(image_bytes)
            
            st.write(f"DEBUG: Metadata result: {metadata}")
            
            if not metadata:
                st.error("Failed to extract metadata. Please check:")
                st.error("- Your API key is valid")
                st.error("- You have internet connection")
                st.info("Check the terminal/console for detailed error messages")
                return False
            
            st.session_state.metadata = metadata
            st.write(f"DEBUG: Metadata saved to session_state")
        
        # Create audio_outputs directory if it doesn't exist
        from pathlib import Path
        audio_dir = Path("audio_outputs")
        audio_dir.mkdir(exist_ok=True)
        
        import uuid
        session_id = str(uuid.uuid4())[:8]  # Short unique ID
        
        # Step 3: Generate description audio and save to file
        st.write("DEBUG: About to generate description audio...")
        st.write(f"DEBUG: Description length: {len(description)} chars")
        
        # Try without spinner first
        description_audio = None
        try:
            st.write("DEBUG: Calling text_to_speech()...")
            description_audio = text_to_speech(description, timeout=60)
            st.write(f"DEBUG: text_to_speech() returned!")
            st.write(f"DEBUG: Result is None: {description_audio is None}")
        except Exception as e:
            st.error(f"Exception during TTS: {e}")
            import traceback
            st.write(traceback.format_exc())
        
        if description_audio:
            st.write(f"DEBUG: Audio size: {len(description_audio)} bytes")
            # Save to file
            audio_path = audio_dir / f"description_{session_id}.mp3"
            with open(audio_path, "wb") as f:
                f.write(description_audio)
            st.session_state.description_audio_path = str(audio_path)
            st.success(f"Description audio saved to {audio_path}")
        else:
            st.warning("Could not generate audio description")
            st.session_state.description_audio_path = None
        
        # Step 4: Generate metadata audio and save to file
        metadata_text = f"""
Artist: {metadata.get('artist', 'Unknown')}.
Title: {metadata.get('title', 'Unknown')}.
Year: {metadata.get('year', 'Unknown')}.
Period: {metadata.get('period', 'Unknown')}.
"""
        st.write("DEBUG: About to generate metadata audio...")
        st.write(f"DEBUG: Metadata text length: {len(metadata_text)} chars")
        
        # Try without spinner first
        metadata_audio = None
        try:
            st.write("DEBUG: Calling text_to_speech()...")
            metadata_audio = text_to_speech(metadata_text, timeout=60)
            st.write(f"DEBUG: text_to_speech() returned!")
            st.write(f"DEBUG: Result is None: {metadata_audio is None}")
        except Exception as e:
            st.error(f"Exception during TTS: {e}")
            import traceback
            st.write(traceback.format_exc())
        
        if metadata_audio:
            st.write(f"DEBUG: Audio size: {len(metadata_audio)} bytes")
            # Save to file
            audio_path = audio_dir / f"metadata_{session_id}.mp3"
            with open(audio_path, "wb") as f:
                f.write(metadata_audio)
            st.session_state.metadata_audio_path = str(audio_path)
            st.success(f"Metadata audio saved to {audio_path}")
        else:
            st.warning("Could not generate metadata audio")
            st.session_state.metadata_audio_path = None
        
        st.success("Analysis complete!")
        st.write("DEBUG: Returning True")
        return True
        
    except Exception as e:
        st.error(f"Unexpected error during analysis: {str(e)}")
        st.info("Check the terminal/console for detailed error messages")
        print(f"Exception in analyze_image: {str(e)}")
        import traceback
        traceback.print_exc()
        st.write(f"DEBUG: Exception occurred: {str(e)}")
        return False


def audio_guide_page():
    """Audio-Guide mode interface"""
    st.title("Audio-Guide")
    
    # Back button
    if st.button("← Back to Home"):
        reset_session()
        st.rerun()
    
    st.markdown("---")
    
    # Check if analysis is complete
    analysis_complete = (
        st.session_state.image is not None and 
        st.session_state.description is not None and 
        st.session_state.metadata is not None
    )
    
    # Step 1: Upload Image & Analyze
    if not analysis_complete:
        st.markdown("## Step 1: Upload Artwork Photo")
        st.info("Upload a photo of the artwork. Audio description will be automatically generated.")
        
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Supported formats: JPG, PNG, WEBP"
        )
        
        if uploaded_file:
            # Store image
            st.session_state.image = uploaded_file.getvalue()
            
            # Show image
            image = Image.open(io.BytesIO(st.session_state.image))
            st.image(image, caption="Uploaded artwork", width="stretch")
            
            # Analyze button
            if st.button("Analyze Artwork", type="primary"):
                success = analyze_image(st.session_state.image)
                if success:
                    st.rerun()
                else:
                    st.error("Analysis failed. Please try again.")
    
    # Step 2: Show results and audio player (only if analysis is complete)
    else:
        st.write("=" * 60)
        st.write("DEBUG: AFTER RERUN - Checking session state:")
        st.write(f"   - image exists: {st.session_state.image is not None}")
        st.write(f"   - description exists: {st.session_state.description is not None}")
        st.write(f"   - metadata exists: {st.session_state.metadata is not None}")
        st.write(f"   - description_audio_path exists: {st.session_state.description_audio_path is not None}")
        st.write(f"   - metadata_audio_path exists: {st.session_state.metadata_audio_path is not None}")
        if st.session_state.description_audio_path:
            st.write(f"   - description_audio_path: {st.session_state.description_audio_path}")
        if st.session_state.metadata_audio_path:
            st.write(f"   - metadata_audio_path: {st.session_state.metadata_audio_path}")
        st.write("=" * 60)
        
        # Display image
        image = Image.open(io.BytesIO(st.session_state.image))
        st.image(image, width="stretch")
        
        st.markdown("---")
        
        # Audio Description
        st.markdown("## Audio Description")
        
        st.write(f"DEBUG: description_audio_path: {st.session_state.description_audio_path}")
        
        if st.session_state.description_audio_path:
            from pathlib import Path
            audio_file = Path(st.session_state.description_audio_path)
            if audio_file.exists():
                st.write(f"EBUG: Audio file exists, size: {audio_file.stat().st_size} bytes")
                
                # Read audio file as base64
                import base64
                with open(audio_file, "rb") as f:
                    audio_bytes = f.read()
                audio_b64 = base64.b64encode(audio_bytes).decode()
                
                # Create HTML5 audio player
                audio_html = f"""
                <audio controls style="width: 100%;">
                    <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
                st.success("Audio player loaded - click play to listen!")
            else:
                st.error("Audio file not found")
            
            # with st.expander("View text description"):
            #     st.write(st.session_state.description)
        else:
            st.warning("Audio description not generated.")
            st.write(st.session_state.description)
        
        st.markdown("---")
        
        # Audio Metadata
        st.markdown("## Artwork Metadata (Audio)")
        
        st.write(f"DEBUG: metadata_audio_path: {st.session_state.metadata_audio_path}")
        
        if st.session_state.metadata_audio_path:
            from pathlib import Path
            audio_file = Path(st.session_state.metadata_audio_path)
            if audio_file.exists():
                st.write(f"DEBUG: Audio file exists, size: {audio_file.stat().st_size} bytes")
                
                # Read audio file as base64
                import base64
                with open(audio_file, "rb") as f:
                    audio_bytes = f.read()
                audio_b64 = base64.b64encode(audio_bytes).decode()
                
                # Create HTML5 audio player
                audio_html = f"""
                <audio controls style="width: 100%;">
                    <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)
                st.success("Audio player loaded - click play to listen!")
            else:
                st.error("Audio file not found")
            
            # with st.expander("View metadata"):
            #     col1, col2 = st.columns(2)
            #     with col1:
            #         st.write(f"**Artist:** {st.session_state.metadata.get('artist', 'Unknown')}")
            #         st.write(f"**Title:** {st.session_state.metadata.get('title', 'Unknown')}")
            #     with col2:
            #         st.write(f"**Year:** {st.session_state.metadata.get('year', 'Unknown')}")
            #         st.write(f"**Period:** {st.session_state.metadata.get('period', 'Unknown')}")
        else:
            st.warning("Audio metadata not generated.")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Artist:** {st.session_state.metadata.get('artist', 'Unknown')}")
                st.write(f"**Title:** {st.session_state.metadata.get('title', 'Unknown')}")
            with col2:
                st.write(f"**Year:** {st.session_state.metadata.get('year', 'Unknown')}")
                st.write(f"**Period:** {st.session_state.metadata.get('period', 'Unknown')}")
        
        st.markdown("---")
            
        st.markdown("### Record Your Question")
            
        # Audio recorder widget
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#ff4b4b",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="3x",
        )
        
        if audio_bytes:
            st.success("Recording received!")
            
            # Show playback
            st.audio(audio_bytes, format="audio/wav")
            
            if st.button("Process Question", type="primary", key="process_live_recording"):
                with st.spinner("Transcribing question..."):
                    # Save audio for debugging
                    import uuid
                    audio_dir = Path("audio_outputs")
                    audio_dir.mkdir(exist_ok=True)
                    debug_id = str(uuid.uuid4())[:8]
                    debug_path = audio_dir / f"recording_{debug_id}.wav"
                    
                    with open(debug_path, "wb") as f:
                        f.write(audio_bytes)
                    
                    st.write(f"DEBUG: Audio saved to {debug_path} for review")
                    st.write(f"DEBUG: Audio size: {len(audio_bytes)} bytes")
                    
                    # Transcribe question (auto-detect language)
                    question = speech_to_text(audio_bytes, language=None)
                    
                    st.write(f"**Your question:** {question}")
                    st.info(f"If transcription is wrong, play the audio file: {debug_path}")
                
                with st.spinner("Getting answer..."):
                    # Get answer from chatbot
                    answer = chat_with_artwork(
                        question=question,
                        artwork_description=st.session_state.description,
                        metadata=st.session_state.metadata,
                        chat_history=st.session_state.chat_history
                    )
                    
                    st.write(f"**Answer:** {answer}")
                    
                    # Update chat history
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                with st.spinner("Generating audio answer..."):
                    # Generate audio answer
                    answer_audio = text_to_speech(answer, timeout=60)
                    
                    if answer_audio:
                        st.write(f"DEBUG: Audio size: {len(answer_audio)} bytes")
                        
                        # Save to file
                        import uuid
                        audio_dir = Path("audio_outputs")
                        audio_dir.mkdir(exist_ok=True)
                        answer_id = str(uuid.uuid4())[:8]
                        audio_path = audio_dir / f"answer_{answer_id}.mp3"
                        
                        with open(audio_path, "wb") as f:
                            f.write(answer_audio)
                        
                        st.markdown("### Audio Answer:")
                        
                        # Read audio and convert to base64
                        import base64
                        with open(audio_path, "rb") as f:
                            audio_bytes_read = f.read()
                        audio_b64 = base64.b64encode(audio_bytes_read).decode()
                        
                        # Create HTML5 audio player
                        audio_html = f"""
                        <audio controls style="width: 100%;" autoplay>
                            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                            Your browser does not support the audio element.
                        </audio>
                        """
                        st.markdown(audio_html, unsafe_allow_html=True)
                        st.success("Audio answer playing!")
                    else:
                        st.error("Failed to generate audio answer.")
        
        # Display chat history
        if st.session_state.chat_history:
            with st.expander("View conversation history"):
                for i, msg in enumerate(st.session_state.chat_history):
                    if msg['role'] == 'user':
                        st.markdown(f"**You:** {msg['content']}")
                    else:
                        st.markdown(f"**Assistant:** {msg['content']}")
                    if i < len(st.session_state.chat_history) - 1:
                        st.markdown("---")
        
        st.markdown("---")
        
        # New analysis button
        if st.button("Analyze New Artwork"):
            st.session_state.image = None
            st.session_state.description = None
            st.session_state.metadata = None
            st.session_state.chat_history = []
            st.session_state.description_audio_path = None
            st.session_state.metadata_audio_path = None
            st.rerun()


def visual_guide_page():
    """Visual-Guide mode interface"""
    st.title("Visual-Guide")
    
    # Back button
    if st.button("← Back to Home"):
        reset_session()
        st.rerun()
    
    st.markdown("---")
    
    # Check if analysis is complete
    analysis_complete = (
        st.session_state.image is not None and 
        st.session_state.description is not None and 
        st.session_state.metadata is not None
    )
    
    # Step 1: Upload Image & Analyze
    if not analysis_complete:
        st.markdown("## Upload Artwork")
        
        uploaded_file = st.file_uploader(
            "Choose an image of the artwork",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Supported formats: JPG, PNG, WEBP"
        )
        
        if uploaded_file:
            # Store image
            st.session_state.image = uploaded_file.getvalue()
            
            # Show image
            image = Image.open(io.BytesIO(st.session_state.image))
            st.image(image, caption="Uploaded artwork", width="stretch")
            
            # Analyze button
            if st.button("Analyze Artwork", type="primary"):
                success = analyze_image(st.session_state.image)
                if success:
                    st.rerun()
                else:
                    st.error("Analysis failed. Please try again.")
    
    # Step 2: Show results (only if analysis is complete)
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display image
            image = Image.open(io.BytesIO(st.session_state.image))
            st.image(image, width="stretch")
            
            # Description section
            st.markdown("### Artwork Description")
            
            if st.session_state.description:
                st.write(st.session_state.description)
                
                # Audio playback option
                if st.session_state.description_audio_path:
                    from pathlib import Path
                    audio_file = Path(st.session_state.description_audio_path)
                    if audio_file.exists():
                        st.markdown("#### 🔊 Listen to description")
                        
                        import base64
                        with open(audio_file, "rb") as f:
                            audio_bytes = f.read()
                        audio_b64 = base64.b64encode(audio_bytes).decode()
                        
                        audio_html = f"""
                        <audio controls style="width: 100%;">
                            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                            Your browser does not support the audio element.
                        </audio>
                        """
                        st.markdown(audio_html, unsafe_allow_html=True)
            else:
                st.warning("Description not available. Please try analyzing again.")
        
        with col2:
            # Metadata card
            st.markdown("### Metadata")
            
            if st.session_state.metadata:
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                    <h4> {st.session_state.metadata.get('artist', 'Unknown')}</h4>
                    <p><strong>Title:</strong> {st.session_state.metadata.get('title', 'Unknown')}</p>
                    <p><strong>Year:</strong> {st.session_state.metadata.get('year', 'Unknown')}</p>
                    <p><strong>Period:</strong> {st.session_state.metadata.get('period', 'Unknown')}</p>
                    <p><strong>Confidence:</strong> {st.session_state.metadata.get('confidence', 'unknown')}</p>
                </div>
                """, unsafe_allow_html=True) 
            else:
                st.warning("⚠️ Metadata not available. Please try analyzing again.")
        
        st.markdown("---")
        
        # Chat section
        st.markdown("### Ask Questions")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**Assistant:** {msg['content']}")
                st.markdown("---")
        
        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            user_question = st.text_input(
                "Type your question about the artwork:",
                placeholder="e.g., What colors are used? What techniques did the artist use?"
            )
            submit_button = st.form_submit_button("Send")
        
        if submit_button and user_question:
            # Check if we have description and metadata
            if not st.session_state.description or not st.session_state.metadata:
                st.error("Please analyze an artwork first before asking questions!")
            else:
                with st.spinner("Thinking..."):
                    try:
                        # Get answer
                        answer = chat_with_artwork(
                            question=user_question,
                            artwork_description=st.session_state.description,
                            metadata=st.session_state.metadata,
                            chat_history=st.session_state.chat_history
                        )
                        
                        # Update chat history
                        st.session_state.chat_history.append({"role": "user", "content": user_question})
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Chat error: {str(e)}")
                
                st.rerun()
        
        st.markdown("---")
        
        # New analysis button
        if st.button("Analyze New Artwork"):
            st.session_state.image = None
            st.session_state.description = None
            st.session_state.metadata = None
            st.session_state.chat_history = []
            st.session_state.description_audio_path = None
            st.session_state.metadata_audio_path = None
            st.rerun()


def main():
    """Main application logic"""
    
    # Check if API key is set
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your_openai_api_key_here":
        st.error("OpenAI API Key not set! Please set it in the .env file.")
        st.stop()
    
    # Route to appropriate page
    if st.session_state.mode is None:
        show_home_page()
    elif st.session_state.mode == "audio":
        audio_guide_page()
    elif st.session_state.mode == "visual":
        visual_guide_page()


if __name__ == "__main__":
    main()
