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
if 'audio_played' not in st.session_state:
    st.session_state.audio_played = False
if 'welcome_audio_played' not in st.session_state:
    st.session_state.welcome_audio_played = False
if 'welcome_audio_path' not in st.session_state:
    st.session_state.welcome_audio_path = None
if 'welcome_audio_bytes' not in st.session_state:
    st.session_state.welcome_audio_bytes = None
if 'auto_analyze_done' not in st.session_state:
    st.session_state.auto_analyze_done = False


def reset_session():
    """Reset session to initial state"""
    st.session_state.mode = None
    st.session_state.image = None
    st.session_state.description = None
    st.session_state.metadata = None
    st.session_state.chat_history = []
    st.session_state.description_audio_path = None
    st.session_state.metadata_audio_path = None
    st.session_state.audio_played = False
    st.session_state.welcome_audio_played = False
    st.session_state.welcome_audio_path = None
    st.session_state.welcome_audio_bytes = None
    st.session_state.auto_analyze_done = False


def generate_welcome_audio():
    """Generate welcome audio message for Audio-Guide mode"""
    welcome_text = """Welcome to the accessible Audio Guide of the Museum Guide App. 
    
    You have entered a fully accessible mode designed for blind and visually impaired visitors.
    
    Here's what you can do:
    First, upload a photo of an artwork by tapping the upload button.
    Once uploaded, the artwork will be automatically analyzed.
    After analysis, you will hear a detailed audio description of the artwork, followed by information about the artist, title, year, and period.
    You can then ask questions about the artwork using voice recording.
    Simply tap the microphone button, ask your question, and you will receive an audio answer.
    
    Let's begin. Please tap the upload button to select a photo of the artwork you want to explore."""
    
    try:
        import uuid
        audio_dir = Path("audio_outputs")
        audio_dir.mkdir(exist_ok=True)
        
        print(f"Generating welcome audio with {len(welcome_text)} characters...")
        
        # Generate audio
        audio_bytes = text_to_speech(welcome_text, timeout=60)
        
        if audio_bytes and len(audio_bytes) > 0:
            print(f"Welcome audio generated: {len(audio_bytes)} bytes")
            # Save to file
            welcome_id = str(uuid.uuid4())[:8]
            audio_path = audio_dir / f"welcome_{welcome_id}.mp3"
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            
            print(f"Welcome audio saved to: {audio_path}")
            return audio_bytes  # Return bytes directly instead of path
        else:
            print("ERROR: text_to_speech returned None or empty bytes")
            return None
    except Exception as e:
        print(f"Error generating welcome audio: {e}")
        import traceback
        traceback.print_exc()
        return None


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
    
    try:
        # Step 1: Analyze artwork
        with st.spinner("Examining the artwork and identifying visual elements..."):
            description = analyze_artwork(image_bytes)
            
            if not description:
                st.error("Unable to analyze the artwork. Please ensure your internet connection is active and try again.")
                return False
            
            st.session_state.description = description
        
        # Step 2: Extract metadata
        with st.spinner("Identifying the artist, title, and historical context..."):
            metadata = get_metadata(image_bytes)
            
            if not metadata:
                st.error("Unable to retrieve artwork information. Please check your connection and try again.")
                return False
            
            st.session_state.metadata = metadata
        
        # Create audio_outputs directory if it doesn't exist
        from pathlib import Path
        audio_dir = Path("audio_outputs")
        audio_dir.mkdir(exist_ok=True)
        
        import uuid
        session_id = str(uuid.uuid4())[:8]
        
        # Step 3: Generate description audio and save to file
        with st.spinner("Preparing audio description of the artwork..."):
            description_audio = text_to_speech(description, timeout=60)
        
        if description_audio:
            audio_path = audio_dir / f"description_{session_id}.mp3"
            with open(audio_path, "wb") as f:
                f.write(description_audio)
            st.session_state.description_audio_path = str(audio_path)
        else:
            st.session_state.description_audio_path = None
        
        # Step 4: Generate metadata audio and save to file
        metadata_text = f"""
Artist: {metadata.get('artist', 'Unknown')}.
Title: {metadata.get('title', 'Unknown')}.
Year: {metadata.get('year', 'Unknown')}.
Period: {metadata.get('period', 'Unknown')}.
"""
        with st.spinner("Creating audio summary of artwork details..."):
            metadata_audio = text_to_speech(metadata_text, timeout=60)
        
        if metadata_audio:
            audio_path = audio_dir / f"metadata_{session_id}.mp3"
            with open(audio_path, "wb") as f:
                f.write(metadata_audio)
            st.session_state.metadata_audio_path = str(audio_path)
        else:
            st.session_state.metadata_audio_path = None
        
        st.success("Artwork analysis complete. Audio ready to play.")
        return True
        
    except Exception as e:
        st.error("An error occurred while processing the artwork. Please try again.")
        return False


def audio_guide_page():
    """Audio-Guide mode interface - Fully accessible for blind users"""
    
    # Apply high-contrast CSS for accessibility
    st.markdown("""
    <style>
    /* High-contrast theme for Audio Guide */
    .stApp {
        background-color: #000000;
        color: #FFFF00;
    }
    
    .stButton > button {
        background-color: #FFFF00;
        color: #000000;
        font-size: 24px;
        font-weight: bold;
        padding: 20px 40px;
        border: 3px solid #FFFFFF;
        border-radius: 10px;
        min-height: 60px;
        min-width: 200px;
    }
    
    .stButton > button:hover {
        background-color: #FFFFFF;
        color: #000000;
        border: 3px solid #FFFF00;
    }
    
    .stMarkdown, .stText, p, h1, h2, h3 {
        color: #FFFF00 !important;
        font-size: 20px;
    }
    
    h1 {
        font-size: 36px !important;
        font-weight: bold;
    }
    
    h2 {
        font-size: 28px !important;
    }
    
    .stAlert {
        background-color: #333333;
        color: #FFFF00;
        border: 2px solid #FFFF00;
        font-size: 20px;
    }
    
    /* Large touch targets */
    .stFileUploader {
        font-size: 22px;
    }
    
    .stFileUploader > div {
        min-height: 80px;
    }
    
    /* Hide unnecessary elements for cleaner interface */
    .stImage {
        border: 3px solid #FFFF00;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Audio-Guide (Accessible Mode)")
    
    
    # Back button with clear label
    if st.button("← BACK TO HOME", key="back_button", help="Return to home page"):
        reset_session()
        st.rerun()
    
    st.markdown("---")
    
    # Check if analysis is complete
    analysis_complete = (
        st.session_state.image is not None and 
        st.session_state.description is not None and 
        st.session_state.metadata is not None
    )
    
    # Step 1: Upload Image & Auto-Analyze
    if not analysis_complete:
        st.markdown("## Step 1: Upload Artwork Photo")
        st.info("Upload a photo of the artwork. It will be automatically analyzed and audio will play.")
        
        uploaded_file = st.file_uploader(
            "Choose an image (JPG, PNG, WEBP)",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Tap to select image from your device",
            label_visibility="visible"
        )
        
        if uploaded_file:
            # Store image
            st.session_state.image = uploaded_file.getvalue()
            
            # Show image with high contrast border
            image = Image.open(io.BytesIO(st.session_state.image))
            st.image(image, caption="Uploaded artwork", use_container_width=True)
            
            # AUTO-ANALYZE: Automatically analyze without button click
            if not st.session_state.auto_analyze_done:
                success = analyze_image(st.session_state.image)
                if success:
                    st.session_state.auto_analyze_done = True
                    st.rerun()
                else:
                    st.error("Analysis failed. Please try uploading again.")
                    st.session_state.auto_analyze_done = False
    
    # Step 2: Show results and audio playback (only if analysis is complete)
    else:
        # Display image
        image = Image.open(io.BytesIO(st.session_state.image))
        st.image(image, use_container_width=True)
        
        st.markdown("---")
        
        # Audio Description
        st.markdown("## Description")
        
        if st.session_state.description_audio_path:
            from pathlib import Path
            audio_file = Path(st.session_state.description_audio_path)
            if audio_file.exists():
                # Read audio file and encode to base64
                import base64
                with open(audio_file, "rb") as f:
                    audio_bytes = f.read()
                
                audio_b64 = base64.b64encode(audio_bytes).decode()
                
                # Add unique ID for this audio player
                audio_id = "description_audio"
                
                # Create HTML with autoplay JavaScript
                audio_html = f'''
                <audio id="{audio_id}" controls style="width: 100%;">
                    <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                </audio>
                <script>
                    // Auto-play the audio when page loads
                    (function() {{
                        var audio = document.getElementById('{audio_id}');
                        if (audio) {{
                            // Try to play - handle promise for newer browsers
                            var playPromise = audio.play();
                            if (playPromise !== undefined) {{
                                playPromise.then(function() {{
                                    console.log('Audio playing automatically');
                                }}).catch(function(error) {{
                                    console.log('Autoplay prevented:', error);
                                }});
                            }}
                        }}
                    }})();
                </script>
                '''
                
                st.markdown(audio_html, unsafe_allow_html=True)
            else:
                st.error("Audio file not found")
        else:
            st.warning("Audio description not generated.")
            with st.expander("View text description"):
                st.write(st.session_state.description)
        
        st.markdown("---")
        
        # Audio Metadata
        st.markdown("## Artwork Information")
        
        if st.session_state.metadata_audio_path:
            from pathlib import Path
            audio_file = Path(st.session_state.metadata_audio_path)
            if audio_file.exists():
                # Read audio file and encode to base64
                import base64
                with open(audio_file, "rb") as f:
                    audio_bytes = f.read()
                
                audio_b64 = base64.b64encode(audio_bytes).decode()
                
                # Add unique ID for this audio player
                metadata_audio_id = "metadata_audio"
                
                # Create HTML with autoplay after description ends
                audio_html = f'''
                <audio id="{metadata_audio_id}" controls style="width: 100%;">
                    <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                </audio>
                <script>
                    // Auto-play metadata after description ends
                    (function() {{
                        var descriptionAudio = document.getElementById('description_audio');
                        var metadataAudio = document.getElementById('{metadata_audio_id}');
                        
                        if (descriptionAudio && metadataAudio) {{
                            // Wait for description to end, then play metadata
                            descriptionAudio.addEventListener('ended', function() {{
                                console.log('Description ended, playing metadata...');
                                setTimeout(function() {{
                                    var playPromise = metadataAudio.play();
                                    if (playPromise !== undefined) {{
                                        playPromise.then(function() {{
                                            console.log('Metadata playing automatically');
                                        }}).catch(function(error) {{
                                            console.log('Metadata autoplay prevented:', error);
                                        }});
                                    }}
                                }}, 1000); // 1 second pause between audio clips
                            }});
                        }}
                    }})();
                </script>
                '''
                
                st.markdown(audio_html, unsafe_allow_html=True)
            else:
                st.error("Audio file not found")
        else:
            st.warning("Audio metadata not generated.")
            with st.expander("View text metadata"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Artist:** {st.session_state.metadata.get('artist', 'Unknown')}")
                    st.write(f"**Title:** {st.session_state.metadata.get('title', 'Unknown')}")
                with col2:
                    st.write(f"**Year:** {st.session_state.metadata.get('year', 'Unknown')}")
                    st.write(f"**Period:** {st.session_state.metadata.get('period', 'Unknown')}")
        
        st.markdown("---")
        
        # Voice Q&A Section with clear instructions
        st.markdown("## Chatbot")
        
        # Check if audio recorder is available
        if not AUDIO_RECORDER_AVAILABLE:
            st.error("Audio recorder not available. Please install: pip install audio-recorder-streamlit")
            st.stop()
        
        st.info("Tap the button below to record your question.")
        
        # Audio recorder widget
        audio_bytes = audio_recorder(
            text="TAP TO RECORD QUESTION",
            recording_color="#FF0000",
            neutral_color="#FFFFFF",
            icon_name="microphone",
            icon_size="3x",
        )
        
        if audio_bytes:
            # Show playback
            st.audio(audio_bytes, format="audio/wav")
            
            if st.button("GET ANSWER", type="primary", key="process_live_recording", help="Process your question and get an answer"):
                
                with st.spinner("Converting your voice recording to text..."):
                    # Save audio for debugging
                    import uuid
                    audio_dir = Path("audio_outputs")
                    audio_dir.mkdir(exist_ok=True)
                    debug_id = str(uuid.uuid4())[:8]
                    debug_path = audio_dir / f"recording_{debug_id}.wav"
                    
                    with open(debug_path, "wb") as f:
                        f.write(audio_bytes)
                    
                    # Transcribe question (auto-detect language)
                    question = speech_to_text(audio_bytes, language=None)
                
                with st.spinner("Analyzing the artwork to answer your question..."):
                    # Get answer from chatbot
                    answer = chat_with_artwork(
                        question=question,
                        artwork_description=st.session_state.description,
                        metadata=st.session_state.metadata,
                        chat_history=st.session_state.chat_history
                    )
                    
                    # Update chat history
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                with st.spinner("Preparing audio response..."):
                    # Generate audio answer
                    answer_audio = text_to_speech(answer, timeout=60)
                    
                    if answer_audio:
                        # Save to file
                        import uuid
                        audio_dir = Path("audio_outputs")
                        audio_dir.mkdir(exist_ok=True)
                        answer_id = str(uuid.uuid4())[:8]
                        audio_path = audio_dir / f"answer_{answer_id}.mp3"
                        
                        with open(audio_path, "wb") as f:
                            f.write(answer_audio)
                        
                        st.markdown("### Audio Answer")
                        st.info("Tap the play button to hear the answer")
                        
                        # Read from file, encode to base64, and display
                        import base64
                        with open(audio_path, "rb") as f:
                            audio_data = f.read()
                        
                        audio_b64 = base64.b64encode(audio_data).decode()
                        audio_html = f'<audio controls src="data:audio/mp3;base64,{audio_b64}" style="width: 100%;"></audio>'
                        st.markdown(audio_html, unsafe_allow_html=True)
                    else:
                        st.error("Unable to generate audio response. Please try again.")
        
        st.markdown("---")
        
        # New analysis button - large and prominent
        if st.button("ANALYZE NEW ARTWORK", type="primary", key="new_artwork", help="Start over with a new artwork"):
            st.session_state.image = None
            st.session_state.description = None
            st.session_state.metadata = None
            st.session_state.chat_history = []
            st.session_state.description_audio_path = None
            st.session_state.metadata_audio_path = None
            st.session_state.audio_played = False
            st.session_state.auto_analyze_done = False
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
            st.image(image, caption="Uploaded artwork", use_container_width=True)
            
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
            st.image(image, use_container_width=True)
            
            # Description section
            st.markdown("### Artwork Description")
            
            if st.session_state.description:
                with st.expander("View Description", expanded=False):
                    st.write(st.session_state.description)
                
                # Audio playback option
                if st.session_state.description_audio_path:
                    from pathlib import Path
                    audio_file = Path(st.session_state.description_audio_path)
                    if audio_file.exists():
                        st.markdown("#### Audio Description")
                        
                        # Read audio file and encode to base64
                        import base64
                        with open(audio_file, "rb") as f:
                            audio_bytes = f.read()
                        
                        audio_b64 = base64.b64encode(audio_bytes).decode()
                        
                        # Create HTML audio player (same method as audio guide)
                        audio_html = f'''
                        <audio controls style="width: 100%;">
                            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                        </audio>
                        '''
                        
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
        st.markdown("### Chatbot")
        
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
                st.error("Please analyze an artwork first.")
            else:
                with st.spinner("Finding the answer to your question..."):
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
                        st.error("Unable to process your question. Please try again.")
                
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
