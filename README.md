---
title: Museum Audio Guide
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: false
license: mit
---

# Museum Audio Guide 🎨🔊

AI-powered museum guide that makes art accessible through computer vision, conversational AI, and text-to-speech. Built with accessibility in mind for blind and visually impaired visitors.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🚀 **[Try the Live Demo on Hugging Face](https://huggingface.co/spaces/FeePieper/museum_guide_app)**

---

## 🎯 Overview

Upload a photo of any artwork and receive:
- AI-generated description (visual elements, style, mood)
- Artwork metadata (artist, title, period, historical context)
- Interactive Q&A about techniques, symbolism, and meaning
- Audio narration for hands-free accessibility

**Solution:** Multi-modal AI system combining GPT-4o Vision, RAG vector search, conversational AI, and text-to-speech.

---

## ✨ Key Features

### 1. Dual-Mode Interface
- **Audio-Guide Mode**: Automated audio playback, voice-based Q&A
- **Visual-Guide Mode**: Text display, metadata cards, text-based chat

### 2. Multi-Tier Recognition System (35s → 6-8s)
```
Tier 1: Perceptual Hash    (~0.25s)  ⚡ 95% of known artworks
Tier 2: Pre-check          (~0.05s)  ⚡ Filters non-artworks
Tier 3: RAG Vector Search  (2.5s)    🔍 Semantic similarity
Tier 4: Vision API         (2-3s)    🤖 Universal fallback
```

### 3. Optimized Performance
- Chat responses: 5s → 1-2s (60% faster)
- Token usage: 620 → 190 tokens (68% reduction)
- API cost: 71% cheaper per request

---

## 🏗️ Architecture

```
Gradio Interface → Core Engine → Cache/Services → OpenAI + Pinecone
                   (Multi-tier   (Hash cache)     (Vision, TTS,
                    recognition,  (Service layer)   Chat, RAG)
                    parallel API,
                    retry logic)
```

### Key Design Decisions

**Multi-Tier Recognition:** Hash → Pre-check → RAG → Vision API (95% cache hit, 80% cost reduction)

**Parallel Processing:** ThreadPoolExecutor for concurrent API calls (2x speedup)

**Token Optimization:** Reduced prompts from 620 → 190 tokens (71% cost savings)

**Tech Stack:** Gradio (rapid prototyping), GPT-4o-mini (speed/cost balance), Pinecone (managed vector DB)

---

## 🛠️ Technology Stack

- **Gradio 6.0.1** - Web interface
- **OpenAI GPT-4o-mini** - Vision API, Chat API, TTS API
- **Pinecone 8.0.0** - Vector database (RAG)
- **Pillow 10.1.0** - Image processing
- **imagehash 4.3.1** - Perceptual hashing

**Production features:** Environment config, JSON logging, health checks, retry logic, rate limiting

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [OpenAI API key](https://platform.openai.com/api-keys)
- [Pinecone API key](https://www.pinecone.io/)

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/museum_guide_app.git
cd museum_guide_app

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your API keys to .env:
# OPENAI_API_KEY=sk-...
# PINECONE_API_KEY=pcsk-...

# Run application
python app.py
```

Access at **http://localhost:7860**

### Docker

```bash
docker build -t museum-guide:latest .
docker run -d --name museum-guide -p 7860:7860 --env-file .env museum-guide:latest
```

---

## 💡 Usage

**Audio-Guide Mode** (blind/visually impaired visitors):
1. Upload artwork photo
2. Automatic audio playback (description + metadata)
3. Voice-based Q&A

**Visual-Guide Mode** (sighted visitors):
1. Upload artwork photo
2. Read description and metadata
3. Text-based chat ("What techniques did the artist use?", "What does this symbolize?")

---

## ⚙️ Configuration

Key environment variables in `.env`:

```bash
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk-...
ENVIRONMENT=development  # or production, testing
LOG_LEVEL=INFO
PORT=7860
```

---

## 📁 Project Structure

```
museum_guide_app/
├── app.py                     # Main Gradio application
├── config.py                  # Environment configuration
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
├── README.md                  # Project documentation
├── AI_EVALUATION_REPORT.md    # AI quality test results
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── src/                       # Source code
│   ├── __init__.py
│   ├── core/                  # Core business logic
│   │   ├── analyze.py         # Multi-tier artwork recognition
│   │   ├── error_handler.py   # Retry logic & error handling
│   │   ├── health_check.py    # Health monitoring
│   │   └── logging_config.py  # Logging configuration
│   ├── services/              # External service integrations
│   │   ├── vision.py          # OpenAI Vision API
│   │   ├── audio.py           # OpenAI TTS API
│   │   ├── chat.py            # OpenAI Chat API
│   │   ├── rag_database.py    # Pinecone vector database
│   │   └── image_similarity.py # Perceptual hashing
│   └── models/                # Data models
│       └── types.py           # Type definitions
├── tests/                     # Testing & evaluation
│   ├── test_ai_quality.py     # AI quality test suite
│   ├── test_data/             # Test images
│   │   ├── known_artworks/    # Vision API tests
│   │   ├── RAG_images/        # RAG database tests
│   │   └── generic_images/    # Hallucination tests
│   └── results/               # Test results (JSON)
├── data/                      # Application data
│   ├── RAG_database/          # Artwork images for RAG
│   ├── image_hash_index.json # Perceptual hash index
│   └── test_paintings/        # Test artwork images
├── outputs/                   # Generated outputs
│   └── audio/                 # TTS audio files
└── logs/                      # Application logs
    └── app.log                # Main log file
```

---

## 🧪 Testing & Evaluation

### AI Quality Testing

**4-metric testing framework:**

1. **Accuracy**: 100% (6/6 artworks - 3 Vision API + 3 RAG)
2. **Hallucination Detection**: 0% (perfect rejection of non-artworks)
3. **Context Relevancy**: 100% (all chat responses stay on-topic)
4. **Performance**: Vision API ~14s, RAG <1s

**Run tests:**
```bash
python tests/test_ai_quality.py
```

**Full report:** [AI_EVALUATION_REPORT.md](AI_EVALUATION_REPORT.md)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Credits & Acknowledgments

**Developer:** Fee Pieper

**External APIs & Services:**
- [OpenAI](https://openai.com/) - GPT-4o-mini Vision API, Chat API, TTS API
- [Pinecone](https://www.pinecone.io/) - Vector database infrastructure

**Libraries & Frameworks:**
- [Gradio](https://www.gradio.app/) - Web interface framework
- [Pillow](https://python-pillow.org/) - Image processing library
- [ImageHash](https://github.com/JohannesBuchner/imagehash) - Perceptual hashing

**Documentation & Resources:**
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Pinecone Documentation](https://docs.pinecone.io)
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/) - Accessibility standards

---

**Built for accessible, interactive art education**

*Making museums accessible to everyone, one artwork at a time.*
