# 🧠 SmartQ-A — Smart Document & Video Q&A Assistant

> An AI-powered Retrieval-Augmented Generation (RAG) assistant that lets you upload PDFs, text documents, and educational videos — then ask natural language questions and get intelligent, contextually grounded answers.

---

## 📸 Overview

SmartQ-A combines a modern Streamlit web UI with a powerful backend pipeline that can ingest multiple document types, transcribe speech from videos, extract visual keyframes, and answer questions using a locally running LLM. It supports multi-document sessions with intelligent media-type routing so you can ask "summarize the video" or "what does the PDF say about X" and get the right context back every time.

---

## ✨ Features

- **Multi-format ingestion** — Upload PDFs, plain text/markdown files, and video files (`.mp4`, `.avi`, `.mov`, `.mkv`)
- **Video speech transcription** — Automatically extracts audio via FFmpeg and transcribes it using OpenAI Whisper
- **Visual keyframe analysis** — Samples 10 evenly spaced frames and classifies visual content using Google's Vision Transformer (ViT)
- **Semantic chunking with overlap** — Splits content into overlapping ~800-character chunks with 200-character context carry-over for better RAG recall
- **Dense vector search** — Embeds all content using `sentence-transformers/all-MiniLM-L6-v2` (384-dim) and retrieves top-16 chunks via cosine similarity
- **Media-type routing** — Automatically detects whether a query is asking about a video, PDF, or both, and routes context accordingly
- **LLM-powered answers** — Sends retrieved context to a locally running Ollama instance (`llama3` model) for grounded, hallucination-resistant responses
- **Persistent session chat** — Full chat history maintained across queries within a session
- **Multi-document awareness** — Groups retrieved chunks by source document to prevent cross-document answer contamination

---

## 🗂️ Project Structure

```
SmartQ-A-main/
├── app_gui.py              # Streamlit web UI (entry point)
├── smart_qa_complete.py    # Core backend: ingestion, chunking, embedding, search, LLM
├── extract_idf.py          # Utility for extracting content metadata
├── test_routing.py         # Tests for media-type query routing logic
├── requirements.txt        # All Python dependencies
└── README.md               # This file
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Streamlit Web UI (app_gui.py)            │
│   Sidebar: File Upload & Status  │  Main: Chat Interface  │
└───────────────────┬──────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────┐
│              Core Backend (smart_qa_complete.py)          │
│                                                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  Ingestion  │  │  Chunking &  │  │  Semantic Search│  │
│  │  Pipeline   │→ │  Embedding   │→ │  + LLM Q&A      │  │
│  └─────────────┘  └──────────────┘  └─────────────────┘  │
│        │                                                  │
│  ┌─────▼──────────────────────────────────────────────┐  │
│  │          In-Memory Vector Store (documents {})      │  │
│  └────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────┘
                    │                          │
      ┌─────────────▼──────┐      ┌────────────▼───────────┐
      │  HuggingFace Models│      │  Ollama (Local LLM)    │
      │  • all-MiniLM-L6-v2│      │  • llama3              │
      │  • ViT (vision)    │      │  localhost:11434       │
      │  • Whisper (audio) │      └────────────────────────┘
      └────────────────────┘
```

### Ingestion Pipeline (per file type)

| File Type | Extraction Method | Output |
|-----------|------------------|--------|
| `.pdf` | PyPDF2 page-by-page text extraction | Raw text string |
| `.txt` / `.md` | Python `open()` with UTF-8 | Raw text string |
| `.mp4` / `.avi` / `.mov` / `.mkv` | FFmpeg → WAV → Whisper transcription + OpenCV frames + ViT vision | Combined transcript + visual labels |

### Search & Routing

The search function (`search()`) performs global cosine similarity ranking across all loaded chunks, then applies media-type routing based on keywords detected in the query:

- `"video"` only → returns only video-type chunks
- `"pdf"` / `"document"` only → returns only PDF/document-type chunks  
- `"both"` / `"all"` or explicit mix → balanced sampling across all loaded documents
- No keyword → returns globally top-ranked chunks regardless of type

---

## ⚙️ Prerequisites

### System Dependencies

- **Python 3.9+**
- **[FFmpeg](https://ffmpeg.org/download.html)** — Required for video audio extraction
- **[Ollama](https://ollama.com/)** — Required to run the local LLM

### Install FFmpeg

**Windows:** Download from https://ffmpeg.org/download.html and add to PATH  
**macOS:** `brew install ffmpeg`  
**Linux:** `sudo apt install ffmpeg`

### Install & Start Ollama

```bash
# Install Ollama from https://ollama.com/
# Pull the llama3 model
ollama pull llama3

# Start the Ollama server (runs on localhost:11434)
ollama serve
```

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/SmartQ-A.git
cd SmartQ-A
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download AI models (~600MB total): `all-MiniLM-L6-v2`, `google/vit-base-patch16-224`, and `openai/whisper-base`. This only happens once.

### 4. (Optional) Install Whisper for video transcription

```bash
pip install openai-whisper
```

Without Whisper, videos will still be processed using visual frame analysis only — speech will not be transcribed.

---

## ▶️ Running the App

Make sure Ollama is running (`ollama serve`) in a separate terminal, then:

```bash
streamlit run app_gui.py
```

The app opens in your browser at `http://localhost:8501`.

---

## 💡 Usage

### Uploading Documents

1. Use the **sidebar** to choose a file (PDF, TXT, MD, or video)
2. Click **"Process File"** — the system will extract, chunk, and embed all content
3. The sidebar shows all loaded materials with their chunk counts
4. You can load **multiple documents** in the same session

### Asking Questions

Type any question in the chat input at the bottom. Examples:

- `"Summarize the key points from the video"`
- `"What does the PDF say about machine learning?"`
- `"Generate 5 multiple choice questions based on both files"`
- `"Create a comparison table of concepts from the video and the document"`
- `"Explain the main topic covered in all uploaded materials"`

### Query Tips

| Goal | Example Query |
|------|--------------|
| Ask about video content only | `"What topics were discussed in the video?"` |
| Ask about PDF content only | `"What does the document cover about neural networks?"` |
| Cross-document summary | `"Create a combined summary of both the video and the PDF"` |
| Generate MCQs | `"Generate 5 MCQs based on both files with an answer key at the end"` |
| Comparison | `"Create a 3-column markdown table comparing concepts from both sources"` |

---

## 🔧 Configuration

Key parameters can be adjusted in `smart_qa_complete.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | `800` chars | Target size of each semantic chunk |
| `overlap_size` | `200` chars | Context overlap between adjacent chunks |
| `k` (search) | `16` | Number of top chunks retrieved per query |
| Ollama model | `llama3` | Change to `llama2`, `mistral`, `phi`, etc. |
| Embedding model | `all-MiniLM-L6-v2` | Fallback: `all-mpnet-base-v2` |
| LLM temperature | `0.5` | Lower = more factual, Higher = more creative |
| LLM timeout | `180s` | Increase for very long documents |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `sentence-transformers` | Dense text embeddings (`all-MiniLM-L6-v2`) |
| `transformers` | Vision Transformer (ViT) for frame classification |
| `PyPDF2` | PDF text extraction |
| `opencv-python` | Video frame extraction |
| `openai-whisper` | Speech-to-text transcription |
| `torch` | Deep learning backend |
| `numpy` / `scipy` | Numerical operations, cosine similarity |
| `requests` | HTTP calls to Ollama REST API |
| `Pillow` | Image handling for ViT input |
| `chromadb` | Optional persistent vector store |

---

## 🤖 Compatible LLM Models (via Ollama)

Any model available in Ollama can be used. To switch, change `"llama3"` in `smart_qa_complete.py`:

```python
"model": "mistral"   # or "llama2", "phi", "codellama", "gemma", etc.
```

Pull any model with: `ollama pull <model-name>`

---

## ⚡ Performance Notes

| Task | Approximate Time |
|------|-----------------|
| PDF ingestion (50 pages) | ~10–30 seconds |
| Video ingestion — 10 min (with Whisper, CPU) | ~15–25 minutes |
| Video ingestion — 10 min (with Whisper, GPU) | ~5–10 minutes |
| Query response | ~5–30 seconds (depends on LLM + context size) |
| Subsequent queries (same session) | Fast — embeddings are cached in memory |

---

## 🛠️ Troubleshooting

**"Please upload and process a document first"** — No file has been ingested. Use the sidebar to upload and click Process File.

**"Sorry, I couldn't generate an answer. Make sure Ollama is running."** — Start Ollama with `ollama serve` in a separate terminal and verify it's at `localhost:11434`.

**Video has no speech transcription** — Install Whisper: `pip install openai-whisper`. Also ensure FFmpeg is installed and on your system PATH.

**Very slow on first run** — Model downloads (~600MB) happen once on first launch. Subsequent runs are faster.

**Out of memory on large videos** — Reduce `chunk_size` or process shorter video segments.

---

## 🔮 Roadmap

- [ ] Persistent vector storage with ChromaDB across sessions
- [ ] Support for `.docx` / `.pptx` document formats
- [ ] Streaming LLM responses in the chat UI
- [ ] GPU-accelerated embedding generation
- [ ] Multi-user session isolation
- [ ] Export chat history as PDF/markdown
- [ ] REST API layer (FastAPI) for programmatic access

---

## 📄 License

This project is open source. See `LICENSE` for details.
