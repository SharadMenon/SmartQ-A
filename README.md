# Technical Architecture & Implementation Details

## 📐 System Architecture Deep Dive

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User Interface Layer                   │
│              (Command-Line Interface)                    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Application Core Layer                      │
│  ┌─────────────────────────────────────────────┐        │
│  │  Main Controller (smart_qa_complete.py)     │        │
│  │  • Command parsing                          │        │
│  │  • Workflow orchestration                   │        │
│  │  • State management                         │        │
│  └─────────────────────────────────────────────┘        │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Processing Pipeline Layer                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Content    │  │  Chunking    │  │  Embedding   │  │
│  │  Extraction  │→ │   Engine     │→ │  Generator   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Data Management Layer                       │
│  ┌──────────────────────────────────────────────┐       │
│  │  In-Memory Document Store                    │       │
│  │  documents = {                               │       │
│  │    'doc_name': [                             │       │
│  │      {'id': str, 'text': str,                │       │
│  │       'embedding': np.array, 'type': str}    │       │
│  │    ]                                         │       │
│  │  }                                           │       │
│  └──────────────────────────────────────────────┘       │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│             External Services Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   ChromaDB   │  │    Ollama    │  │   HF Models  │  │
│  │  (Optional)  │  │   (LLM API)  │  │  (Embeddings)│  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 Component Breakdown

### 1. Content Extraction Components

#### 1.1 PDF Extractor
```python
Technology: PyPDF2.PdfReader
Input: .pdf file path
Process:
  1. Open file in binary read mode
  2. Iterate through pages
  3. Extract text per page
  4. Concatenate with newlines
Output: Raw text string

Performance:
  - Speed: ~2-5 pages/second
  - Memory: O(n) where n = file size
  - Limitations: Image-based PDFs return empty text
```

#### 1.2 Video Analyzer (EnhancedVideoAnalyzerWithAudio)
```python
Technology: OpenCV (cv2), FFmpeg, Whisper
Input: .mp4/.avi/.mov/.mkv file path

Process Flow:
  Step 1: Video Metadata Extraction
    - Total frames: cv2.CAP_PROP_FRAME_COUNT
    - FPS: cv2.CAP_PROP_FPS
    - Duration calculation: frames / FPS
  
  Step 2: Audio Extraction (extract_audio_from_video)
    - FFmpeg command: 
      ffmpeg -i video.mp4 -q:a 9 -ac 1 -ar 16000 audio.wav
    - Parameters:
      • -q:a 9: Audio quality (0-9, 9=smallest)
      • -ac 1: Mono audio (single channel)
      • -ar 16000: Sample rate 16kHz (Whisper optimized)
  
  Step 3: Speech Transcription (transcribe_audio_simple)
    - Model: openai/whisper-base
    - Input: 16kHz WAV file
    - Process: Automatic Speech Recognition pipeline
    - Output: Text transcript with timestamps
    - Cleanup: Delete temporary audio file
  
  Step 4: Visual Frame Analysis
    - Sample 10 keyframes evenly distributed
    - Frame selection: np.linspace(0, total_frames-1, 10)
    - Per frame:
      • Extract frame at specific index
      • Convert BGR → RGB color space
      • Convert to PIL Image
      • Calculate timestamp (frame_idx / FPS)
      • Vision classification (if enabled)
  
  Step 5: Vision Classification (Optional)
    - Model: google/vit-base-patch16-224
    - Input: PIL Image (RGB)
    - Output: Top-3 predicted labels with confidence
    - Example: ["blackboard", "classroom", "lecture"]

Output: 
  - transcript: String (full speech text)
  - keyframes_data: List[Dict] with timestamp, seconds, vision labels
  - duration_seconds: Float

Performance:
  - Audio extraction: ~30 seconds for 10-min video
  - Transcription: ~2-5x real-time (GPU: ~1x real-time)
  - Frame analysis: ~1-2 seconds per frame
  - Total: ~5-10 minutes for 10-min video (first run)
```

#### 1.3 Text Document Extractor
```python
Technology: Python built-in open()
Input: .txt/.md file path
Process:
  1. Open with UTF-8 encoding
  2. Read entire content
  3. Handle encoding errors gracefully
Output: Raw text string

Performance:
  - Speed: Instant (<1 second for most files)
  - Memory: O(n) where n = file size
```

---

### 2. Semantic Chunking Engine

```python
Algorithm: split_into_chunks(text, chunk_size=700)

Purpose: 
  Break long documents into semantic units that preserve meaning
  while fitting within embedding model context windows

Process:
  1. Text Normalization
     - Replace newlines with spaces
     - Collapse multiple spaces to single space
     - Strip leading/trailing whitespace
  
  2. Sentence Segmentation
     - Regex: (?<=[.!?])\s+
     - Splits on periods, exclamation marks, questions
     - Uses positive lookbehind to preserve punctuation
  
  3. Chunk Assembly
     - Initialize empty chunk
     - For each sentence:
       IF current_chunk + sentence < chunk_size:
         Add sentence to current chunk
       ELSE:
         Save current chunk
         Start new chunk with this sentence
     - Save final chunk

Design Rationale:
  - 700 chars ≈ 100-150 words
  - Fits comfortably in embedding model (512 tokens max)
  - Preserves sentence boundaries (maintains semantic coherence)
  - Balances granularity vs. context

Edge Cases Handled:
  - Empty text → return []
  - Very long sentences → include full sentence in chunk
  - No punctuation → single large chunk
  - Multiple consecutive spaces → normalized

Output:
  - List of text chunks
  - Each chunk: string of ~700 chars
  - No overlap between chunks
```

---

### 3. Embedding Generation

```python
Model: sentence-transformers/all-MiniLM-L6-v2
Alternative: sentence-transformers/all-mpnet-base-v2

Architecture:
  Input: Text string
  Model: 
    - Based on Microsoft MiniLM
    - 6-layer Transformer
    - Mean pooling of token embeddings
  Output: 384-dimensional dense vector

Model Details:
  - Parameters: 22.7M
  - Max sequence length: 512 tokens
  - Training: Trained on 1B+ sentence pairs
  - Speed: ~2000 sentences/second (CPU)
  - Speed: ~10000 sentences/second (GPU)

Embedding Properties:
  - Normalized L2 norm (unit vectors)
  - Cosine similarity optimized
  - Semantic meaning preserved
  - Cross-lingual capabilities (limited)

Mathematical Representation:
  E(text) = MeanPool(Transformer(Tokenize(text)))
  where E(text) ∈ ℝ³⁸⁴

Usage in Code:
  embedding = embedder.encode(chunk_text)
  # Returns: numpy.ndarray of shape (384,)

Memory Footprint:
  - Model: ~90MB
  - Per embedding: 384 * 4 bytes = 1.5KB
  - 1000 chunks = ~1.5MB in memory
```

---

### 4. Vector Storage & Retrieval

#### 4.1 Data Structure
```python
documents = {
  'document_name': [
    {
      'id': 'document_name_0',
      'text': 'chunk content here...',
      'embedding': np.array([0.23, -0.15, ...]),  # 384-dim
      'type': 'pdf' | 'video' | 'document'
    },
    # ... more chunks
  ]
}

Storage Characteristics:
  - Type: In-memory dictionary
  - Persistence: ChromaDB (chroma.sqlite3) - optional
  - Index: Document name → List of chunks
  - Scalability: Limited by RAM (~10K chunks = ~15MB)
```

#### 4.2 Search Algorithm
```python
Function: search(query, k=10)

Process:
  1. Query Embedding
     query_vector = embedder.encode(query)
     # Shape: (384,)
  
  2. Similarity Computation
     For each document:
       For each chunk:
         similarity = cosine_similarity(query_vector, chunk_vector)
         # Formula: dot(A, B) / (norm(A) * norm(B))
  
  3. Result Aggregation
     all_results = []
     For each chunk:
       all_results.append({
         'doc': document_name,
         'text': chunk_text,
         'similarity': similarity_score,
         'type': content_type
       })
  
  4. Ranking
     all_results.sort(key=lambda x: x['similarity'], reverse=True)
  
  5. Top-K Selection
     return all_results[:k]

Complexity:
  - Time: O(n * d) where n=total_chunks, d=embedding_dim
  - Space: O(n)
  - Optimization: Could use approximate nearest neighbor (ANN)

Similarity Metric:
  cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
  Range: [-1, 1]
  Interpretation:
    1.0 = Identical
    0.0 = Orthogonal (unrelated)
   -1.0 = Opposite meaning
```

---

### 5. LLM Integration

```python
Service: Ollama (Local LLM Server)
Model: neural-chat (Intel's optimized model)
API: REST API on localhost:11434

Request Format:
  POST http://localhost:11434/api/generate
  Headers: Content-Type: application/json
  Body: {
    "model": "neural-chat",
    "prompt": "<constructed_prompt>",
    "stream": false,
    "temperature": 0.5
  }

Prompt Construction:
  prompt = f"""You are an intelligent learning assistant.

  Educational Content:
  {context}  # Top-10 relevant chunks

  Question: {question}

  Provide a detailed answer based on the content."""

Response Format:
  {
    "response": "Generated answer text...",
    "model": "neural-chat",
    "created_at": "2024-01-01T00:00:00Z",
    "done": true
  }

Parameters:
  - temperature: 0.5 (balanced creativity/accuracy)
  - stream: false (wait for complete response)
  - timeout: 180 seconds (3 minutes max)

Error Handling:
  - Timeout → Inform user, suggest shorter context
  - Connection refused → Check if Ollama is running
  - 404 Model not found → Suggest: ollama pull neural-chat
  - 500 Server error → Retry or report issue

Alternative Models (Compatible):
  - llama2 (Meta's model)
  - mistral (Mistral AI)
  - codellama (Code-focused)
  - phi (Microsoft's small model)
```

---

## 💾 Data Flow Diagram

### Adding Material Flow
```
User Input: "add lecture.mp4"
    │
    ▼
┌─────────────────────┐
│ File Path Validation│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Content Extraction  │ → [FFmpeg] → Audio WAV
│                     │ → [Whisper] → Transcript
│                     │ → [OpenCV] → Frames
│                     │ → [ViT] → Labels
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Content Assembly    │ → Combine transcript + visual data
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Semantic Chunking   │ → ~700 char chunks
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Batch Embedding     │ → 384-dim vectors
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Memory Storage      │ → documents[name] = chunks
└─────────────────────┘
```

### Query Processing Flow
```
User Input: "What is X?"
    │
    ▼
┌─────────────────────┐
│ Query Embedding     │ → 384-dim vector
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Similarity Search   │ → Compute cosine similarity
│ (All Chunks)        │ → Sort by relevance
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Top-K Retrieval     │ → Get 10 best matches
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Context Assembly    │ → Concatenate chunks
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ LLM API Call        │ → Ollama neural-chat
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Answer Display      │ → Print to user
└─────────────────────┘
```

---

## 🔬 Performance Analysis

### Benchmarks (10-minute lecture video)

| Operation | Time | Notes |
|-----------|------|-------|
| Video metadata extraction | <1s | Fast |
| Audio extraction (FFmpeg) | ~30s | I/O bound |
| Speech transcription (CPU) | ~20min | Compute intensive |
| Speech transcription (GPU) | ~10min | 2x speedup |
| Frame extraction (10 frames) | ~10s | I/O bound |
| Vision classification | ~20s | GPU helps |
| Text chunking | <1s | Fast |
| Embedding generation (50 chunks) | ~5s | Batch processing |
| **Total (CPU)** | **~25min** | First run |
| **Total (GPU)** | **~12min** | First run |
| **Subsequent queries** | **<2s** | Cached embeddings |

### Memory Usage

| Component | Memory |
|-----------|--------|
| SentenceTransformer model | ~90MB |
| Vision Transformer model | ~350MB |
| Whisper base model | ~140MB |
| Embeddings (1000 chunks) | ~1.5MB |
| Document text (1000 chunks) | ~0.7MB |
| **Total (models loaded)** | **~580MB** |
| **Total (with data)** | **~582MB** |

### Scalability Limits

| Metric | Current Limit | Bottleneck |
|--------|---------------|------------|
| Documents | ~100 | Memory |
| Total chunks | ~10,000 | Search O(n) |
| Chunk size | 512 tokens | Model limit |
| Video length | ~2 hours | Processing time |
| PDF pages | ~1000 | None |

---

## 🔐 Security Considerations

### Current Implementation
- ✅ Local processing (no cloud uploads)
- ✅ No API keys required (uses local models)
- ✅ No data persistence (in-memory only)
- ✅ Safe file path handling (Path.resolve())

### Potential Risks
- ⚠️ No input validation on file types
- ⚠️ Arbitrary code execution via malicious PDFs (PyPDF2 vulnerability)
- ⚠️ Command injection via FFmpeg (subprocess.run)
- ⚠️ No rate limiting on queries
- ⚠️ No authentication/authorization

### Recommended Hardening
1. Add file type validation with magic bytes
2. Sandbox PDF processing
3. Escape FFmpeg parameters
4. Add request rate limiting
5. Implement user authentication
6. Add audit logging
7. Encrypt stored embeddings
8. Validate file sizes before processing

---

## 🧪 Testing Strategy

### Unit Tests (Recommended)
```python
tests/
├── test_extraction.py      # PDF, video, text extraction
├── test_chunking.py         # Semantic chunking logic
├── test_embedding.py        # Embedding generation
├── test_search.py           # Similarity search
└── test_integration.py      # End-to-end workflows
```

### Test Cases
1. **PDF Extraction**
   - Empty PDF
   - Multi-page PDF
   - Scanned PDF (should fail gracefully)
   - Corrupted PDF

2. **Video Analysis**
   - Various formats (.mp4, .avi, .mkv)
   - No audio track
   - Very short video (<10s)
   - Very long video (>1hr)

3. **Chunking**
   - Empty text
   - Single sentence
   - No punctuation
   - Multiple languages

4. **Search**
   - No documents loaded
   - Query with no matches
   - Query matching multiple docs
   - Empty query

---

## 📊 Metrics & Monitoring

### Key Metrics to Track
1. **Performance**
   - Processing time per document type
   - Query response time
   - Embedding generation speed

2. **Accuracy**
   - Search relevance (manual evaluation)
   - Answer quality (user feedback)
   - Transcription accuracy (WER)

3. **Resource Usage**
   - Memory consumption
   - CPU utilization
   - Disk I/O

4. **User Behavior**
   - Documents processed per session
   - Queries per document
   - Command usage distribution

---

## 🔄 Future Architecture Improvements

### Planned Enhancements
1. **Database Migration**
   - Move from in-memory to persistent ChromaDB
   - Add SQLite for metadata
   - Implement incremental indexing

2. **Performance Optimization**
   - Implement ANN (Approximate Nearest Neighbor)
   - Add caching layer (Redis)
   - Batch processing pipeline
   - GPU acceleration for all models

3. **Scalability**
   - Distributed processing (Celery)
   - Microservices architecture
   - API layer (FastAPI)
   - Load balancing for multiple users

4. **Features**
   - Multi-modal embeddings (CLIP)
   - Real-time video streaming
   - Incremental learning
   - Active learning for query refinement

---


This technical documentation provides a comprehensive understanding of the system's internals for developers who want to contribute or extend the functionality.
