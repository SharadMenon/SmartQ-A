#!/usr/bin/env python3
"""
ULTIMATE KNOWLEDGE ASSISTANT - WITH AUDIO EXTRACTION
Now extracts BOTH video frames AND audio/speech content
Understands lectures by transcribing what the teacher says
"""

import os
from pathlib import Path
import re
import requests
import numpy as np
import cv2
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from sentence_transformers import SentenceTransformer
    import PyPDF2
    from PIL import Image
except:
    print("Installing packages...")
    os.system("pip install --no-deps sentence-transformers PyPDF2 numpy opencv-python requests Pillow -q > NUL 2>&1")
    os.system("pip install transformers librosa scipy -q > NUL 2>&1")
    from sentence_transformers import SentenceTransformer
    import PyPDF2
    from PIL import Image

print("\n" + "="*70)
print("ULTIMATE KNOWLEDGE ASSISTANT - WITH AUDIO EXTRACTION")
print("Now understands lectures through speech recognition")
print("="*70 + "\n")

print("[INIT] Loading models...")
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("[EMBED] ✓ Embedding model ready")
except Exception as e:
    embedder = SentenceTransformer("all-mpnet-base-v2")
    print("[EMBED] ✓ Alternative model ready")

try:
    from transformers import pipeline
    image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    print("[VISION] ✓ Image recognition ready")
    VISION_AVAILABLE = True
except Exception as e:
    print(f"[VISION] Using fallback")
    VISION_AVAILABLE = False
    image_classifier = None

# Try to load speech recognition
try:
    import librosa
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
    print("[SPEECH] ✓ Speech recognition ready")
except:
    print("[SPEECH] ⚠ Speech recognition not available (will use FFmpeg method)")
    SPEECH_AVAILABLE = False

print("[INIT] ✓ Ready\n")

documents = {}

# ======================= ENHANCED VIDEO ANALYZER WITH AUDIO =======================

class EnhancedVideoAnalyzerWithAudio:
    """
    Analyzes videos to extract:
    1. Visual content (frames, objects)
    2. Audio content (speech, sound)
    3. Speech transcription (what teacher says)
    """
    
    def __init__(self):
        self.vision_model = image_classifier
    
    def extract_audio_from_video(self, video_path):
        """Extract audio track from video file"""
        try:
            import subprocess
            audio_path = video_path.replace('.mp4', '_audio.wav').replace('.avi', '_audio.wav').replace('.mov', '_audio.wav').replace('.mkv', '_audio.wav')
            
            print(f"  [AUDIO] Extracting audio from video...")
            
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg', '-i', video_path, '-q:a', '9', '-n', '-ac', '1', '-ar', '16000', audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            
            if os.path.exists(audio_path):
                print(f"  [AUDIO] ✓ Audio extracted ({os.path.getsize(audio_path) / 1024 / 1024:.1f}MB)")
                return audio_path
        except Exception as e:
            print(f"  [AUDIO] Note: FFmpeg not installed - will use OpenCV audio (limited)")
            return None
    
    def transcribe_audio_simple(self, audio_path):
        """
        Transcribe audio using native OpenAI Whisper library directly
        Fallback: Use generic descriptions if transcription unavailable
        """
        try:
            import whisper
            import torch
            import warnings
            print(f"  [TRANSCRIBE] Loading native Whisper model...")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                whisper_model = whisper.load_model("base", device=device)
            
            print(f"  [TRANSCRIBE] Transcribing audio (this may take a few minutes)...")
            
            result = whisper_model.transcribe(audio_path)
            transcript = result["text"]
            
            print(f"  [TRANSCRIBE] ✓ Transcription complete ({len(transcript)} characters)")
            return transcript
        
        except Exception as e:
            import traceback
            print(f"  [TRANSCRIBE] Error during transcription: {e}")
            traceback.print_exc()
            return None
    
    def extract_educational_content_with_audio(self, video_path):
        """Extract complete educational content from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration_seconds = total_frames / fps if fps > 0 else 0
            
            print(f"  Video: {total_frames} frames @ {fps:.1f} fps")
            print(f"  Duration: {int(duration_seconds // 60)}:{int(duration_seconds % 60):02d}")
            
            # Step 1: Extract audio
            audio_path = self.extract_audio_from_video(video_path)
            
            # Step 2: Transcribe audio
            transcript = None
            if audio_path:
                print(f"  [TRANSCRIBE] Extracting speech content...")
                transcript = self.transcribe_audio_simple(audio_path)
                
                # Cleanup
                try:
                    os.remove(audio_path)
                except:
                    pass
            
            # Step 3: Extract visual content
            print(f"  [VISUAL] Extracting key visual moments...")
            
            keyframes_data = []
            frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                seconds = idx / fps if fps > 0 else 0
                minutes = int(seconds // 60)
                secs = int(seconds % 60)
                timestamp = f"{minutes:02d}:{secs:02d}"
                
                vision_labels = None
                if VISION_AVAILABLE:
                    try:
                        vision_labels = self.vision_model(frame_pil, top_k=3)
                        vision_labels = [r['label'] for r in vision_labels]
                    except:
                        pass
                
                keyframes_data.append({
                    'timestamp': timestamp,
                    'seconds': seconds,
                    'vision': vision_labels
                })
            
            cap.release()
            
            return transcript, keyframes_data, duration_seconds
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None, [], 0

# ======================= SEMANTIC CHUNKING =======================

def split_into_chunks(text, chunk_size=800, overlap_size=200):
    """Split content into overlapping semantic chunks for maximum RAG recall"""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return []
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_sentences = []
    current_length = 0
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
            
        current_sentences.append(sent)
        current_length += len(sent) + 1
        
        if current_length >= chunk_size:
            chunk_text = " ".join(current_sentences)
            chunks.append(chunk_text)
            
            # Rolling window overlap (retain sentences at the end of the chunk to carry over context)
            overlap_length = 0
            overlap_sentences = []
            for s in reversed(current_sentences):
                # We need at least one sentence to trickle forward to prevent infinite loops if sentence > overlap_size
                if overlap_length + len(s) > overlap_size and overlap_sentences:
                    break
                overlap_sentences.insert(0, s)
                overlap_length += len(s) + 1
                
            current_sentences = overlap_sentences
            current_length = overlap_length
            
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        if chunk_text.strip() and chunk_text not in chunks:
            chunks.append(chunk_text.strip())
            
    return chunks

# ======================= DOCUMENT EXTRACTION =======================

def extract_pdf_text(pdf_path):
    """Extract text from PDF"""
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            from PyPDF2 import PdfReader
            reader = PdfReader(f)
            total_pages = len(reader.pages)
            print(f"  PDF: {total_pages} pages")
            
            for i, page in enumerate(reader.pages):
                text += page.extract_text() + "\n"
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{total_pages} pages...")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return ""
    
    return text

def extract_text_file(text_path):
    """Extract from text/document file"""
    try:
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return text
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return ""

# ======================= SEARCH & SIMILARITY =======================

def cosine_similarity(a, b):
    """Calculate cosine similarity"""
    a = np.array(a)
    b = np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0
    return np.dot(a, b) / denom

def search(query, k=16):
    """Dynamic Semantic Search with Media Type Routing"""
    if not documents:
        return []
    
    query_embedding = embedder.encode(query)
    all_chunks = []
    
    for doc_name, chunks in documents.items():
        for chunk in chunks:
            all_chunks.append({
                'doc': doc_name,
                'text': chunk['text'],
                'embedding': chunk['embedding'],
                'type': chunk.get('type', '')
            })
            
    # Score all chunks globally
    for c in all_chunks:
        c['similarity'] = cosine_similarity(query_embedding, c['embedding'])
        
    all_chunks.sort(key=lambda x: x['similarity'], reverse=True)
    
    query_lower = query.lower()
    
    wants_video = "video" in query_lower and "pdf" not in query_lower and "document" not in query_lower
    wants_pdf = ("pdf" in query_lower or "document" in query_lower) and "video" not in query_lower
    wants_both = "both" in query_lower or "all" in query_lower or ("video" in query_lower and ("pdf" in query_lower or "document" in query_lower))
    
    final_results = []
    
    if wants_video:
        final_results = [c for c in all_chunks if c['type'] == 'video']
    elif wants_pdf:
        final_results = [c for c in all_chunks if c['type'] in ['pdf', 'document']]
    elif wants_both and len(documents) > 1:
        k_per_doc = max(1, k // len(documents))
        doc_counts = {d: 0 for d in documents.keys()}
        for c in all_chunks:
            if doc_counts[c['doc']] < k_per_doc:
                final_results.append(c)
                doc_counts[c['doc']] += 1
            if len(final_results) >= k:
                break
    else:
        final_results = all_chunks
        
    return final_results[:k]

# ======================= MAIN OPERATIONS =======================

def add_material(file_path):
    """Add material with audio extraction for videos"""
    file_path = str(Path(file_path).resolve())
    
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}\n")
        return
    
    doc_name = Path(file_path).stem
    ext = Path(file_path).suffix.lower()
    
    print(f"\n[ADD] {Path(file_path).name}")
    
    analyzer = EnhancedVideoAnalyzerWithAudio()
    text = ""
    content_type = ""
    
    if ext == '.pdf':
        print(f"  Type: PDF Document")
        text = extract_pdf_text(file_path)
        content_type = "pdf"
    
    elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']:
        print(f"  Type: Educational Video")
        
        # Extract audio AND video
        transcript, keyframes_data, duration = analyzer.extract_educational_content_with_audio(file_path)
        
        # Build comprehensive content
        content_parts = []
        
        # Add transcript if available
        if transcript:
            print(f"  [CONTENT] ✓ Speech transcription available ({len(transcript)} chars)")
            content_parts.append(f"[VIDEO TRANSCRIPT]\n{transcript}\n")
        else:
            print(f"  [CONTENT] ⚠ No speech transcription (install: pip install openai-whisper)")
        
        # Add visual descriptions
        if keyframes_data:
            visual_desc = f"\n[VIDEO VISUAL CONTENT]\nDuration: {int(duration//60)}:{int(duration%60):02d}\nKey Moments:\n"
            for frame in keyframes_data:
                if frame['vision']:
                    visual_desc += f"  [{frame['timestamp']}] {', '.join(frame['vision'][:3])}\n"
            content_parts.append(visual_desc)
        
        text = "\n".join(content_parts)
        content_type = "video"
    
    elif ext in ['.txt', '.md', '.doc', '.docx']:
        print(f"  Type: Document/Text")
        text = extract_text_file(file_path)
        content_type = "document"
    
    else:
        print(f"✗ Unsupported: {ext}")
        return
    
    if not text.strip():
        print("✗ No content extracted\n")
        return
    
    print(f"  Content size: {len(text)} characters")
    
    print(f"[CHUNK] Creating chunks...")
    chunks_list = split_into_chunks(text)
    
    if not chunks_list:
        print("✗ No chunks\n")
        return
    
    print(f"  ✓ {len(chunks_list)} sections")
    
    print(f"[EMBED] Generating embeddings...")
    doc_chunks = []
    
    for i, chunk_content in enumerate(chunks_list):
        try:
            embedding = embedder.encode(chunk_content)
            doc_chunks.append({
                'id': f"{doc_name}_{i}",
                'text': chunk_content,
                'embedding': embedding,
                'type': content_type
            })
            
            if (i + 1) % 40 == 0:
                print(f"  {i+1}/{len(chunks_list)}...")
        except:
            continue
    
    documents[doc_name] = doc_chunks
    
    print(f"\n✓ ADDED!")
    print(f"  Sections: {len(doc_chunks)}\n")
    return True

def ask_question(question):
    """Ask question"""
    print(f"\n[Q] {question}")
    
    if not documents:
        print("✗ No materials loaded!\n")
        return None
    
    total_chunks = sum(len(c) for c in documents.values())
    print(f"[SEARCH] Searching {total_chunks} sections...")
    
    results = search(question, k=16)
    
    if not results:
        print("✗ No results\n")
        return None
    
    print(f"[SEARCH] ✓ {len(results)} results")
    
    # Group contexts by document to prevent LLM crosstalk hallucination
    doc_groups = {}
    for r in results:
        doc_groups.setdefault(r['doc'], []).append(r['text'])
        
    context_parts = []
    for doc_name, texts in doc_groups.items():
        context_parts.append(f"--- SOURCE DOCUMENT: {doc_name} ---")
        context_parts.append("\n".join(texts))
        
    context = "\n\n".join(context_parts)
    
    print(f"[ASSISTANT] Analyzing...\n")
    
    system_prompt = """You are a professional, intelligent learning assistant analyzing educational materials.
You have been provided with excerpts from one or more distinct educational documents (videos, PDFs, etc.).
1. Use the provided educational content to directly fulfill the student's request accurately.
2. If they ask for specific formats (like MCQs, summaries, or bullet points), follow their instructions exactly.
3. NEVER mix or confuse the contents of different documents. Always clearly indicate which document a piece of information belongs to. Keep facts strictly separate based on their SOURCE DOCUMENT tags.
4. IMPORTANT: ALL 'attachments', 'uploaded files', 'videos', and 'pdfs' the user asks about are exactly the contents provided in the 'Educational Content' block below. NEVER say you cannot see or interpret files; answer using the provided text. 
5. Adapt the length, depth, and format of your response based strictly on the user's request. If the user asks for a brief summary, be concise. If they ask for a detailed explanation, provide comprehensive depth. Be flexible and intuitive."""
    
    prompt = f"""Educational Content:
{context}

Student Request: {question}"""
    
    try:
        # You can change 'llama3' to 'mistral' or 'neural-chat' depending on your hardware!
        # Remember to run `ollama pull llama3` in your terminal if you don't have it installed yet.
        ai_model = "llama3"
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": ai_model,
                "system": system_prompt,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.5,
                "options": {
                    "num_ctx": 8192
                }
            },
            timeout=180
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('response', '').strip()
            
            if answer:
                print(f"[ANSWER]\n{answer}\n")
                return answer
            else:
                print("✗ Empty\n")
                return None
        else:
            print(f"✗ Error\n")
            return None
    
    except requests.Timeout:
        print("✗ Timeout\n")
        return None
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return None

def show_status():
    """Show status"""
    if not documents:
        print("\n[STATUS] No materials\n")
        return
    
    print(f"\n[STATUS] Loaded ({len(documents)}):")
    total = 0
    for doc_name, chunks in documents.items():
        doc_type = chunks[0]['type'] if chunks else '?'
        print(f"  ✓ {doc_name} ({doc_type}): {len(chunks)}")
        total += len(chunks)
    
    print(f"\nTotal: {total} sections\n")

# ======================= MAIN =======================

if __name__ == "__main__":
    print("COMMANDS:")
    print("  add <file>  - Add video/PDF/document")
    print("  ask <q>     - Ask question")
    print("  status      - Show materials")
    print("  help        - Help")
    print("  exit        - Quit\n")

    while True:
        try:
            cmd = input("> ").strip()
            
            if not cmd:
                continue
            
            if cmd.lower() == "exit":
                print("Bye!\n")
                break
            
            elif cmd.lower().startswith("add "):
                file_path = cmd[4:].strip()
                if file_path:
                    add_material(file_path)
            
            elif cmd.lower().startswith("ask "):
                question = cmd[4:].strip()
                if question:
                    ask_question(question)
            
            elif cmd.lower() == "status":
                show_status()
            
            elif cmd.lower() == "help":
                print("""
COMMANDS:
  add <file>  - Add video/PDF/document (now with audio extraction!)
  ask <q>     - Ask questions
  status      - Show materials
  exit        - Quit

SETUP FOR VIDEO AUDIO EXTRACTION:
  1. Install FFmpeg: https://ffmpeg.org/download.html
  2. Install Whisper: pip install openai-whisper
  3. Then videos will extract speech automatically
""")
            
            else:
                print("Unknown\n")
        
        except KeyboardInterrupt:
            print("\nBye!\n")
            break
        except Exception as e:
            print(f"Error: {e}\n")