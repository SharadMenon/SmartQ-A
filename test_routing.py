import os
import sys

# Add path if necessary
sys.path.append(os.getcwd())

from smart_qa_complete import documents, search, embedder, cosine_similarity

# 1. Mock the documents
try:
    documents['test_video'] = [
        {'text': f'Video chunk {i} about machine learning', 'embedding': embedder.encode(f'Video chunk {i} about machine learning'), 'type': 'video'}
        for i in range(15)
    ]
    documents['test_pdf'] = [
        {'text': f'PDF chunk {i} about data science', 'embedding': embedder.encode(f'PDF chunk {i} about data science'), 'type': 'pdf'}
        for i in range(15)
    ]
except Exception as e:
    print(f"Error mocking: {e}")

print("=== IDF FORM B: VALIDATION TEST SUITE ===")

q1 = "what is the video about?"
res1 = search(q1, k=16)
video_count = sum(1 for r in res1 if r['type'] == 'video')
pdf_count = sum(1 for r in res1 if r['type'] == 'pdf')
print(f"Query: '{q1}'")
print(f"Results: {video_count} Video Chunks | {pdf_count} PDF Chunks")
print(f"Routing Success: {'PASS' if video_count > 0 and pdf_count == 0 else 'FAIL'}")
print("-" * 30)

q2 = "what is the document talking about?"
res2 = search(q2, k=16)
video_count2 = sum(1 for r in res2 if r['type'] == 'video')
pdf_count2 = sum(1 for r in res2 if r['type'] == 'pdf')
print(f"Query: '{q2}'")
print(f"Results: {video_count2} Video Chunks | {pdf_count2} PDF Chunks")
print(f"Routing Success: {'PASS' if video_count2 == 0 and pdf_count2 > 0 else 'FAIL'}")
print("-" * 30)

q3 = "compare the concepts in both the video and the pdf"
res3 = search(q3, k=16)
video_count3 = sum(1 for r in res3 if r['type'] == 'video')
pdf_count3 = sum(1 for r in res3 if r['type'] == 'pdf')
print(f"Query: '{q3}'")
print(f"Results: {video_count3} Video Chunks | {pdf_count3} PDF Chunks")
print(f"Routing Success: {'PASS' if video_count3 == 8 and pdf_count3 == 8 else 'FAIL'}")
print("-" * 30)
