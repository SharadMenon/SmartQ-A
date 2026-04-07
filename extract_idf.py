import sys
from PyPDF2 import PdfReader

try:
    pdf_path = r'e:\Smart Document&VIdeo Q&A system\Invention_Disclosure_Form_IDF_A_B_Guide_to_Describing_Your_Innovation (1).pdf'
    reader = PdfReader(pdf_path)
    text = '\n'.join([p.extract_text() for p in reader.pages if p.extract_text()])
    
    with open('idf_extracted.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
