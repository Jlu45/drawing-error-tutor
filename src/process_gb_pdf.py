import os
import fitz
from rag_knowledge_base import DualKnowledgeBase

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"提取PDF文本失败: {e}")
        return ""

def split_text_into_chunks(text, chunk_size=1000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def process_gb_pdf():
    kb = DualKnowledgeBase()
    gb_standards_dir = 'data/gb_standards'
    for filename in os.listdir(gb_standards_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(gb_standards_dir, filename)
            print(f"处理文件: {filename}")
            text = extract_text_from_pdf(pdf_path)
            if text:
                chunks = split_text_into_chunks(text)
                for i, chunk in enumerate(chunks):
                    title = f"{filename} - 第{i+1}部分"
                    kb.add_text_knowledge(title, chunk, filename)
                print(f"成功处理 {filename}，提取了 {len(chunks)} 个文本块")
            else:
                print(f"处理 {filename} 失败，未提取到文本")

if __name__ == '__main__':
    try:
        import fitz
    except ImportError:
        print("正在安装PyMuPDF...")
        import subprocess
        subprocess.run(['pip', 'install', 'PyMuPDF'])
        import fitz
    process_gb_pdf()
