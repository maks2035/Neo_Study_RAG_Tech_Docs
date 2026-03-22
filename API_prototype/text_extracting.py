import json
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_FILE = "6100.00.0.000-RE_122_.pdf"
OUTPUT_FILE = "chunks_data.json"

CHUNK_SIZE = 300      
CHUNK_OVERLAP = 50     

def process_pdf_to_chunks(pdf_path, output_path, chunk_size, chunk_overlap):
   if not os.path.exists(pdf_path):
        print(f"Ошибка: Файл не найден: {pdf_path}")
        return False
   
   try:
      loader = PyPDFLoader(pdf_path)
      pages = loader.load()
   except Exception as e:
      print(f"Ошибка при загрузке PDF: {e}")
      return False


   if not pages:
      print("Предупреждение: Документ пуст или не содержит текста.")
      return False
   
   text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
   chunks = text_splitter.split_documents(pages)

   processed_chunks = []

   for i, chunk in enumerate(chunks):
      meta = chunk.metadata
      
      source_name = meta.get("source", "unknown.pdf")
      
      
      page_num_raw = meta.get("page_label", 0)
      page_label = page_num_raw
      
      clean_meta = {
         "source": source_name,
         "page": page_label,
      }

      chunk_data = {
         "id": i,                  
         "text": chunk.page_content,
         "metadata": clean_meta
      }
      
      processed_chunks.append(chunk_data)

   try:
      with open(output_path, "w", encoding="utf-8") as f:
         json.dump(processed_chunks, f, ensure_ascii=False, indent=2)
         return True

   except Exception as e:
        print(f"Ошибка при сохранении файла: {e}")
        return False
   
 
success = process_pdf_to_chunks(pdf_path=PDF_FILE,output_path=OUTPUT_FILE,chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP)

print(success)
