import json
import time
from openai import OpenAI

from API_prototype import (
   load_api_key,
   load_chunks,
   chunks_to_documents,
   create_ensemble_retriever,
   rag_query,
   NEO_MODEL_GPT,
   NEO_URL,
   FILE_WITH_CHUNKS
)

def test_retriver(retriever, input_file="test_questions.json", 
                      output_json="results_test_for_retiver.json"):
   questions = load_chunks(input_file)

   results = []

   for q_entry in questions:
      question_text = q_entry.get("question")
      chunk_id = q_entry.get("chunk_id")

      try:
            # Вызываем retriever напрямую
            retrieved_docs = retriever.invoke(question_text)
            
            # Формируем список источников в удобном формате
            sources = []
            for doc in retrieved_docs:
                sources.append({
                    "chunk_id": doc.metadata.get("chunk_id") if hasattr(doc, 'metadata') else None,
                    "page": doc.metadata.get("page") if hasattr(doc, 'metadata') else None,
                    "source": doc.metadata.get("source") if hasattr(doc, 'metadata') else None,
                    "text": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                })
            
            # Формируем итоговую запись
            result_entry = {
                "question": {
                    "chunk_id": chunk_id,
                    "text": question_text
                },
                "retrieved_chunks": {
                    "count": len(sources),
                    "items": sources
                }
            }
            results.append(result_entry)

      except Exception as e:
         print(f"шибка при retrieval: {e}")
         results.append({
               "question": {"chunk_id": chunk_id, "text": question_text},
               "retrieved_chunks": {"count": 0, "items": [], "error": str(e)},
         })

      with open(output_json, 'w', encoding='utf-8') as f:
         json.dump(results, f, ensure_ascii=False, indent=2)

def test_rag(retriever, client, MODEL, TEMPERATURE, input_file="test_questions.json", 
                      output_json="results_test_for_rag.json"):
    
   # Обрабатывает список вопросов и сохраняет ответы в JSON и CSV
    
   questions = load_chunks(input_file)

   results = []

   for q_entry in questions:
      question_text = q_entry.get("question")
      chunk_id = q_entry.get("chunk_id")
      
      
      try:
         
         response = rag_query(
               question=question_text,
               retriever=retriever,
               client=client,
               model=MODEL,
               temperature=TEMPERATURE
         )
         
         
         try:
               answer_data = json.loads(response)
         except json.JSONDecodeError:
               answer_data = {"answer": "Ошибка парсинга ответа модели", "sources": [], "raw_response": response}
         
         # Формируем итоговую запись
         result_entry = {
               "question": {
                  "chunk_id": chunk_id,
                  "text": question_text
               },
               "answer": answer_data,
               "metadata": {
                  "model": MODEL,
                  "temperature": TEMPERATURE
               }
         }
         
         results.append(result_entry)
         time.sleep(5)

      except Exception as e:
         print(f"Ошибка при обработке: {e}")
         # Сохраняем запись с ошибкой, чтобы не терять прогресс
         results.append({
               "question": {"chunk_id": chunk_id, "text": question_text},
               "answer": {"answer": f"Ошибка выполнения: {str(e)}", "sources": []},
               "metadata": {"error": True}
         })
      
   with open(output_json, 'w', encoding='utf-8') as f:
      json.dump(results, f, ensure_ascii=False, indent=2)
   
   return results


if __name__ == "__main__":

   KEY = load_api_key("NEO_KEY.txt")

   client = OpenAI(
        api_key=KEY,
        base_url=NEO_URL
    )
   
   chunks = load_chunks(FILE_WITH_CHUNKS)
    
   docs = chunks_to_documents(chunks)

   retriever = create_ensemble_retriever(docs, k=3)

   results_test_retriver = test_retriver(retriever)

   results_test_rag = test_rag(retriever, client, NEO_MODEL_GPT, 0.3)