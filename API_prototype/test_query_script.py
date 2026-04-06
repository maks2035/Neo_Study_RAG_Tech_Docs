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

import logging


def compute_hits(results, k):
    total = len(results)
    hits = 0

    for r in results:
        rank = r["retrieved_chunks"].get("rank")
        if rank is not None and rank <= k:
            hits += 1

    return hits / total

def compute_mrr(results):
    total = len(results)
    rr_sum = 0

    for r in results:
        rank = r["retrieved_chunks"].get("rank")
        if rank is not None:
            rr_sum += 1 / rank

    return rr_sum / total

def setup_logger(log_file="rag_test.log"):
    logger = logging.getLogger("rag_logger")
    logger.setLevel(logging.INFO)

    # чтобы не дублировались хендлеры при повторных запусках
    if not logger.handlers:
      formatter = logging.Formatter('%(message)s')

      console_handler = logging.StreamHandler()
      console_handler.setLevel(logging.INFO)
      console_handler.setFormatter(formatter)
      logger.addHandler(console_handler)
      
      if log_file != None:
         file_handler = logging.FileHandler(log_file, encoding='utf-8')
         file_handler.setLevel(logging.INFO)
         file_handler.setFormatter(formatter)
         logger.addHandler(file_handler)

    return logger


def test_retriver(retriever, input_file="test_questions.json", 
                      output_json="results_test_for_retiver.json", 
                      flag_save=False):
   questions = load_chunks(input_file)

   results = []

   for q_entry in questions:
        question_text = q_entry.get("question")
        chunk_id = q_entry.get("chunk_id")
        question_id = q_entry.get("question_id")

        try:
            retrieved_docs = retriever.invoke(question_text)

            sources = []
            found = False
            rank = None  

            for i, doc in enumerate(retrieved_docs, start=1):
                doc_chunk_id = doc.metadata.get("chunk_id") if hasattr(doc, 'metadata') else None
                
                if doc_chunk_id == chunk_id and rank is None:
                    rank = i
                    found = True  

                sources.append({
                     "chunk_id": doc_chunk_id,
                     "page": doc.metadata.get("page") if hasattr(doc, 'metadata') else None,
                     "source": doc.metadata.get("source") if hasattr(doc, 'metadata') else None,
                     "text": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                })

            
            logger.info(f"Question №{question_id}, chunk_id={chunk_id} → FOUND: {found}")

            result_entry = {
                "question": {
                     "chunk_id": chunk_id,
                     "question_id": question_id,
                     "text": question_text
                },
                "retrieved_chunks": {
                     "count": len(sources),
                     "items": sources,
                     "has_target_chunk": found,
                     "rank": rank
                }
            }
            results.append(result_entry)

        except Exception as e:
            logger.info(f"Ошибка при retrieval: {e}")
            logger.info(f"Question №{question_id}, chunk_id={chunk_id} → FOUND: False")

            results.append({
                "question": {"chunk_id": chunk_id, "question_id": question_id, "text": question_text},
                "retrieved_chunks": {
                    "count": 0,
                    "items": [],
                    "error": str(e),
                    "has_target_chunk": False
                },
            })

   if flag_save:
      with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

   return results

def test_rag(retriever, client, MODEL, TEMPERATURE, input_file="test_questions.json", 
                      output_json="results_test_for_rag.json", flag_save=False):
    
   # Обрабатывает список вопросов и сохраняет ответы в JSON и CSV
    
   questions = load_chunks(input_file)

   results = []

   for q_entry in questions:
      question_text = q_entry.get("question")
      chunk_id = q_entry.get("chunk_id")
      question_id = q_entry.get("question_id")
      
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
      
         sources = answer_data.get("sources", [])

         found = False
         for src in sources:
            if src.get("chunk_id") == chunk_id:
               found = True
               break

         not_found_phrase = "The information was not found in the submitted documents"
         answer_text = answer_data.get("answer", "")

         is_grounded = False

         if found:
            is_grounded = True
         elif not found and not_found_phrase.lower() in answer_text.lower():
            is_grounded = True


         logger.info(f"Question №{question_id}, chunk_id={chunk_id} → FOUND in LLM answer: {found}, is grounded {is_grounded}")

         result_entry = {
               "question": {
                  "chunk_id": chunk_id,
                  "question_id": question_id,
                  "text": question_text
                },
               "has_target_chunk": found,
               "is_grounded": is_grounded,
               "answer": answer_data,
               "metadata": {
                  "model": MODEL,
                  "temperature": TEMPERATURE
               }
         }
         
         results.append(result_entry)
         time.sleep(5)

      except Exception as e:
         logger.info(f"Ошибка при обработке: {e}")
         logger.info(f"Question №{question_id}, chunk_id={chunk_id} → FOUND in LLM answer: False, is grounded False")
         results.append({
               "question": {"chunk_id": chunk_id, "question_id": question_id, "text": question_text},
               "has_target_chunk": found,
               "is_grounded": is_grounded,
               "answer": {"answer": f"Ошибка выполнения: {str(e)}", "sources": []},
               "metadata": {"error": True}
         })
   if flag_save:   
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

   logger = setup_logger("test_retriver.txt") 
   retriever = create_ensemble_retriever(docs, k=5)

   logger.info("======TEST RETRIVER===========")
   results_test_retriver = test_retriver(retriever, flag_save = True)

   hit_1 = compute_hits(results_test_retriver, 1)
   hit_5 = compute_hits(results_test_retriver, 5)
   mrr = compute_mrr(results_test_retriver)
   logger.info(f"hit@1 = {hit_1}")
   logger.info(f"hit@5 = {hit_5}")
   logger.info(f"mrr = {mrr}")

   #logger.info("======TEST RAG===========")
   #results_test_rag = test_rag(retriever, client, NEO_MODEL_GPT, 0.3)

