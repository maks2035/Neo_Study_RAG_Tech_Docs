import json
import time
import os
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm 

# Импорт готовых функций из RAG-кода
from API_prototype import (
   load_api_key,
   load_chunks,
   chunks_to_documents,
   NEO_URL,
   NEO_MODEL_GPT,
)

SYSTEM_PROMPT = """
<ROLE>
You are generating evaluation questions for a RAG system based on technical documentation.
</ROLE>

<TASK>
Generate ONE question based ONLY on the provided text chunk.
</TASK>

<RULES>
1. Generate the question in RUSSIAN language
2. The question must be answerable ONLY using the provided chunk text
3. source_fragment must be an EXACT substring from the chunk text (copy-paste, no paraphrasing)
4. Choose source_class based on the content type:
   - safety: warnings, prohibitions, fire/electrical hazards
   - operation: usage instructions, controls, settings
   - troubleshooting: fault diagnosis, problem solutions
   - maintenance: cleaning, part replacement
   - specs: technical parameters, dimensions, temperatures
5. Choose question_class based on answer structure:
   - factual: single fact/value (1-2 sentences)
   - procedural: sequence of steps (1, 2, 3...)
   - diagnostic: condition + recommendation ("if X, then Y")
6. Do NOT invent facts not present in the text
7. Return ONLY valid JSON matching the required schema
</RULES>

<PRIORITIZATION>
Actively prefer PROCEDURAL or DIAGNOSTIC question types whenever the chunk contains step-by-step instructions, setup/maintenance procedures, troubleshooting tips, warnings with actions, or conditional logic.
Reserve FACTUAL questions ONLY for chunks that contain isolated specifications, standalone parameters, or single-value rules without sequential or conditional elements.
</PRIORITIZATION>

<OUTPUT_FORMAT>
Return ONLY valid JSON, no additional text or explanations.
</OUTPUT_FORMAT>
""".strip()

def build_generation_prompt(chunk_text):
   #Формирует user prompt с текстом чанка в XML-тегах
   
   prompt = f"""
<CHUNK_TEXT>
{chunk_text}
</CHUNK_TEXT>
""".strip()
   
   return prompt

class GeneratedQuestion(BaseModel):
   #Модель ответа от LLM при генерации вопроса
   question: str = Field(..., description="The question text in Russian")
   source_fragment: str = Field(..., description="Exact fragment from the source chunk used to generate the question")
   source_class: str = Field(..., description="Type of information in the source", pattern="^(safety|operation|troubleshooting|maintenance|specs)$")
   question_class: str = Field(..., description="Type of question/expected answer", pattern="^(factual|procedural|diagnostic)$")


class QuestionEntry(BaseModel):
   #Полная запись для сохранения в датасет
   question_id: int = Field(..., description="Sequential question number")
   chunk_id: int = Field(..., description="ID of the source chunk")
   question: str = Field(..., description="The question text in Russian")
   source_fragment: str = Field(..., description="Exact fragment from the source chunk")
   source_class: str = Field(..., description="Type of information in the source")
   question_class: str = Field(..., description="Type of question/expected answer")


def get_next_question_id(path_file):
    
   # Определяет следующий question_id.
   
   if not os.path.exists(path_file):
      return 1
   
   try:
      with open(path_file, 'r', encoding='utf-8') as f:
         data = json.load(f)
      
      if not data:
         return 1
      
      max_id = max(question_json.get("question_id", 0) for question_json in data if isinstance(question_json.get("question_id"), int))
      return max_id + 1
   
   except (json.JSONDecodeError, IOError):
      return 1

def generate_question_for_chunk(chunk, client, model, system_prompt, temperature=0.7):
   #Генерирует один вопрос для одного чанка
   chunk_id = chunk["id"]
   chunk_text = chunk["text"]
   
   user_prompt = build_generation_prompt(chunk_text)
   
   try:
      response = client.chat.completions.create(
         model=model,
         messages=[
               {"role": "system", "content": system_prompt},
               {"role": "user", "content": user_prompt}
         ],
         temperature=temperature,
         response_format={
               "type": "json_schema",
               "json_schema": {
                  "name": "generated_question",
                  "schema": GeneratedQuestion.model_json_schema()
               }
         }
      )
      raw_output = response.choices[0].message.content
      
      parsed = GeneratedQuestion.model_validate_json(raw_output)

      return parsed
      
   except ValidationError as e:
      print(f"Pydantic validation error for chunk {chunk_id}: {e}")
      return None
   except Exception as e:
      print(f"Error calling LLM for chunk {chunk_id}: {e}")
      return None


def generate_questions_batch(chunks, client, start_idx, end_idx, output_file, model, system_prompt,
                             append_mode=False, temperature=0.7, delay=2):
   #Генерирует вопросы для пакета чанков и сохраняет результат.
   
   # Определяем диапазон чанков для обработки
   chunks_to_process = chunks[start_idx:(end_idx if end_idx is not None else len(chunks))]
   
   # Определяем стартовый question_id
   if append_mode and os.path.exists(output_file):
      next_qid = get_next_question_id(output_file)
      with open(output_file, 'r', encoding='utf-8') as f:
         existing_results = json.load(f)
   else:
      next_qid = 1
      existing_results = []
   
   results = []
   
   for i, chunk in enumerate(tqdm(chunks_to_process, desc="Generating questions", unit="chunk"), start=1):
      
      entry = generate_question_for_chunk(
         chunk=chunk,
         client=client,
         model=model,
         temperature=temperature,
         system_prompt=system_prompt,
      )
      
      if entry is not None:
         
         full_entry = QuestionEntry(
               question_id=next_qid,
               chunk_id=chunk["id"],
               question=entry.question,
               source_fragment=entry.source_fragment,
               source_class=entry.source_class,
               question_class=entry.question_class
         )
         results.append(full_entry.model_dump())
         next_qid += 1
      
      time.sleep(delay)
   
   final_results = existing_results + results if append_mode else results
   
   if final_results:
      with open(output_file, 'w', encoding='utf-8') as f:
         json.dump(final_results, f, ensure_ascii=False, indent=2)
      tqdm.write(f"\nSaved {len(final_results)} questions to {output_file}")
   
   return results


if __name__ == "__main__":
   chunks = load_chunks("chunks_data.json")
   print(f"Loaded {len(chunks)} chunks")

   api_key = load_api_key("NEO_KEY.txt")
   client = OpenAI(api_key=api_key, base_url=NEO_URL)
   
   generated = generate_questions_batch(
      chunks=chunks,
      client=client,
      start_idx=23,
      end_idx=258,
      output_file="generated_questions.json",
      model=NEO_MODEL_GPT,
      system_prompt=SYSTEM_PROMPT,
      append_mode=False,
      temperature=0.7,
      delay=5
   )
   
   if generated:
      stats = {}
      for q in generated:
         key = f"{q.get('source_class')}/{q.get('question_class')}"
         stats[key] = stats.get(key, 0) + 1
      print(f"\nDistribution: {stats}")