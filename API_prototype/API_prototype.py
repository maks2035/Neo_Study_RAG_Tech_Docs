import json
import time
from openai import OpenAI
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS


#MODEL_LLAMA="meta-llama/llama-3.3-70b-instruct:free"
#MODEL_QWEN="qwen/qwen3-next-80b-a3b-instruct:free"
#MODEL_GEMMA="google/gemma-3-27b-it:free"
MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
NEO_MODEL_QWEN = "qwen-3-235b-a22b-instruct-2507"

NEO_MODEL_GPT = "gpt-oss-120b"

NEO_URL = "https://litellm.happyhub.ovh/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"

FILE_WITH_CHUNKS = "chunks_data.json"

class Source(BaseModel):
    chunk_id: int
    page: int
    source: str
    text: str


class TopKRetriever:
    def __init__(self, retriever, k):
        self.retriever = retriever
        self.k = k

    def invoke(self, query):
        docs = self.retriever.invoke(query)
        return docs[:self.k]

class AnswerSchema(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    sources: List[Source]

SYSTEM_PROMPT = f"""
        You are a documentation assistant.

        GOAL:
        Answer the user's question based only on the provided context.

        TASKS:
        1. Analyze the context
        2. Find relevant information
        3. Generate an answer
        4. Provide sources
        5. Return JSON

        RULES:
        - Use ONLY the provided context
        - Do NOT use prior knowledge
        - Do NOT make assumptions
        - Answer in the same language as the question

        - If the answer is not found, return EXACTLY:
        "The information was not found in the submitted documents"

        OUTPUT FORMAT:
            - Return ONLY valid JSON
            - Do NOT add text outside JSON
            - The response must strictly follow the provided Pydantic schema
    """


def build_user_prompt(context, question) :
    #Создаёт user prompt с контекстом и вопросом
    return f"""
<CONTEXT>
{context}
</CONTEXT>
<QUESTION>
{question}
</QUESTION>
""".strip()


def format_context_for_prompt(retriever_results):
    #Форматирует найденные чанки в читаемый контекст для промпта
    formatted = []
    for i, doc in enumerate(retriever_results, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        text = doc.page_content.strip()
        formatted.append(f"[{i}],chunk_id {chunk_id}, {source}, стр. {page}:\n{text}")
    return "\n\n".join(formatted)

def create_ensemble_retriever(docs, k = 3):
    #Создаёт комбинированный ретривер (векторный + BM25)
    
    # Векторный поиск
    embeddings = HuggingFaceEmbeddings(
        model_name="cointegrated/LaBSE-en-ru",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    # BM25 (ключевые слова)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = k
    
    # Комбинированный
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

    final_retriever = TopKRetriever(ensemble_retriever, k)
    return final_retriever

def chunks_to_documents(chunks):
    #Конвертирует словари в Document объекты LangChain
    return [
        Document(
            page_content=chunk["text"],
            metadata={
                **chunk["metadata"],
                "chunk_id": chunk["id"]        
            }
        )
        for chunk in chunks
    ]

def load_chunks(filepath):
    #Загружает чанки из JSON файла
    with open(filepath, 'r', encoding='utf-8') as f:
        if filepath.endswith('.jsonl'):
            chunks = []
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
            return chunks
        else:
            return json.load(f)

def load_api_key(path_to_key):
    #Загрузка ключа openroutera
    key_path = Path(__file__).resolve().parent.parent.parent / path_to_key
    with open(key_path, "r") as f:
        return f.read().strip()

def ask_llm(user_prompt, system_prompt, client, MODEL, temperature=0.7, retries=3):
    #Отправка вопроса и получение ответа
    for _ in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer_schema",
                        "schema": AnswerSchema.model_json_schema()
                    }
                }
            )

            raw_output = response.choices[0].message.content

            parsed = AnswerSchema.model_validate_json(raw_output)

            return parsed.model_dump_json(ensure_ascii=False)

        except Exception as e:
            print("Retrying...", e)
            time.sleep(5)

    return None

def rag_query(question, retriever, client, model, temperature = 0.5):

    #Основной RAG-пайплайн: поиск → контекст → LLM → ответ
    
    results = retriever.invoke(question)
    
    if not results:
        return json.dumps({
            "answer": "The information was not found in the submitted documents",
            "sources": []
        }, ensure_ascii=False)
    
    
    context = format_context_for_prompt(results)

    user_prompt = build_user_prompt(context, question)
    
    response = ask_llm(user_prompt, SYSTEM_PROMPT, client, model, temperature=temperature)
    
    if response is None:
        return json.dumps({
            "answer": "Error generating response",
            "sources": []
        }, ensure_ascii=False)

    return response




if __name__ == "__main__":

    KEY = load_api_key("NEO_KEY.txt")
    
    client = OpenAI(
        api_key=KEY,
        base_url=NEO_URL
    )

    chunks = load_chunks(FILE_WITH_CHUNKS)
    
    docs = chunks_to_documents(chunks)
    
    retriever = create_ensemble_retriever(docs, k=5)

    test_query = "к чему может привести использование прибора с истекшим сроком службы?"
    
    result_json = rag_query(test_query, retriever, client, NEO_MODEL_GPT, 0.5)
    
    print(result_json)
