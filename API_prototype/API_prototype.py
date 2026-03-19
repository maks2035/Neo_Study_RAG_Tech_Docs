import json
import time
from openai import OpenAI
from pathlib import Path
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

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer to the user's question"
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer"},
                    "source": {"type": "string"},
                    "text": {"type": "string"}
                },
                "required": ["page", "source", "text"]
            }
        }
    },
    "required": ["answer", "sources"],
    "additionalProperties": False
}

SYSTEM_PROMPT = f"""
        You are a documentation assistant.
        Answer only using the provided context.
        Give the answer in the same language in which the question was asked.
        If the answer is not in the context, say "The information was not found in the submitted documents"and nothing more.
        Return JSON matching this schema: {json.dumps(SCHEMA, ensure_ascii=False)}.
    """


def build_user_prompt(context, question) :
    #Создаёт user prompt с контекстом и вопросом
    return f"""
CONTEXT:
{context}

QUESTION:
{question}
""".strip()


def format_context_for_prompt(retriever_results):
    """Форматирует найденные чанки в читаемый контекст для промпта"""
    formatted = []
    for i, doc in enumerate(retriever_results, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        text = doc.page_content.strip()
        formatted.append(f"[{i}] {source}, стр. {page}:\n{text}")
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
    
    return ensemble_retriever

def chunks_to_documents(chunks):
    #Конвертирует словари в Document объекты LangChain
    return [
        Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"]
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

def ask_llm(user_prompt, system_prompt, client, MODEL, temperature=0.7, retries=1):
    #Отправка вопроса и получение ответа
    for _ in range(retries):
        try:
            result = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return result

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
    
    if response is None or isinstance(response, str):
        return json.dumps({
            "answer": "Error generating response",
            "sources": []
        }, ensure_ascii=False)
    
    return response.choices[0].message.content




if __name__ == "__main__":

    KEY = load_api_key("NEO_KEY.txt")
    
    client = OpenAI(
        api_key=KEY,
        base_url=NEO_URL
    )

    chunks = load_chunks(FILE_WITH_CHUNKS)

    docs = chunks_to_documents(chunks)

    retriever = create_ensemble_retriever(docs, k=3)

    test_query = "к чему может привести использование прибора с истекшим сроком службы?"

    result_json = rag_query(test_query, retriever, client, NEO_MODEL_GPT, 0.3)

    print(result_json)
