
import json
import time
from openai import OpenAI


#MODEL_LLAMA="meta-llama/llama-3.3-70b-instruct:free"
#MODEL_QWEN="qwen/qwen3-next-80b-a3b-instruct:free"
#MODEL_GEMMA="google/gemma-3-27b-it:free"
MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer to the user's question"
        },
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "page": {"type": "int"},
            "description": "List of sources"
        }
    },
    "required": ["answer", "sources"],
    "additionalProperties": False
}

CONTEXT = [
    {"text": "шарик синий", "metadata": {"source": "текст1.md", "page": 12}},
    {"text": "шарик не черный", "metadata": {"source": "текст42.md", "page": 132}},
    {"text": "куб в в красную крапинку", "metadata": {"source": "текст42.md", "page": 32}},
]

question = "какого цвета пирамида?"

user_prompt = f"""
CONTEXT:
{json.dumps(CONTEXT, ensure_ascii=False)}

QUESTION:
{question}
"""

system_prompt = f"""
        You are a documentation assistant.
        Answer only using the provided context.
        Give the answer in the same language in which the question was asked.
        If the answer is not in the context, say "The information was not found in the submitted documents"and nothing more.
        Return JSON matching this schema: {json.dumps(SCHEMA, ensure_ascii=False)}.
    """


OPENROUTER_KEY = ''

from pathlib import Path

key_path = Path(__file__).resolve().parent.parent.parent / "OPENROUTER_KEY.txt"

with open(key_path, "r") as f:
    OPENROUTER_KEY = f.read().strip()

client = OpenAI(
    api_key=OPENROUTER_KEY,
    base_url="https://openrouter.ai/api/v1"
)

def ask_llm(user_prompt, system_prompt, MODEL, temperature=0.7, retries=3):
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

    return "Failed after retries"


answer = ask_llm(user_prompt, system_prompt, MODEL, temperature=0.3)

print(answer)
print(answer.choices[0].message.content)