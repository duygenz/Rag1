import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel

from core.processor import create_embedding
from core.vector_db import search_similar_chunks

app = FastAPI()

class SearchQuery(BaseModel):
    query: str

def call_gemini_api(prompt):
    """Hàm gọi Gemini API, tương tự như trong file JS của bạn."""
    api_key = os.environ.get("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status() # Báo lỗi nếu request thất bại
    return response.json()['candidates'][0]['content']['parts'][0]['text']


@app.get("/")
def read_root():
    return {"Status": "API is running"}

@app.post("/api/search")
def search_and_analyze(search_query: SearchQuery):
    # 1. Tạo embedding cho câu hỏi
    query_vector = create_embedding(search_query.query)

    # 2. Tìm các chunk liên quan trong Vector DB
    similar_chunks = search_similar_chunks(query_vector, top_k=7)

    # 3. Xây dựng ngữ cảnh RAG
    rag_context = "\n---\n".join([
        f"Trích dẫn từ bài '{match['metadata']['articleTitle']}': {match['metadata']['text']}"
        for match in similar_chunks
    ])

    # 4. Tạo prompt cuối cùng và gọi Gemini
    final_prompt = f"""Dựa vào các thông tin được trích xuất dưới đây, hãy trả lời câu hỏi của người dùng.
    Câu hỏi: "{search_query.query}"

    Thông tin trích xuất:
    {rag_context}
    """
    answer = call_gemini_api(final_prompt)

    # 5. Trả kết quả về cho frontend
    return {
        "question": search_query.query,
        "answer": answer,
        "sources": [chunk['metadata'] for chunk in similar_chunks]
    }
