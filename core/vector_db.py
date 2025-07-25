import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Khởi tạo kết nối
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

INDEX_NAME = "financial-news"

def init_vector_db():
    """Khởi tạo index nếu chưa tồn tại."""
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=768, # Kích thước vector của model embedding
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(INDEX_NAME)

index = init_vector_db()

def upsert_chunks(chunks_with_vectors):
    """Đưa các chunk và vector vào DB."""
    # chunks_with_vectors là list của: (id, vector, metadata)
    index.upsert(vectors=chunks_with_vectors)

def search_similar_chunks(query_vector, top_k=5):
    """Tìm kiếm các chunks tương tự."""
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return results['matches']
