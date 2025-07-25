# core/processor.py

from sentence_transformers import SentenceTransformer

# THAY ĐỔI DÒNG NÀY:
# model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2') # <--- MODEL CŨ, QUÁ LỚN

# BẰNG DÒNG NÀY:
model = SentenceTransformer('all-MiniLM-L6-v2') # <--- MODEL MỚI, NHỎ GỌN

# Kích thước vector của model mới này là 384, bạn cần cập nhật trong Pinecone
MODEL_DIMENSION = 384

def create_embedding(text):
    """Tạo vector embedding cho một đoạn văn bản."""
    return model.encode(text).tolist()
