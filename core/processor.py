from sentence_transformers import SentenceTransformer

# Chọn một model embedding tốt, hỗ trợ tiếng Việt
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def create_embedding(text):
    """Tạo vector embedding cho một đoạn văn bản."""
    return model.encode(text).tolist()
