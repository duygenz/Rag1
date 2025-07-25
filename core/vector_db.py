# core/vector_db.py
# ...
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # <--- CẬP NHẬT KÍCH THƯỚC Ở ĐÂY
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
# ...
