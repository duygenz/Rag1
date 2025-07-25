import os
import numpy as np
from flask import Flask, jsonify, request
from pinecone import Pinecone

# --- 1. CẤU HÌNH VÀ KẾT NỐI PINECONE ---

# Lấy thông tin cấu hình từ các biến môi trường đã thiết lập trên Render
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_HOST = os.environ.get('PINECONE_HOST')
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME')

# Kiểm tra xem các biến môi trường đã được thiết lập chưa
if not all([PINECONE_API_KEY, PINECONE_HOST, PINECONE_INDEX_NAME]):
    raise ValueError("Lỗi: Vui lòng thiết lập các biến môi trường PINECONE_API_KEY, PINECONE_HOST, và PINECONE_INDEX_NAME")

# Khởi tạo kết nối tới Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # Kết nối tới index đã tồn tại của bạn (sử dụng host để kết nối trực tiếp)
    index = pc.Index(name=PINECONE_INDEX_NAME, host=PINECONE_HOST)
    print(f"Kết nối thành công tới index '{PINECONE_INDEX_NAME}'")
except Exception as e:
    raise RuntimeError(f"Không thể khởi tạo kết nối Pinecone: {e}")


# --- 2. TẠO ỨNG DỤNG WEB FLASK ---

app = Flask(__name__)


# --- 3. ĐỊNH NGHĨA CÁC ROUTE (ĐƯỜNG DẪN) ---

@app.route('/')
def health_check():
    """
    Route cơ bản để kiểm tra "sức khỏe" của ứng dụng
    và trạng thái kết nối tới Pinecone.
    """
    try:
        # Lấy thông tin thống kê của index để xác nhận kết nối
        stats = index.describe_index_stats()
        return jsonify({
            "message": "Ứng dụng đang hoạt động và đã kết nối thành công tới Pinecone.",
            "index_name": PINECONE_INDEX_NAME,
            "index_stats": stats.to_dict() # Chuyển object sang dict để jsonify
        })
    except Exception as e:
        return jsonify({"error": f"Không thể lấy thông tin từ Pinecone: {e}"}), 500


@app.route('/query')
def query_example():
    """
    Route ví dụ để thực hiện một truy vấn tìm kiếm vector.
    Nó sẽ tạo ra một vector ngẫu nhiên để minh họa.
    """
    try:
        # Tạo một vector ngẫu nhiên có 768 chiều (phải khớp với DIMENSIONS của bạn)
        random_vector = np.random.rand(768).tolist()

        # Thực hiện truy vấn
        query_results = index.query(
            vector=random_vector,
            top_k=5,  # Lấy 5 kết quả gần nhất
            include_metadata=True
        )
        
        # Trả về kết quả dưới dạng JSON
        return jsonify(query_results.to_dict())

    except Exception as e:
        return jsonify({"error": f"Lỗi xảy ra trong quá trình truy vấn: {e}"}), 500


# --- 4. CHẠY ỨNG DỤNG ---
if __name__ == "__main__":
    # Render sẽ tự động gán PORT qua biến môi trường
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

