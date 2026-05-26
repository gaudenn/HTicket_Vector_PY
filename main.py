import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import logging
from gensim.models import Word2Vec
from underthesea import word_tokenize

# Khởi tạo tên logger (Cấu hình chi tiết sẽ được kích hoạt ở sự kiện startup)
logger = logging.getLogger("HTicketAI")
app = FastAPI(title="HTicket Vector AI Service")

# Sự kiện startup: Chạy ngay khi Uvicorn vừa khởi động xong để cấu hình lại hệ thống Log
@app.on_event("startup")
def startup_event():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.INFO)
    logger.propagate = True
    logger.info("🚀 Hệ thống Log Custom cho HTicket AI đã được kích hoạt thành công!")

# Định nghĩa cấu trúc dữ liệu nhận từ C#
class EventData(BaseModel):
    id: int
    name: str

class RetrieveRequest(BaseModel):
    query: str
    events: List[EventData]

class PythonResponse(BaseModel):
    relevantEventIds: List[int]

# ====================================================================
# KHỞI TẠO CÁC BIẾN TOÀN CỤC (GLOBAL CACHE) ĐỂ TRÁNH RE-TRAIN LIÊN TỤC
# ====================================================================
global_model = None
cached_event_count = 0

# Hàm huấn luyện mô hình toán học Word2Vec
def train_global_model(events: List[EventData], query: str):
    global global_model
    logger.info("🤖 [AI Core] Phát hiện dữ liệu thay đổi hoặc bộ nhớ đệm trống. Tiến hành huấn luyện lại mô hình Word2Vec...")
    
    # Tiền xử lý dữ liệu: Tách từ tiếng Việt cho toàn bộ danh mục sự kiện và câu truy vấn hiện tại
    sentences = [word_tokenize(e.name, format="text").split() for e in events]
    sentences.append(word_tokenize(query, format="text").split())
    
    # Khởi tạo và huấn luyện nhanh mô hình Word2Vec
    global_model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, workers=1, epochs=20)
    logger.info("✅ [AI Core] Huấn luyện mô hình Word2Vec thành công!")

# Hàm tính toán vector trung bình cho câu
# Danh sách các từ xuất hiện trong câu lệnh đặt vé/hỏi đáp nhưng làm lệch vector tên sự kiện
IGNORE_WORDS = {
    "đặt", "mua", "vé", "thường", "vip", "đi", "giùm", "hộ", "cho", "xin", 
    "nào", "có", "sự", "kiện", "trận", "show", "buổi", "lịch", "giá",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"
}

def get_sentence_vector(sentence, model):
    # Tách từ tiếng Việt (Ví dụ: "đặt_vé", "bơi_lội")
    words = word_tokenize(sentence, format="text").split()
    
    # Lọc bỏ từ nhiễu (Chuyển về chữ thường để so khớp chính xác)
    filtered_words = [w for w in words if w.lower().replace("_", " ") not in IGNORE_WORDS and w.lower() not in IGNORE_WORDS]
    
    # Nếu sau khi lọc mà không còn từ nào (hoặc câu quá ngắn), quay lại dùng tập từ gốc
    if not filtered_words:
        filtered_words = words

    # Lấy vector của các từ có tồn tại trong từ điển Word2Vec
    vectors = [model.wv[word] for word in filtered_words if word in model.wv]
    
    # Nếu không có từ nào hợp lệ, trả về vector không
    if not vectors:
        return np.zeros(model.vector_size)
        
    # Tính trung bình cộng vector của các từ cốt lõi
    return np.mean(vectors, axis=0)

@app.post("/api/retrieve", response_model=PythonResponse)
async def retrieve_events(request: RetrieveRequest):
    global global_model, cached_event_count
    logger.info(f"📩 Tiếp nhận yêu cầu xử lý Query từ C#: '{request.query}'")
    
    if not request.events:
        logger.warning("Danh sách sự kiện đầu vào bị rỗng!")
        return PythonResponse(relevantEventIds=[])
        
    try:
        # Kiểm tra số lượng sự kiện hiện tại trong Database gửi sang
        current_event_count = len(request.events)
        
        # KIỂM TRA ĐIỀU KIỆN CACHE: 
        # Chỉ tái huấn luyện nếu chưa có mô hình HOẶC số lượng sự kiện thay đổi (Thêm/Xóa sự kiện ở Admin)
        if global_model is None or current_event_count != cached_event_count:
            train_global_model(request.events, request.query)
            cached_event_count = current_event_count  # Cập nhật số lượng phần tử vào bộ nhớ đệm
        else:
            logger.info(f"⚡ [Cache Hit] Tái sử dụng mô hình Word2Vec đã lưu trữ. Số lượng sự kiện ổn định: {cached_event_count}")

        # 3. Vector hóa câu truy vấn của người dùng sử dụng mô hình cache
        query_vec = get_sentence_vector(request.query, global_model)
        scores = []
        
        # 4. Tính toán độ tương đồng toán học Cosine Similarity cho từng sự kiện
        for event in request.events:
            event_vec = get_sentence_vector(event.name, global_model)
            
            dot_product = np.dot(query_vec, event_vec)
            norm_q = np.linalg.norm(query_vec)
            norm_e = np.linalg.norm(event_vec)
            
            # Khắc phục lỗi chia cho 0: Kiểm tra độ dài vector trước khi chia
            if norm_q > 0 and norm_e > 0:
                similarity = float(dot_product / (norm_q * norm_e))
            else:
                similarity = 0.0
                
            scores.append((event.id, similarity))
            # ĐÃ XÓA DÒNG IN LOG CHI TIẾT TỪNG EVENT Ở ĐÂY ĐỂ TRÁNH LOG FLOODING
            
        # 5. Sắp xếp danh sách theo thứ tự độ tương đồng giảm dần
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 6. Sàng lọc các phần tử vượt ngưỡng tương thích (Ngưỡng tối ưu > 0.1)
        result_ids = [item[0] for item in scores if item[1] > 0.1]
        
        # Cơ chế Fallback cục bộ: Nếu không tìm thấy kết quả nào khớp sâu, gợi ý các phần tử sẵn có
        if not result_ids:
            logger.info("Không có sự kiện nào vượt ngưỡng 0.1. Kích hoạt gợi ý mặc định.")
            max_fallback = min(2, len(request.events))
            result_ids = [e.id for e in request.events[:max_fallback]]
            
        logger.info(f" Hoàn tất xử lý toán học. Trả về các ID tương thích: {result_ids}")
        return PythonResponse(relevantEventIds=result_ids)
        
    except Exception as e:
        logger.error(f"❌ Lỗi hệ thống trong quá trình xử lý AI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    # Chạy máy chủ uvicorn lắng nghe trên cổng 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)