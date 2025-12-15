import json
import os
import shutil
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. ĐỊNH NGHĨA ĐƯỜNG DẪN ---

# DÙNG FILE NÀY: File JSON *đã gộp* (chứa cả luật VÀ biển báo)
KNOWLEDGE_BASE_PATH = "scripts/knowledge_base_final.json" 

# Model embedding (giữ nguyên)
EMB_MODEL_PATH = "models/bkai_vn_bi_encoder"

# Nơi lưu DB MỚI (đổi tên để không ghi đè lên DB cũ)
PERSIST_DIRECTORY = "Vecto_Database/db_bienbao_2"


# --- 2. TẢI VÀ CHUẨN HÓA DỮ LIỆU ---
print(f"🔄 Đang tải cơ sở kiến thức từ: {KNOWLEDGE_BASE_PATH}...")

all_documents = []
try:
    with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f) # data là một list các chunks
    
    if not isinstance(data, list):
        raise ValueError("File JSON không phải là một danh sách (list).")

    # Chuyển đổi từ dict sang Document object của LangChain
    for i, item in enumerate(data):
        # Đảm bảo chunk có đủ 2 trường
        if "page_content" not in item or "metadata" not in item:
            print(f"⚠️ Cảnh báo: Bỏ qua mục {i} vì thiếu 'page_content' hoặc 'metadata'.")
            continue
            
        doc = Document(
            page_content=item["page_content"],
            metadata=item["metadata"]
        )
        all_documents.append(doc)
        
    print(f"✅ Đã tải và chuẩn hóa {len(all_documents)} chunks tài liệu.")
    
    # Kiểm tra thử chunk cuối (thường là biển báo)
    if all_documents:
        print("\n--- VÍ DỤ CHUNK CUỐI CÙNG (Kiểm tra xem biển báo đã vào chưa) ---")
        print("NỘI DUNG:")
        print(all_documents[-1].page_content) #
        print("\nMETADATA:")
        print(all_documents[-1].metadata) #

except FileNotFoundError:
    print(f"❌ LỖI: Không tìm thấy file {KNOWLEDGE_BASE_PATH}.")
    print("Vui lòng chạy lại script gộp file từ lượt trước để tạo file này.")
    exit() # Thoát script nếu không có file
except Exception as e:
    print(f"❌ Đã xảy ra lỗi khi đọc file JSON: {e}")
    exit()

    
# --- 3. TẢI MODEL EMBEDDING ---
print("\n🔄 Đang tải mô hình embedding (BKAI)...")
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=EMB_MODEL_PATH,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("✅ Đã tải mô hình embedding thành công.")


# --- 4. TẠO VÀ LƯU VECTOR STORE ---
print(f"\n🔄 Đang tạo Vector Database tại: {PERSIST_DIRECTORY}...")

# (Tùy chọn) Xóa DB cũ nếu bạn muốn tạo mới hoàn toàn
if os.path.exists(PERSIST_DIRECTORY):
    print(f"    (Phát hiện DB cũ tại '{PERSIST_DIRECTORY}', đang xóa...)")
    shutil.rmtree(PERSIST_DIRECTORY)

# Tạo DB từ các Document đã xử lý
vectordb = Chroma.from_documents(
    documents=all_documents,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY
)

print(f"✅ Đã tạo và lưu Vector Database thành công với {len(all_documents)} chunks.")
print("--- HOÀN TẤT ---")