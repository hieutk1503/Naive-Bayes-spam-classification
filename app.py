import streamlit as st
import pickle
import re
import pdfplumber
from docx import Document
from pyvi import ViTokenizer # <--- THƯ VIỆN QUAN TRỌNG

# Import hàm predict từ file naive_bayes.py
from naive_bayes import predict

# ==========================================
# 1. CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(
    page_title="Vietnamese Spam Classifier",
    page_icon="📩",
    layout="centered"
)

st.title("📧 Phân Loại Thư Rác Tiếng Việt")

# ==========================================
# 2. LOAD MODEL
# ==========================================
# Đảm bảo bạn đã chạy file train_and_save.py trước đó!
try:
    with open("model.pkl", "rb") as f:
        model_data = pickle.load(f)

    word_counts = model_data["word_counts"]
    class_counts = model_data["class_counts"]
    vocab = model_data["vocab"]
except FileNotFoundError:
    st.error("❌ Lỗi: Không tìm thấy file 'model.pkl'. Hãy chạy file train trước!")
    st.stop()

# ==========================================
# 3. HÀM TIỀN XỬ LÝ (Phải giống hệt file Train)
# ==========================================
def clean_text(text):
    if text is None: return ""
    if not isinstance(text, str): text = str(text)

    # 1. Chữ thường
    text = text.lower()
    
    # 2. Xóa rác
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text) # Xóa số
    text = re.sub(
        r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
        r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
        r"ùúụủũưừứựửữỳýỵỷỹđ]",
        " ",
        text
    )
    text = re.sub(r"\s+", " ", text).strip()
    
    # 3. TÁCH TỪ TIẾNG VIỆT
    # Input: "khuyến mãi cực sốc" -> Output: "khuyến_mãi cực sốc"
    text = ViTokenizer.tokenize(text)
    
    return text

# ==========================================
# 4. CÁC HÀM ĐỌC FILE
# ==========================================
def read_txt(file):
    return file.read().decode("utf-8", errors="ignore")

def read_docx(file):
    doc = Document(file)
    full_text = [p.text for p in doc.paragraphs]
    return "\n".join(full_text)

def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extract = page.extract_text()
            if extract:
                text += extract + "\n"
    return text

# ==========================================
# 5. GIAO DIỆN & XỬ LÝ CHÍNH
# ==========================================
st.write("### Nhập dữ liệu")
uploaded_file = st.file_uploader("Tải lên file văn bản (txt, pdf, docx)", type=["txt", "pdf", "docx"])

text_content = ""

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type == "txt":
        text_content = read_txt(uploaded_file)
    elif file_type == "docx":
        text_content = read_docx(uploaded_file)
    elif file_type == "pdf":
        text_content = read_pdf(uploaded_file)

input_text = st.text_area("Nội dung email/tin nhắn:", value=text_content, height=200)

if st.button("🔍 Kiểm tra ngay", type="primary"):
    if not input_text.strip():
        st.warning("⚠ Vui lòng nhập nội dung để kiểm tra.")
    else:
        # 1. Làm sạch dữ liệu & Tách từ
        cleaned_input = clean_text(input_text)
        
        # (Debug: Có thể mở dòng dưới nếu muốn xem nó tách từ thế nào)
        # st.code(cleaned_input) 
        
        # 2. Kiểm tra rỗng
        if not cleaned_input:
            st.warning("⚠ Văn bản không có nội dung hợp lệ.")
        else:
            # 3. Dự đoán
            label = predict(cleaned_input, word_counts, class_counts, vocab)
            
            # 4. Kết quả
            st.markdown("---")
            if label == "spam":
                st.error("🚨 KẾT QUẢ: ĐÂY LÀ THƯ RÁC (SPAM)")
            else:
                st.success("✅ KẾT QUẢ: ĐÂY LÀ THƯ BÌNH THƯỜNG (HAM)")