import streamlit as st
import pickle
import pdfplumber
from docx import Document

from naive_bayes import predict   # dùng predict tự xây dựng


# ==========================
# 📌 Load model đã lưu
# ==========================
with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)

word_counts = model_data["word_counts"]
class_counts = model_data["class_counts"]
vocab = model_data["vocab"]


# ==========================
# 📌 Đọc file TXT
# ==========================
def read_txt(file):
    return file.read().decode("utf-8", errors="ignore")


# ==========================
# 📌 Đọc file DOCX
# ==========================
def read_docx(file):
    doc = Document(file)
    full_text = [p.text for p in doc.paragraphs]
    return "\n".join(full_text)


# ==========================
# 📌 Đọc file PDF
# ==========================
def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text


# ==========================
# 📌 Dự đoán
# ==========================
def classify(message):
    label = predict(message, word_counts, class_counts, vocab)
    return label


# ==========================
# 📌 Giao diện Streamlit
# ==========================
st.set_page_config(page_title="Spam Classifier", page_icon="📩")

st.title("📧 Naive Bayes Spam Classification")
st.write("Upload file văn bản hoặc nhập text để phân loại Spam / Ham")

uploaded_file = st.file_uploader(
    "Chọn file cần phân loại",
    type=["txt", "pdf", "docx"]
)

text_content = ""

# Nếu tải file
if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]

    if file_type == "txt":
        text_content = read_txt(uploaded_file)

    elif file_type == "docx":
        text_content = read_docx(uploaded_file)

    elif file_type == "pdf":
        text_content = read_pdf(uploaded_file)

    else:
        st.error("❌ Định dạng không hỗ trợ.")
        st.stop()

# Ô nhập text thủ công
st.subheader("✍ Nhập nội dung hoặc xem nội dung file tải lên:")
input_text = st.text_area("Nội dung:", text_content, height=250)


# Nút phân loại
if st.button("🔍 Phân loại"):
    if not input_text.strip():
        st.error("⚠ Vui lòng nhập nội dung hoặc tải file!")
    else:
        label = classify(input_text)

        st.subheader("📌 Kết quả phân loại:")
        if label == "spam":
            st.error("🚨 SPAM – Thư rác")
        else:
            st.success("✅ HAM – Thư bình thường")