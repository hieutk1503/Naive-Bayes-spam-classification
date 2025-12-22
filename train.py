import pickle
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from pyvi import ViTokenizer  # <--- THƯ VIỆN QUAN TRỌNG

# Import các file của bạn (giữ nguyên)
from naive_bayes import train, predict
from data_loader import doc_file_csv 

# ===========================
# Tiền xử lý văn bản (Đã nâng cấp)
# ===========================
def clean_text(text):
    # Ép về string, xử lý NaN
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # 1. Chuyển về chữ thường
    text = text.lower()
    
    # 2. Xóa URL
    text = re.sub(r"http\S+", " ", text)
    
    # 3. Xóa số (Tùy chọn, ở đây mình giữ nguyên logic cũ của bạn là xóa)
    text = re.sub(r"\d+", " ", text)
    
    # 4. Xóa ký tự đặc biệt
    text = re.sub(
        r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
        r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
        r"ùúụủũưừứựửữỳýỵỷỹđ]",
        " ",
        text
    )
    
    # 5. Xóa khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()
    
    # 6. TÁCH TỪ TIẾNG VIỆT (QUAN TRỌNG NHẤT)
    # Biến "sinh viên" thành "sinh_viên"
    text = ViTokenizer.tokenize(text)
    
    return text


def train_and_save(
    csv_path="vi_dataset.csv",
    model_path="model.pkl"
):
    print("[INFO] Đang đọc dữ liệu...")
    try:
        raw_data = doc_file_csv(csv_path)
    except Exception as e:
        print(f"[LOI] Khong doc duoc file CSV: {e}")
        return

    # ===========================
    # Clean + giới hạn độ dài
    # ===========================
    data = []
    print("[INFO] Đang xử lý và tách từ tiếng Việt...")
    
    for label, text in raw_data:
        text = clean_text(text) # <--- Hàm này giờ đã có tách từ

        # Giới hạn độ dài để tránh bias
        words = text.split()
        if len(words) < 3:
            continue

        if len(words) > 300: # Tăng lên chút vì từ ghép sẽ làm câu ngắn lại
            text = " ".join(words[:300])

        data.append((label, text))

    print(f"[INFO] Tổng mẫu sau clean: {len(data)}")

    # ===========================
    # Chia train/test
    # ===========================
    train_data, test_data = train_test_split(
        data, test_size=0.3, random_state=42, stratify=[l for l, _ in data]
    )

    print(f"[INFO] Train: {len(train_data)} | Test: {len(test_data)}")

    # ===========================
    # Train Naive Bayes
    # ===========================
    print("[INFO] Đang training...")
    word_counts, class_counts, vocab = train(train_data)

    # ===========================
    # Đánh giá
    # ===========================
    print("[INFO] Đang kiểm tra độ chính xác...")
    correct = 0
    for label, text in test_data:
        if predict(text, word_counts, class_counts, vocab) == label:
            correct += 1

    acc = correct / len(test_data) * 100
    print(f"[INFO] Accuracy: {acc:.2f}%")

    # ===========================
    # Lưu model
    # ===========================
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "word_counts": word_counts,
                "class_counts": class_counts,
                "vocab": vocab
            },
            f
        )

    print(f"[INFO] ✔ Model đã lưu: {model_path}")

if __name__ == "__main__":
    train_and_save()