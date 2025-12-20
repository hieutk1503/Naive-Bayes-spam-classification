import pickle
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from naive_bayes import train, predict
from data_loader import doc_file_csv


# ===========================
# Tiền xử lý văn bản
# ===========================
def clean_text(text):
    # Ép về string, xử lý NaN
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(
        r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
        r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
        r"ùúụủũưừứựửữỳýỵỷỹđ]",
        " ",
        text
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_and_save(
    csv_path="vi_dataset.csv",
    model_path="model.pkl"
):
    print("[INFO] Đang đọc dữ liệu...")
    raw_data = doc_file_csv(csv_path)

    # ===========================
    # Clean + giới hạn độ dài
    # ===========================
    data = []
    for label, text in raw_data:
        text = clean_text(text)

        # ⚠ Giới hạn độ dài để tránh bias
        words = text.split()
        if len(words) < 3:
            continue

        if len(words) > 200:
            text = " ".join(words[:200])

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
    word_counts, class_counts, vocab = train(train_data)

    # ===========================
    # Đánh giá
    # ===========================
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