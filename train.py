import pickle
from sklearn.model_selection import train_test_split
from naive_bayes import train, predict
from data_loader import doc_file_csv


def train_and_save(
    csv_path="sms_spam_vi.csv",
    model_path="model.pkl"
):
    # ===========================
    # 1. Load dữ liệu
    # ===========================
    print("[INFO] Đang đọc dữ liệu...")
    data = doc_file_csv(csv_path)   # Trả về list [(label, text), ...]

    print(f"[INFO] Tổng số mẫu: {len(data)}")

    # ===========================
    # 2. Chia train/test 70/30
    # ===========================
    train_data, test_data = train_test_split(
        data, test_size=0.3, random_state=42
    )

    print(f"[INFO] Train: {len(train_data)} mẫu")
    print(f"[INFO] Test : {len(test_data)} mẫu")

    # ===========================
    # 3. Train mô hình Naive Bayes
    # ===========================
    print("[INFO] Đang train mô hình Naive Bayes...")
    word_counts, class_counts, vocab = train(train_data)

    # ===========================
    # 4. Đánh giá nhanh
    # ===========================
    correct = 0
    for label, text in test_data:
        pred = predict(text, word_counts, class_counts, vocab)
        if pred == label:
            correct += 1

    accuracy = correct / len(test_data) * 100
    print(f"[INFO] Accuracy: {accuracy:.2f}%")

    # ===========================
    # 5. Lưu mô hình
    # ===========================
    print(f"[INFO] Đang lưu mô hình vào {model_path} ...")
    model_data = {
        "word_counts": word_counts,
        "class_counts": class_counts,
        "vocab": vocab
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    print("[INFO] ✔ Lưu thành công!")
    print("[INFO] File tạo ra:")
    print(f"       → {model_path}")


if __name__ == "__main__":
    train_and_save()