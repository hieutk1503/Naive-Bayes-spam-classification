import tkinter as tk
from tkinter import scrolledtext
from sklearn.model_selection import train_test_split

from naive_bayes import train, predict
from data_loader import doc_file_csv

# ===============================
# Train model (70% train – 30% test)
# ===============================
def train_model():
    data = doc_file_csv("sms_spam_vi.csv")

    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    word_counts, class_counts, vocab = train(train_data)

    correct = 0
    for label, text in test_data:
        pred = predict(text, word_counts, class_counts, vocab)
        if pred == label:
            correct += 1

    accuracy = 100 * correct / len(test_data)
    print(f"[INFO] Accuracy (Test 30%): {accuracy:.2f}%")

    return word_counts, class_counts, vocab


# ========== TRAIN KHI CHẠY ==========
word_counts, class_counts, vocab = train_model()


# ===============================
# APP GUI
# ===============================
def classify_message():
    message = input_box.get("1.0", tk.END).strip()

    if not message:
        result_label.config(text="Vui lòng nhập nội dung để phân loại!", fg="red")
        return

    prediction = predict(message, word_counts, class_counts, vocab)

    if prediction == "spam":
        result_label.config(text="KẾT QUẢ: THƯ RÁC (SPAM)", fg="red")
    else:
        result_label.config(text="KẾT QUẢ: TIN THƯỜNG (HAM)", fg="green")


# ===============================
# Giao diện Tkinter
# ===============================
window = tk.Tk()
window.title("Phân loại thư rác – Naive Bayes")
window.geometry("600x450")

title_label = tk.Label(window, text="Bộ phân loại thư rác (Naive Bayes)", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

input_box = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=60, height=10, font=("Arial", 12))
input_box.pack(pady=10)

predict_btn = tk.Button(window, text="Phân loại", font=("Arial", 14), width=20, command=classify_message)
predict_btn.pack(pady=10)

result_label = tk.Label(window, text="", font=("Arial", 16))
result_label.pack(pady=10)

window.mainloop()