import csv, random

ham_templates = [
    "Hôm nay {action} {time}",
    "Bạn {action} {time} không?",
    "Đừng quên {action} {time}",
    "Nhớ {action} nhé",
    "{greeting}, {action} {time}"
]

spam_templates = [
    "{action} ngay để nhận {gift}",
    "Khuyến mãi {discount}, {action} ngay",
    "Bạn đã trúng {gift}, {action} ngay",
    "{action} để nhận {gift} miễn phí",
    "Cơ hội {action} {gift}, đừng bỏ lỡ"
]

ham_actions = ["đi học", "đi ăn trưa", "nộp bài tập", "học nhóm", "họp online"]
ham_times = ["lúc 8h", "hôm nay", "cuối tuần", "sáng mai", "chiều nay"]
ham_greetings = ["Chào bạn", "Hi", "Xin chào", "Hey", "Hello"]

spam_actions = ["Click vào đây", "Đăng ký", "Nhanh tay", "Nhấn vào link", "Mua ngay"]
spam_gifts = ["iPhone 15", "voucher 100k", "thẻ cào miễn phí", "ưu đãi 50%", "phần mềm bản quyền"]
spam_discounts = ["50%", "70%", "30%", "ưu đãi đặc biệt", "giảm giá sốc"]

data = []

# Tạo 350 ham
for _ in range(350):
    template = random.choice(ham_templates)
    text = template.format(
        action=random.choice(ham_actions),
        time=random.choice(ham_times),
        greeting=random.choice(ham_greetings)
    )
    data.append(['ham', text])

# Tạo 150 spam
for _ in range(150):
    template = random.choice(spam_templates)
    text = template.format(
        action=random.choice(spam_actions),
        gift=random.choice(spam_gifts),
        discount=random.choice(spam_discounts)
    )
    data.append(['spam', text])

random.shuffle(data)

# Ghi ra file CSV
with open('sms_spam_vi.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['label', 'text'])
    writer.writerows(data)

print("Đã tạo file sms_spam_vi.csv với dữ liệu phong phú hơn.")