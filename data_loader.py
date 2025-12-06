import pandas as pd

def doc_file_csv(filename):
    df = pd.read_csv(filename, encoding="utf-8")  # dùng utf-8 cho tiếng Việt
    df = df[['label', 'text']]  # đổi tên cột cho đúng
    data = list(df.itertuples(index=False, name=None))
    return data