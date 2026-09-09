import pandas as pd
from sklearn.linear_model import LinearRegression

# Đọc dữ liệu từ file CSV
du_lieu = pd.read_csv('btvn_hoiquytuyentinh/dulieu.csv')
X = du_lieu[['DienTich']]
y = du_lieu['GiaNha']

# Khởi tạo và cho mô hình học từ dữ liệu
mo_hinh = LinearRegression()
mo_hinh.fit(X, y)

# Dự đoán thử giá của căn nhà 70m2
gia_du_doan = mo_hinh.predict([[70]])
print(f"Dự đoán giá nhà 70m2: {gia_du_doan[0]:.2f} tỷ VNĐ")