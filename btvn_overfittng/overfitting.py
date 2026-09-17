import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Đọc dữ liệu từ file csv riêng
df = pd.read_csv('du_lieu.csv')
X = df[['DienTich']].values
y = df['GiaNha'].values

# 2. Chia tập Train và Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Ép mô hình bị Overfitting (học vẹt)
poly = PolynomialFeatures(degree=6)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model_loi = LinearRegression()
model_loi.fit(X_train_poly, y_train)

# 4. Chứng minh bằng con số
print("--- CHỨNG MINH OVERFITTING ---")
print(f"Độ chính xác Train: {r2_score(y_train, model_loi.predict(X_train_poly))*100:.2f}%")
print(f"Độ chính xác Test : {r2_score(y_test, model_loi.predict(X_test_poly))*100:.2f}%")