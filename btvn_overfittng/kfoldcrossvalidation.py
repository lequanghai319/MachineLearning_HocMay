import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# 1. Đọc dữ liệu
df = pd.read_csv('du_lieu.csv')
X = df[['DienTich']].values
y = df['GiaNha'].values

print("--- SỬ DỤNG K-FOLD CROSS VALIDATION ĐỂ CHIA TRAIN/TEST ---")

# 2. Định nghĩa K-Fold: Chia 10 dòng thành 5 tập (folds), mỗi tập 2 dòng
kf = KFold(n_splits=5, shuffle=True, random_state=42)
poly = PolynomialFeatures(degree=6) 

fold = 1
for train_index, test_index in kf.split(X):
    # Trích xuất dữ liệu Train/Test theo chỉ số của từng Fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train và Test ở từng vòng lặp
    print(f"\n[Fold {fold}]")
    print(f"  + Index tập Train : {train_index}")
    print(f"  + Index tập Test  : {test_index}")
    
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # 3. Sử dụng mô hình Ridge Regression (Regularization) để tránh Overfitting
    model_ridge = Ridge(alpha=100.0) 
    model_ridge.fit(X_train_poly, y_train)
    
    # Chấm điểm R2 Score trên tập Test chưa từng thấy
    score_test = r2_score(y_test, model_ridge.predict(X_test_poly)) * 100
    print(f"  -> R2 Score trên tập Test: {score_test:.2f}%")
    
    fold += 1