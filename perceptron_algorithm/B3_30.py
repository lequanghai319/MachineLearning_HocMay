from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = load_breast_cancer()
X = data.data
y = data.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH PERCEPTRON:")
print(f"- Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"- Precision: {precision_score(y_test, y_pred):.4f}")
print(f"- Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"- F1-score:  {f1_score(y_test, y_pred):.4f}")
