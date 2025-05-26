from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split # Hàm chia dts thành 75% train và 25% test => se return 4 tham so
from sklearn.tree import DecisionTreeClassifier # Thuật toán phân loại cây quyết định

iris_dataset = load_iris()

print(iris_dataset.data) # in ra các đặc trưng của dữ liệu
print(iris_dataset.feature_names) # in ra tên các đặc trưng
print(iris_dataset.target) # với mỗi dòng dữ liệu có nhãn là gì ?
print(len(iris_dataset.target)) # kiem tra số lượng nhãn
print(iris_dataset.target_names) # in ra tên các nhãn

# random_state=0 # mỗi lần chạy sẽ cho kết quả giống nhau =! # random_state=1 # mỗi lần chạy sẽ cho kết quả khác nhau
X_train, X_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, random_state=0)
print(X_train)
print(type(X_train))

model = DecisionTreeClassifier()
mymodel = model.fit(X_train, y_train) # Huấn luyện mô hình với dữ liệu huấn luyện

print(mymodel.predict(X_test)) # Dự đoán nhãn của dữ liệu test
print(mymodel.score(X_test, y_test)) # Đánh giá mô hình với dữ liệu test

# Dự đoán nhãn của một mẫu mới
print(mymodel.predict([[5, 2.9, 1, 0.2]])) # Dự đoán nhãn của một mẫu mới