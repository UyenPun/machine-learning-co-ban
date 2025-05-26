#
#! Huấn luyện một mô hình hồi quy tuyến tính đơn để:
#* Dự đoán doanh số (Sales) dựa trên chi phí quảng cáo trên Radio (Radio advertising cost).
#* Dùng Gradient Descent để tìm ra trọng số (weight) và độ lệch (bias) sao cho mô hình dự đoán tốt nhất.

import pandas as pd # Load the dataset
import matplotlib.pyplot as plt
import numpy as np

dataframe = pd.read_csv("Advertising.csv")
# print(dataframe)

X = dataframe.values[:, 2] # Lấy cột thứ 2 (Radio) làm đặc trưng
# print(X)

y = dataframe.values[:, 4] # Lấy cột thứ 4 (Sales) làm nhãn
# print(y)

# plt.scatter(X, y, marker='o') # Vẽ biểu đồ phân tán
# plt.show()

# Dự đoán → Cho biết nếu quảng cáo radio là new_radio, thì doanh số dự đoán là gì.
def predict(new_radio, weight, bias):
    return new_radio * weight + bias # Hàm dự đoán

# Tính hàm mất mát (Mean Squared Error) – Tính sai số -> Mục tiêu của mô hình là giảm giá trị này xuống thấp nhất.
# Đo xem mô hình hiện tại (với weight/bias hiện tại) đoán sai bao nhiêu so với thực tế.
def cost_function(X, y, weight, bias):
    m = len(X) # Số lượng mẫu
    sum_error = 0
    for i in range(m):
        sum_error += (y[i] - (weight*X[i] + bias)) ** 2 # Tính tổng bình phương sai số
    return sum_error / m # Trả về giá trị trung bình của tổng bình phương sai số

# Cập nhật weight và bias theo đạo hàm (Gradient Descent):
# Áp dụng Gradient Descent: đi dần đến điểm tối ưu của hàm chi phí bằng cách đi ngược hướng đạo hàm.
def update_weight(X, y, weight, bias, learning_rate):
    m = len(X) # Số lượng mẫu
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range(m):
        weight_temp += -2 * X[i] * (y[i] - (weight * X[i] + bias)) # Tính đạo hàm theo trọng số
        bias_temp += -2 * (y[i] - (weight * X[i] + bias)) # Tính đạo hàm theo độ lệch
    weight -= learning_rate * (weight_temp / m) # Cập nhật trọng số
    bias -= learning_rate * (bias_temp / m) # Cập nhật độ lệch
    return weight, bias

# Huấn luyện mô hình:
# Lặp lại nhiều lần (60 lần) để cải thiện weight và bias
# Mỗi lần lặp:
# Cập nhật weight, bias
# Tính lại cost
# Lưu lại giá trị cost để vẽ biểu đồ
# iter: Số lần lặp
def train(X, y, weight, bias, learning_rate, iter):
    cos_history = [] # Lưu trữ hàm mất mát
    for i in range(iter):
        weight, bias = update_weight(X, y, weight, bias, learning_rate) # Cập nhật trọng số và độ lệch
        cost = cost_function(X, y, weight, bias)    # Tính hàm mất mát
        cos_history.append(cost) # xem cost co giam dan khong

    return weight, bias, cos_history

weight, bias, cos_history = train(X, y, 0.03, 0.0014, 0.001, 60) # Huấn luyện mô hình
print("Trọng số:", weight)
print("Độ lệch:", bias)
# print("cos:", cos_history)

print("predict: ", predict(19, weight, bias)) # Dự đoán doanh số bán hàng khi chi tiêu quảng cáo trên radio là 19

# Chuyển đổi mỗi phần tử sang float và nối chúng lại thành một chuỗi
# bằng ' , '
cos_string = ' , '.join([str(float(c)) for c in cos_history])
print("cos:", cos_string)

solanlap = [i for i in range (60)] # Tạo danh sách số lần lặp
plt.plot(solanlap, cos_history) # Vẽ biểu đồ hàm mất mát
plt.show()


#!
#?
#*
