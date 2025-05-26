# sklearn:
## Thuật toán học máy:	Phân lớp, regression, clustering, dimensionality reduction, v.v.
## Tiền xử lý dữ liệu:	Chuẩn hóa, biến đổi, trích xuất đặc trưng
## Đánh giá mô hình	Các chỉ số đo lường hiệu suất, phân chia dữ liệu
## Pipeline	Kết nối các bước xử lý và mô hình thành quy trình thống nhất
## ng dụng	Tài chính, y tế, marketing, AI, khoa học dữ liệu

# Thu thâp data > Xử lý data > Chia data thành train và test > Xây dựng model > Dự đoán kq > Đánh giá ? hqua ko
from sklearn import tree

mytree = tree.DecisionTreeClassifier() # Khởi tạo cây quyết định

# Đặc trưng của dữ liệu
# dactrung = [
#     ['nhe', 'tb', 'tb', 'nhieu'],
#     ['nang', 'thap', 'cao', 'it'],
#     ['nhe', 'thap', 'cao', 'it'],
#     ['nang', 'cao', 'cao', 'it'],
#     ['nhe', 'cao', 'cao', 'nhieu'],
#     ['tb', 'thap', 'tb', 'nhieu'],
#     ['tb', 'tb', 'tb', 'it'],
#     ['nang', 'thap', 'thap', 'nhieu']
# ]

# Sau khi quy ước:
dactrung = [
    [1, 3, 3, 7],
    [5, 2, 4, 6],
    [1, 2, 4, 6],
    [5, 4, 4, 3],
    [1, 4, 4, 7],
    [3, 2, 3, 7],
    [3, 3, 3, 6],
    [5, 2, 2, 7]
]

# Bệnh tym (kết quả của dựa vào các đặc trưng)
nhan = [0, 1, 1, 0, 0, 0, 0, 1]

mytree.fit(dactrung, nhan) # Huấn luyện cây quyết định với dữ liệu và nhãn

kq = mytree.predict([[3, 3, 3, 6],
                    [5, 2, 2, 7]]) # Dự đoán nhãn cho dữ liệu mới

print(kq) # In ra kết quả dự đoán