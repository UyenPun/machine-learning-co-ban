import matplotlib.pyplot as plt
import numpy as np

x = [3, 5]
y = [7, 9]

plt.plot(x, y)
print(plt.show()) # hiển thị đồ thị

# ----------------------------------------------------------------

image = np.random.rand(30, 30) # tạo ảnh ngẫu nhiên 30x30
plt.imshow(image) # vẽ ma trận này thành một bức ảnh, mỗi ô là một pixel có màu sắc tương ứng với giá trị trong ma trận
plt.show() # hiển thị ảnh

