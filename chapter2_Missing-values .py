import pandas as pd # Load the dataset
import numpy as np
from sklearn.impute import SimpleImputer


data = pd.read_csv('data.csv', header=None, sep=';')
print(data)

X = data.values # convert DataFrame to NumPy array

# Create an imputer object with a strategy to fill missing values
# strategy: most frequent: điền giá trị xuất hiện nhiều nhất
imp = SimpleImputer(missing_values=np.nan, strategy='mean') # tạo đối tượng thay giá trị thiếu bằng giá trị trung bình
imp.fit(X) # học giá trị trung bình của từng cột
result = imp.transform(X) # áp dụng giá trị trung bình để thay thế NaN
print(result)