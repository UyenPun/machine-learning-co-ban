import pandas as pd # Load the dataset
import numpy as np

# header=None: header của các cột là 0 1 2 ...
df = pd.read_csv('mapping.csv', header=None) # read data from CSV file
print(df)
print(df[2]) # print the 2rd column of the dataframe
print(df[2].values) # print the values of the 3rd column

# Save column 2 to a new CSV file
dff = df[2] # select the 3rd column
df.to_csv('mapping2.csv')

