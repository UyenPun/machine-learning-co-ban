import torch
import numpy as np

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

print('x_data:', x_data)

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html