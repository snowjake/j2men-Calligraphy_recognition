import tensorflow as tf
import numpy as np
import pandas as pd

new_data = pd.read_csv("image.csv")
new_data = new_data.values.astype(np.float32)
print(new_data)
np.random.shuffle(new_data)
print(new_data)
#new_data = new_data.reshape(-1,16385)
test_x = new_data[:, 1:]
test_y = new_data[:, :1]
print(test_x)
print(test_y)
