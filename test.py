"""import os

classes = {}
index = 0

for className in os.listdir():
    classes[className] = index
    index += 1
    
print(classes)"""

import numpy as np
np.set_printoptions(suppress=True)

data = np.loadtxt("data_small.txt", delimiter = ',', dtype=float)
print(data)
print()


def getData():
    perm = np.random.permutation(data.shape[0])
    data_shuffled = data[perm, :]
    data_train = data_shuffled[:int(data.shape[0]*0.8)]
    data_test = data_shuffled[int(data.shape[0]*0.8):, :]
    return data_train, data_test