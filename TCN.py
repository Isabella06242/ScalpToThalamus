import torch
import pickle
from pytorch_tcn import TCN
import numpy as np

FN_DATA = "DE_test_10\EM40_Sz18_scalp_DE_windowed.csv"
with open(FN_DATA, 'rb') as file:
    input = pickle.load(file)
x = input[:, 2:]

model = TCN(num_inputs=49981*4, num_channels=[32, 32])

out = model(x)
