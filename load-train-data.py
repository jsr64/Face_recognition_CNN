import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("data/train.csv")

X = dataset[dataset.columns[1:]].values
Y = dataset[dataset.columns[0:1]].values

print(X)
print(Y)