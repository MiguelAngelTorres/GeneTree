from numpy import mean
from src.genetree import Genetree
import pandas as pd
import time


data = pd.read_csv("iris-species/Iris.csv")
tree = Genetree(data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], data[['Species']])

times = []
for i in range(1, 20):
    tree = Genetree(data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], data[['Species']])
    start = time.time()
    tree.warm()
    end = time.time()
    times.append(end-start)

print("\nMean time elapsed on warm: " + str(mean(times)))
