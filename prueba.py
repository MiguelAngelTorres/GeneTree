from numpy import mean
from src.genetree import Genetree
import pandas as pd
import time


data = pd.read_csv("iris-species/Iris.csv")

times = []
start = time.time()
genetree = Genetree(data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], data[['Species']])
end = time.time()
times.append(end-start)

print("\nMean time elapsed on warm: " + str(mean(times)/100))