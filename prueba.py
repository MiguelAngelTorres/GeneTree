from src.genetree import Genetree
import pandas as pd
import time


data = pd.read_csv("iris-species/Iris.csv")

start = time.time()
genetree = Genetree(data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], data[['Species']],
                    score_function="auc", num_trees=100, deepness=4, num_rounds=5)
end = time.time()
print("\nMean time elapsed on warm: " + str(end-start))
