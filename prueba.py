from src.genetree import Genetree
import pandas as pd
import time


data = pd.read_csv("iris-species/Iris.csv")

start = time.time()
genetree = Genetree(data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], data[['Species']])
end = time.time()
print("\nMean time elapsed on warm: " + str(end-start))

genetree.calculate_reproductivity_score()

print(genetree.tree_population[0].evaluate(data, True))