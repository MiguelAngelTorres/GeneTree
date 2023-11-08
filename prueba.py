from src.genetree import Genetree
import pandas as pd

data = pd.read_csv("iris-species/Iris.csv")
tree = Genetree(data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], data[['Species']])
tree.warm()
tree.root.plot()
