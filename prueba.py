import tree
import pandas as pd

data = pd.read_csv("iris-species/Iris.csv")
tree = tree.Genetreec(data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], data[['Species']])
tree.warm()
tree.root.plot()
