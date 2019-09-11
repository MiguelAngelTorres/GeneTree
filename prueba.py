import tree
import pandas as pd

data = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4], 'label':['Alto', 'Bajo']})
tree = tree.Genetreec(data[['col1', 'col2']], data[['label']])
tree.warm()
