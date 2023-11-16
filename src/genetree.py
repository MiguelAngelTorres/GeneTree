from src.tree import Tree
import sys
import pandas as pd


class Genetree:
    tree_population = None  # Tree array with population
    data = None             # Train data
    label = None            # Train target

    deepness = None
    min_child_per_leaf = None
    num_tree = None

    features = None         # Features
    n_features = None       # Number of features

    def __init__(self, data, label, num_tree=100, deepness=0, min_child_per_leaf=3):

        if not isinstance(data, pd.DataFrame):
            print('Exit with status 1 \n  Error while initialization Genetree - data must be a pandas.DataFrame')
            sys.exit(1)
        if not isinstance(label, pd.DataFrame):
            print('Exit with status 1 \n  Error while initialization Genetree - label must be a pandas.DataFrame')
            sys.exit(1)
        if data.shape[0] < 10:
            print('Exit with status 1 \n  Error while initialization Genetree - data must have at least 10 rows')
            sys.exit(1)
        if len(label.columns) != 1:
            print('Exit with status 1 \n  Error while initialization Genetree - label must have a single column with label values')
            sys.exit(1)
        if data.shape[0] != label.shape[0]:
            print('Exit with status 1 \n  Error while initialization Genetree - the data and label rows cant be different')
            sys.exit(1)
        if deepness < 0:
            print('Exit with status 1 \n  Error while initialization Genetree - deepness must be greater than 0')
            sys.exit(1)
        if min_child_per_leaf <= 0:
            print('Exit with status 1 \n  Error while initialization Genetree - min_child_per_leaf must be greater than 0')
            sys.exit(1)

        self.label = label.squeeze()
        self.data = data
        self.features = list(data.columns)
        self.n_features = len(self.features)
        if deepness == 0:
            self.deepness = self.n_features
        else:
            self.deepness = deepness
        self.min_child_per_leaf = min_child_per_leaf
        self.num_tree = num_tree

        self.tree_population = []
        for i in range(1, num_tree):
            tree = Tree(self)
            self.tree_population.append(tree)

