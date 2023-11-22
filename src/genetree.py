from src.tree import Tree
import sys
import pandas as pd
from src.utils import accuracy, auc
from sklearn.preprocessing import LabelBinarizer


class Genetree:
    tree_population = None  # Tree array with population
    data = None             # Train data
    label = None            # Train target
    label_binarizer = None

    deepness = None
    min_child_per_leaf = None
    num_tree = None

    score_function = None

    features = None         # Features
    n_features = None       # Number of features

    def __init__(self, data, label, num_tree=100, deepness=0, min_child_per_leaf=3, score_function='accuracy'):

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
        available_score_functions = ['accuracy', 'auc']
        if score_function not in available_score_functions:
            print('Exit with status 1 \n  Error while initialization Genetree - score_function must be on of ' + str(available_score_functions))
            sys.exit(1)

        self.label = label.squeeze()
        self.label_binarizer = LabelBinarizer()
        self.label_binarizer.fit(self.label)
        self.data = data
        self.score_function = score_function
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

    def score_trees(self):
        tree_score = []
        if self.score_function == 'accuracy':
            for tree in self.tree_population:
                tree_score.append(accuracy(self.label, tree.evaluate(self.data)))
        elif self.score_function == 'auc':
            for tree in self.tree_population:
                tree_score.append(auc(self.label_binarizer.fit_transform(self.label), tree.evaluate(self.data, probability=True)))

        return tree_score

    def calculate_reproductivity_score(self):
        tree_score = self.score_trees()
        reproductivity_score = pd.DataFrame({'tree': self.tree_population, 'score': tree_score})
        reproductivity_score = reproductivity_score.sort_values(by=['score'], ascending=False)

        # transform score to probabilities (interval [0,1])
        reproductivity_score['score'] -= reproductivity_score['score'].min()
        sum_score_inv = 1 / reproductivity_score['score'].sum()
        reproductivity_score['score'] *= sum_score_inv
        reproductivity_score['score'] = reproductivity_score['score'].cumsum()

        return reproductivity_score

