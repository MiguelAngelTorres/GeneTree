from src.tree import Tree
from src.leaf import Leaf
from src.node import Node
import sys
import pandas as pd
from src.utils import accuracy, auc
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import time


class Genetree:
    tree_population = None  # Tree array with population
    data = None             # Train data
    label = None            # Train target
    label_binarizer = None
    tags_count = None       # Counter of tags for train data

    deepness = None
    min_child_per_leaf = None
    num_trees = None
    num_rounds = None

    score_function = None

    features = None         # Features
    n_features = None       # Number of features

    def __init__(self, data, label, num_trees=100, num_rounds=50, deepness=1, min_child_per_leaf=3, score_function='accuracy'):

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
        if deepness < 1:
            print('Exit with status 1 \n  Error while initialization Genetree - deepness must be greater than 0')
            sys.exit(1)
        if num_trees < 2:
            print('Exit with status 1 \n  Error while initialization Genetree - num_trees must be greater than 1')
            sys.exit(1)
        if num_rounds < 1:
            print('Exit with status 1 \n  Error while initialization Genetree - num_rounds must be greater than 0')
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
        value_counts = self.label.value_counts()
        self.tags_count = [value_counts[label] if label in value_counts.index else 0 for label in self.label_binarizer.classes_]
        self.data = data
        self.score_function = score_function
        self.features = list(data.columns)
        self.n_features = len(self.features)
        self.deepness = deepness
        self.min_child_per_leaf = min_child_per_leaf
        self.num_trees = num_trees
        self.num_rounds = num_rounds

        self.tree_population = []
        for i in range(1, num_trees):
            tree = Tree(self)
            tree.warm()
            self.tree_population.append(tree)

        print(np.mean(self.score_trees()))
        for i in range(0, num_rounds):
            print("ronda: " + str(i))
            self.tree_population = self.next_generation()
            print(len(self.tree_population))
            print(np.mean(self.score_trees()))

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
        reproductivity_score['score'] = 1 - reproductivity_score['score']

        return reproductivity_score

    def next_generation(self):

        reproductivity_score = self.calculate_reproductivity_score().sort_values(by=['score'], ascending=False)
        probs = np.random.uniform(0, 1, self.num_trees * 2)


        next_generation = []
        a_times = []
        b_times = []
        c_times = []
        for i in range(0, self.num_trees):
            a_tree = reproductivity_score.loc[reproductivity_score.score <= probs[i]].iloc[0].tree
            b_tree = reproductivity_score.loc[reproductivity_score.score <= probs[self.num_trees + i]].iloc[0].tree

            a = time.time()
            print("Dimensions: " + str(a_tree.get_num_nodes()) + ' - ' + str(b_tree.get_num_nodes()))
            atree = self.crossover(a_tree, b_tree)
            b = time.time()
            b_times.append(b-a)
            next_generation.append(atree)

        print("a time:" + str(sum(a_times)))
        print("b time:" + str(sum(b_times)))
        print("c time:" + str(sum(c_times)))
        return next_generation

    def crossover(self, a_tree, b_tree):  # TODO: Deepcopy causes memory costs - nodes have references to original tree
        aside, abranch = a_tree.select_random_branch()
        bside, bbranch = b_tree.select_random_branch()

        tree = Tree(self)
        copying_node = a_tree.root
        tree.root = self.copy_tree(tree, copying_node, abranch, bbranch, aside, bside)

        tree.root.repartition([True] * self.data.shape[0])

        return a_tree

    def copy_tree(self, tree, copying_node, abranch, bbranch, aside, bside):
        if copying_node == abranch:
            if aside == "left":
                if bside == "left":
                    if isinstance(bbranch.left, Leaf):
                        return Leaf(tree, None)
                    else:
                        return Node(tree, copying_node.column, copying_node.pivot,
                                    self.copy_tree(tree, copying_node.right, abranch, bbranch, aside, bside),
                                    self.copy_tree(tree, bbranch.left, abranch, bbranch, aside, bside))
                elif bside == "right":

        else:
            if isinstance(copying_node, Leaf):
                return Leaf(tree, None)
            else:
                return Node(tree, copying_node.column, copying_node.pivot,
                            self.copy_tree(tree, copying_node.right, abranch, bbranch, aside, bside),
                            self.copy_tree(tree, copying_node.left, abranch, bbranch, aside, bside))
