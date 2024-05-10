from src.tree import Tree
from src.leaf import Leaf
from src.node import Node
import sys
import pandas as pd
from src.utils import accuracy, auc
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import polars as pl
from polars import col


class Genetree:
    tree_population = None  # Tree array with population
    data = None             # Train data as polars dataframe
    label = None            # String name of the training column in data
    label_np = None         # Train target as numpy array
    label_binarized = None  # Train target binarized with label_binarizer
    label_binarizer = None  # Auxiliar model with target binarizer
    tags_count = None       # Counter of tags for train data
    mayoritary_class = None # More frequent tag in label

    deepness = None             # Maximum number of consecutive nodes used to take a decision in a tree
    min_child_per_leaf = None   # Minimum number of train rows to split a leaf
    num_trees = None            # Number of trees in population
    num_rounds = None           # Number of rounds in genetic algorithm

    score_function = None       # Score function used to optimize the trees

    features = None             # Features
    n_features = None           # Number of features
    n_rows = None               # Number of observations

    def __init__(self, data, label, num_trees=100, num_rounds=50, deepness=1, min_child_per_leaf=3, score_function='accuracy'):

        if not isinstance(data, pd.DataFrame):
            print('Exit with status 1 \n  Error while initialization Genetree - data must be a pandas.DataFrame')
            sys.exit(1)
        if not isinstance(label, str):
            print('Exit with status 1 \n  Error while initialization Genetree - label must be a string')
            sys.exit(1)
        if data.shape[0] < 10:
            print('Exit with status 1 \n  Error while initialization Genetree - data must have at least 10 rows')
            sys.exit(1)
        if label not in data.columns:
            print('Exit with status 1 \n  Error while initialization Genetree - label must be the name of a column present in data')
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

        self.label = label
        self.label_np = data[self.label].squeeze().to_numpy()
        self.label_binarizer = LabelBinarizer()
        self.label_binarizer.fit(self.label_np)
        self.label_binarized = self.label_binarizer.transform(self.label_np)

        self.data = pl.from_pandas(data).lazy()
        self.tags_count = self.data.group_by(self.label).count().collect()
        self.mayoritary_class = self.tags_count.sort("count").select(pl.first(self.label)).item()
        self.score_function = score_function
        self.features = list(self.data.select(pl.exclude(label)).columns)
        self.n_features = len(self.features)
        self.n_rows = data.shape[0]
        self.deepness = deepness
        self.min_child_per_leaf = min_child_per_leaf
        self.num_trees = num_trees
        self.num_rounds = num_rounds

        self.tree_population = []
        for i in range(1, num_trees):
            tree = Tree(self)
            tree.warm()
            self.tree_population.append(tree)

    def train(self):
        print(sorted(self.score_trees(), reverse=True))
        for i in range(0, self.num_rounds):
            print("round: " + str(i))
            self.tree_population = self.next_generation()
            print("Max depth: %s" % [tree.depth for tree in self.tree_population])
            print(sorted(self.score_trees(), reverse=True))

    def score_trees(self):
        tree_score = []
        if self.score_function == 'accuracy':
            for tree in self.tree_population:
                tree_score.append(accuracy(self.label_np, tree.evaluate(self.data)))

        elif self.score_function == 'auc':
            for tree in self.tree_population:
                tree_score.append(auc(self.label_binarized, tree.evaluate(self.data, probability=True), self.label_binarizer.classes_))

        return tree_score

    def calculate_reproductivity_score(self):

        tree_score = self.score_trees()

        if len(np.unique(tree_score)) > 1:
            reproductivity_score = pl.LazyFrame({'tree': self.tree_population, 'score': tree_score})
            # transform score to probabilities (interval [0,1])
            reproductivity_score = reproductivity_score \
                .sort('score', descending=True) \
                .with_columns((col('score') - pl.min('score')).alias("score0")) \
                .with_columns((col("score0") / pl.sum('score0')).alias('score1')) \
                .with_columns((1 - pl.cum_sum('score1')).alias('score_order')) \
                .select(["tree", "score_order"])
        else:
            reproductivity_score = pl.LazyFrame({'tree': self.tree_population, 'score_order': [i/len(tree_score) for i in range(0,len(tree_score))]})

        return reproductivity_score

    def next_generation(self):

        reproductivity_score = self.calculate_reproductivity_score()
        probs = np.random.uniform(0, 1, self.num_trees * 2)

        next_generation = []
        for i in range(0, self.num_trees):
            a_tree = reproductivity_score.filter(col('score_order') <= probs[i]).head(1).collect().get_column('tree').item()
            b_tree = reproductivity_score.filter(col('score_order') <= probs[self.num_trees + i]).head(1).collect().get_column('tree').item()

            atree = self.crossover(a_tree, b_tree)
            next_generation.append(atree)

        return next_generation

    def crossover(self, a_tree, b_tree):
        aside, abranch, depth_abranch = a_tree.select_random_branch(self.deepness)
        bside, bbranch, depth_bbranch = b_tree.select_random_branch(self.deepness - (a_tree.depth - depth_abranch))

        tree = Tree(self)
        copying_node = a_tree.root
        tree.root = self.copy_tree(tree, copying_node, abranch, bbranch, aside, bside)

        tree.repartition(self.data.with_columns(b=True)) #TODO: Make a function to calculate depth
        tree.mutate()
        #tree.repartition(self.data.with_columns(b=True))

        return tree

    def copy_tree(self, tree, copying_node, abranch=None, bbranch=None, aside=None, bside=None):
        if copying_node == abranch:
            if aside == "left":
                if bside == "left":
                    if isinstance(bbranch.left, Leaf):
                        return Leaf(tree, None)
                    else:
                        return Node(tree, copying_node.column, copying_node.pivot,
                                    self.copy_tree(tree, bbranch.left),
                                    self.copy_tree(tree, copying_node.right))
                elif bside == "right":
                    if isinstance(bbranch.right, Leaf):
                        return Leaf(tree, None)
                    else:
                        return Node(tree, copying_node.column, copying_node.pivot,
                                    self.copy_tree(tree, bbranch.right),
                                    self.copy_tree(tree, copying_node.right))
                else:
                    return Node(tree, copying_node.column, copying_node.pivot,
                                self.copy_tree(tree, bbranch),
                                self.copy_tree(tree, copying_node.right))
            elif aside == "right":
                if bside == "left":
                    if isinstance(bbranch.left, Leaf):
                        return Leaf(tree, None)
                    else:
                        return Node(tree, copying_node.column, copying_node.pivot,
                                    self.copy_tree(tree, copying_node.left),
                                    self.copy_tree(tree, bbranch.left))
                elif bside == "right":
                    if isinstance(bbranch.right, Leaf):
                        return Leaf(tree, None)
                    else:
                        return Node(tree, copying_node.column, copying_node.pivot,
                                    self.copy_tree(tree, copying_node.left),
                                    self.copy_tree(tree, bbranch.right))
                else:
                    return Node(tree, copying_node.column, copying_node.pivot,
                                self.copy_tree(tree, copying_node.left),
                                self.copy_tree(tree, bbranch))

            else:
                if bside == "left":
                    if isinstance(bbranch.left, Leaf):
                        return Node(tree, bbranch.column, bbranch.pivot,
                                    self.copy_tree(tree, bbranch.left),
                                    self.copy_tree(tree, bbranch.right))
                    else:
                        return Node(tree, bbranch.left.column, bbranch.left.pivot,
                                    self.copy_tree(tree, bbranch.left.left),
                                    self.copy_tree(tree, bbranch.left.right))
                elif bside == "right":
                    if isinstance(bbranch.right, Leaf):
                        return Node(tree, bbranch.column, bbranch.pivot,
                                    self.copy_tree(tree, bbranch.left),
                                    self.copy_tree(tree, bbranch.right))
                    else:
                        return Node(tree, bbranch.right.column, bbranch.right.pivot,
                                    self.copy_tree(tree, bbranch.right.left),
                                    self.copy_tree(tree, bbranch.right.right))
                else:
                    return Node(tree, bbranch.column, bbranch.pivot,
                                self.copy_tree(tree, bbranch.left),
                                self.copy_tree(tree, bbranch.right))
        else:
            if isinstance(copying_node, Leaf):
                return Leaf(tree, None)
            else:
                return Node(tree, copying_node.column, copying_node.pivot,
                            self.copy_tree(tree, copying_node.left, abranch, bbranch, aside, bside),
                            self.copy_tree(tree, copying_node.right, abranch, bbranch, aside, bside))
