from random import randrange
from numpy.random import normal
from numpy import where
import polars as pl


class Node:
    tree = None       # Tree
    column = None     # Column name to split
    pivot = None      # Pivot to split data

    right = None      # Node or Leaf positive
    left = None       # Node or Leaf negative

    def __init__(self, tree, column, pivot, right, left):
        self.tree = tree
        self.column = column
        self.pivot = pivot
        self.right = right
        self.left = left

    def set_leaf_tag(self):
        self.right.set_leaf_tag()
        self.left.set_leaf_tag()

    def evaluate(self, tree, criteria, probability=False):
        splits = tree.genetree.data.select(a=(pl.col(self.column) < self.pivot)).with_columns(b=(pl.col("a") & (criteria)),c=(~pl.col("a") & (criteria))).collect().get_columns()

        left_split_fix = [[x] * len(self.tree.genetree.label_binarizer.classes_) for x in splits[0]]

        return where(left_split_fix, self.left.evaluate(tree, splits[0], probability), self.right.evaluate(tree, splits[1], probability))

    def plot(self):
        print('---- Column ' + self.column + ' < ' + str(self.pivot) + ' ----')
        self.left.plot()
        print('\n')
        print('---- Column ' + self.column + ' >= ' + str(self.pivot) + ' ----')
        self.right.plot()

    def select_random_branch(self):
        r = randrange(3)
        if r == 0:  # Elegida rama izq
            side, father = self.left.select_random_branch()
            if isinstance(father, bool):
                if father: 		# If a son is the chosen one
                    return "left", self
                else:					# If son is a leaf
                    return None, True
            return side, father				# If chosen one is deep
        if r == 2:  # Elegida rama der
            side, father = self.right.select_random_branch()
            if isinstance(father, bool):
                if father: 		# If chosen one is a son
                    return "right", self
                else:					# If son is a leaf
                    return None, True
            return side, father				# If chosen one is deep
        if r == 1:
            return None, True

    def get_num_nodes(self):
        return self.left.get_num_nodes() + self.right.get_num_nodes() + 1

    # TODO
    def mutate(self):
        r = randrange(5)
        if r == 0:
            self.pivot = normal(self.pivot, abs(self.pivot/4))
        if r == 1:
            columns = self.tree.data.columns
            columns = columns[columns != self.column]
            self.column = columns[randrange(len(columns))]
            self.pivot = self.tree.data[self.column].mean()

        self.left.mutate()
        self.right.mutate()
        return

    def repartition(self, partition):
        splits = self.tree.genetree.data.select(a=(pl.col(self.column) < self.pivot)).with_columns(b=(pl.col("a") & (partition)),c=(~pl.col("a") & (partition))).collect().get_columns()
        self.left.repartition(splits[0])
        self.right.repartition(splits[1])
        return
