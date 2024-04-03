from random import randrange
from numpy.random import normal
from numpy import asarray, putmask, repeat
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
        splits = tree.genetree.data.select(a=(pl.col(self.column) < self.pivot)).with_columns(b=(pl.col("a") & (criteria)),c=((~pl.col("a")) & (criteria))).collect().get_columns()

        split_left = splits[0].to_numpy()
        split_right = splits[1].to_numpy()

        left = self.left.evaluate(tree, split_left, probability)
        right = self.right.evaluate(tree, split_right, probability)

        if probability:
            output_1 = asarray([left[0]] * len(criteria), dtype=object)
            splits_left = asarray(repeat(split_left, len(self.tree.genetree.label_binarizer.classes_)).reshape(-1, len(self.tree.genetree.label_binarizer.classes_)))
            splits_right = asarray(repeat(split_right, len(self.tree.genetree.label_binarizer.classes_)).reshape(-1, len(self.tree.genetree.label_binarizer.classes_)))
        else:
            output_1 = asarray([left[0]] * len(criteria), dtype=object)
            splits_left = asarray(split_left)
            splits_right = asarray(split_right)

        putmask(output_1, splits_left, left)
        putmask(output_1, splits_right, right)

        return output_1

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
        self.left.repartition(partition.with_columns(a=(pl.col(self.column) < self.pivot)).with_columns(b=(pl.col("a") & pl.col("b"))))
        self.right.repartition(partition.with_columns(a=(pl.col(self.column) >= self.pivot)).with_columns(b=(pl.col("a") & pl.col("b"))))

        return
