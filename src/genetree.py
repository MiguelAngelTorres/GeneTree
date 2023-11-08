from src.leaf import Leaf
import sys
import pandas as pd
from random import randrange


class Genetree:
    root = None      # First Node
    data = None      # DataFrame where columns are saved
    label = None     # DataFrame where label for each row are saved
    deepness = None  # Max deepness of tree

    def __init__(self, data, label, deepness = 0):
        self.data = data
        self.label = label
        self.deepness = deepness
        if not isinstance(self.data, pd.DataFrame):
            print('Exit with status 1 \n  Error while initialization tree - data must be a pandas.DataFrame')
            sys.exit(1)
        if not isinstance(self.label, pd.DataFrame):
            print('Exit with status 1 \n  Error while initialization tree - label must be a pandas.DataFrame')
            sys.exit(1)
        if self.data.shape[0] < 10:
            print('Exit with status 1 \n  Error while initialization tree - data must have at least 10 rows')
            sys.exit(1)
        if len(self.label.columns) != 1:
            print('Exit with status 1 \n  Error while initialization tree - label must have a single column with label values')
            sys.exit(1)
        if self.data.shape[0] != self.label.shape[0]:
            print('Exit with status 1 \n  Error while initialization tree - the data and label rows cant be different')
            sys.exit(1)
        if self.deepness == 0:
            self.deepness = len(data.columns)

    def warm(self):
        self.root = Leaf(self, [True] * self.data.shape[0])
        self.root = self.root.warm(self.deepness)
        self.root.set_leaf_tag()

    def evaluate(self, row):
        return self.root.evaluate(row)

    def select_random_branch(self):
        r = randrange(5)
        if r == 0 or r == 2:
            last_branch_side, last_branch_father = self.root.left.select_random_branch()
            if isinstance(last_branch_father, bool):  # Si el elegido es el hijo
                last_branch_side = "left"
                last_branch_father = self.root
        elif r == 1 or r == 3:
            last_branch_side, last_branch_father = self.root.right.select_random_branch()
            if isinstance(last_branch_father, bool):  # Si el elegido es el hijo
                last_branch_side = "right"
                last_branch_father = self.root
        else:
            last_branch_side = "root"
            last_branch_father = self

        return last_branch_side, last_branch_father

    def get_num_nodes(self):
        return self.root.get_num_nodes()

    def mutate(self):
        self.root.mutate()
        self.root.repartition([True] * self.data.shape[0])
        return
