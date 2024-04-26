from src.leaf import Leaf
from random import randrange
import numpy as np


class Tree:
    genetree = None            # Parent object
    root = None                # First Node
    depth = None               # Depth of nodes

    def __init__(self, genetree):
        self.genetree = genetree

    def warm(self):
        self.root = Leaf(self, self.genetree.data.with_columns(b=True))
        self.root = self.root.warm(self.genetree.deepness)
        self.depth = self.root.set_leaf_tag()

    def repartition(self, partition):
        self.depth = self.root.repartition(partition)
        return

    def evaluate(self, data, probability=False):
        return self.root.evaluate(self, np.array([True] * data.collect().shape[0]), probability)

    def select_random_branch(self, min_depth):
        go_deep = min_depth < self.depth
        if go_deep:
            r = randrange(2)
        else:
            r = randrange(3)

        if r == 0:
            if go_deep and isinstance(self.root.left, Leaf):
                r = 1
        if r == 1:
            if go_deep and isinstance(self.root.right, Leaf):
                r = 0

        if r == 0:
            last_branch_side, last_branch_father, depth_branch = self.root.left.select_random_branch(min_depth)
            if isinstance(last_branch_father, bool):  # Si el elegido es el hijo
                if last_branch_father:
                    last_branch_side = "left"
                else:
                    last_branch_side = 'root'
                    depth_branch += 1
                last_branch_father = self.root
        elif r == 1:
            last_branch_side, last_branch_father, depth_branch = self.root.right.select_random_branch(min_depth)
            if isinstance(last_branch_father, bool):  # Si el elegido es el hijo
                if last_branch_father:
                    last_branch_side = "right"
                else:
                    last_branch_side = 'root'
                    depth_branch += 1
                last_branch_father = self.root
        else:
            last_branch_side = "root"
            last_branch_father = self.root
            depth_branch = self.root.depth

        return last_branch_side, last_branch_father, depth_branch

    def get_num_nodes(self):
        return self.root.get_num_nodes()

    def mutate(self):
        self.root.mutate()
        self.root.repartition([True] * self.genetree.n_rows)
        return

    def plot(self):
        self.root.plot()
        return
