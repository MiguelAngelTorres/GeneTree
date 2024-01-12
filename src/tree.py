from src.leaf import Leaf
from random import randrange
import numpy as np

class Tree:
    genetree = None            # Parent object
    root = None                # First Node

    def __init__(self, genetree):
        self.genetree = genetree

    def warm(self):
        self.root = Leaf(self, np.array([True] * self.genetree.n_rows))
        self.root = self.root.warm(self.genetree.deepness)
        self.root.set_leaf_tag()

    def evaluate(self, data, probability=False):
        return self.root.evaluate(self, np.array([True] * data.collect().shape[0]), probability)

    def select_random_branch(self):
        r = randrange(5)
        if r == 0 or r == 2:
            last_branch_side, last_branch_father = self.root.left.select_random_branch()
            if isinstance(last_branch_father, bool):  # Si el elegido es el hijo
                if last_branch_father:
                    last_branch_side = "left"
                else:
                    last_branch_side = 'root'
                last_branch_father = self.root
        elif r == 1 or r == 3:
            last_branch_side, last_branch_father = self.root.right.select_random_branch()
            if isinstance(last_branch_father, bool):  # Si el elegido es el hijo
                if last_branch_father:
                    last_branch_side = "right"
                else:
                    last_branch_side = 'root'
                last_branch_father = self.root
        else:
            last_branch_side = "root"
            last_branch_father = self.root

        return last_branch_side, last_branch_father

    def get_num_nodes(self):
        return self.root.get_num_nodes()

    def mutate(self):
        self.root.mutate()
        self.root.repartition([True] * self.genetree.n_rows)
        return

    def plot(self):
        self.root.plot()
        return
