from src.node import Node
from src.utils import entropy
from random import sample
from numpy import logical_and, int64, float64
from numpy.random import uniform
import pandas as pd
import polars as pl
import sys


class Leaf:
    tree = None 			# Tree
    tags_count = None       # Counter of tags for train data
    partition = None		# Boolean vector that mask the initial data belonging to the leaf

    def __init__(self, tree, partition):
        self.tree = tree
        self.partition = partition

    # Split the data into two new leaves
    def warm(self, levels):
        ret_node = self
        if sum(self.partition) < self.tree.genetree.min_child_per_leaf * 2:  # min data on leaf to split multipled by num of branches (2)
            return ret_node

        criteria = None
        shuffle_columns = sample(self.tree.genetree.features, self.tree.genetree.n_features)  # random column to split

        for column in shuffle_columns:
            (criteria, pivot) = self.select_pivot(column)
            if criteria is not None:  # If good split
                left = Leaf(self.tree, logical_and(criteria,self.partition))
                right = Leaf(self.tree, logical_and(~criteria, self.partition))

                if levels > 1:
                    right = right.warm(levels-1)
                    left = left.warm(levels-1)
                ret_node = Node(self.tree, column, pivot, right, left)
                break

        if criteria is None:
            if levels == self.tree.genetree.deepness:  # First node
                print('Exit with status 1 \n  Error while initialization tree - the first branch cannot be generated because the data is not splitable')
                sys.exit(1)

        return ret_node

    # Look for the pivot with best split, depending of the generated entropy
    def select_pivot(self, column):
        split_column = self.tree.genetree.data.select(pl.col(column)).collect().get_column(column).to_numpy()
        if split_column.dtype == float64 or split_column.dtype == int64:
            splited_column = split_column[self.partition]
            max_val = splited_column.min()
            min_val = splited_column.max()
            if split_column.dtype == int64:
                grill = sample(range(min_val, max_val), 10)  # create pivot grill for int
            else:
                grill = uniform(min_val, max_val, 10)  # create pivot grill for float

            grill_entropy = []

            total = sum(self.partition)
            total_inverse = 1 / total
            classes = list(dict.fromkeys(self.tree.genetree.label.unique()))
            for x in grill:
                left_split = split_column < x
                right_split = logical_and(~left_split, self.partition)
                left_split = logical_and(left_split, self.partition)
                n_left = sum(left_split)
                n_right = total - n_left

                if n_left < self.tree.genetree.min_child_per_leaf or n_right < self.tree.genetree.min_child_per_leaf:  # low data to split
                    l_entropy = 0.5
                    r_entropy = 0.5
                else:
                    r_entropy = 1
                    l_entropy = 1

                    for clas in classes:
                        r_entropy += entropy(sum(self.tree.genetree.label[right_split] == clas) / n_right)
                        l_entropy += entropy(sum(self.tree.genetree.label[left_split] == clas) / n_left)
                    r_entropy = n_right * total_inverse * r_entropy
                    l_entropy = n_left * total_inverse * l_entropy

                grill_entropy.append(l_entropy + r_entropy)

            pivot = grill[grill_entropy.index(min(grill_entropy))]  # best pivot
            criteria = split_column < pivot  # builds the next mask
            left_count = sum(logical_and(criteria, self.partition))
            if left_count < self.tree.genetree.min_child_per_leaf or total-left_count < self.tree.genetree.min_child_per_leaf:  # low data to split
                return None, None

            return criteria, pivot  # return mask and pivot

    # Select the tag the leaf will have
    def set_leaf_tag(self):
        value_counts = self.tree.genetree.label[self.partition].value_counts()
        self.tags_count = [value_counts[label] if label in value_counts.index else 0 for label in self.tree.genetree.label_binarizer.classes_]

# Return the expected class
    def evaluate(self, tree, criteria, probability=False):
        total = sum(self.tags_count)   # If total is 0, then the leaf has no train data
        if probability:
            if total != 0:
                probabilities = [[v / total for v in self.tags_count]] * len(criteria)
            else:    # give frequency of tag in train data
                probabilities = [[v / len(criteria) for v in self.tree.genetree.tags_count]] * len(criteria)
            return probabilities
        else:
            if total != 0:
                return self.tree.genetree.label_binarizer.classes_[pd.Series(self.tags_count).idxmax()] * len(criteria)
            else:    # give most frequent tag in train data
                return self.tree.genetree.label_binarizer.classes_[pd.Series(self.tree.genetree.tags_count).idxmax()] * len(criteria)

    # Plot the try, on terminal by now
    def plot(self):
        print(self.tags_count)
        return None

    # The selection arrived to a leaf, so return the parent of that leaf
    def select_random_branch(self):
        return None, False

    def get_num_nodes(self):
        return 0

    def mutate(self):
        return

    def repartition(self, partition):
        self.partition = partition
        self.set_leaf_tag()
        return
