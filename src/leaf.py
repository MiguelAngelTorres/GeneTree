from src.node import Node
from src.utils import entropy
from random import sample
from numpy.random import uniform
import pandas as pd
import polars as pl
import sys


class Leaf:
    tree = None 			# Tree
    tags_count = None       # Counter of tags for train data
    partition = None		# Polars Series that mask the initial data belonging to the leaf
    depth = 0               # Depth of nodes

    def __init__(self, tree, partition):
        self.tree = tree
        self.partition = partition

    # Split the data into two new leaves
    def warm(self, levels):
        ret_node = self
        if self.partition.select(pl.len()).collect().item() < self.tree.genetree.min_child_per_leaf * 2:  # min data on leaf to split multipled by num of branches (2)
            return ret_node

        partition_left = None
        shuffle_columns = sample(self.tree.genetree.features, self.tree.genetree.n_features)  # random column to split

        for column in shuffle_columns:
            partition_left, partition_right, pivot = self.select_pivot(column)
            if partition_left is not None:  # If good split
                left = Leaf(self.tree, partition_left)
                right = Leaf(self.tree, partition_right)

                if levels > 1:
                    right = right.warm(levels-1)
                    left = left.warm(levels-1)
                ret_node = Node(self.tree, column, pivot, right, left)
                break

        if partition_left is None:
            if levels == self.tree.genetree.deepness:  # First node
                print('Exit with status 1 \n  Error while initialization tree - the first branch cannot be generated because the data is not splitable')
                sys.exit(1)

        return ret_node

    # Look for the pivot with best split, depending of the generated entropy
    def select_pivot(self, column):
        split_column_type = self.partition.select(column).dtypes[0]
        if split_column_type == pl.Float64 or split_column_type == pl.Int64:
            max_val = self.partition.select(column).max().collect().item()
            min_val = self.partition.select(column).min().collect().item()

            if min_val == max_val:
                return None, None, None

            if split_column_type == pl.Int64:
                grill = sample(range(min_val, max_val), min(10, max_val-min_val))  # create pivot grill for int
            else:
                grill = uniform(min_val, max_val, 10)  # create pivot grill for float

            grill_entropy = []

            total = self.partition.select(pl.len()).collect().item()
            total_inverse = 1 / total
            classes = list(dict.fromkeys(self.tree.genetree.tags_count.get_column(self.tree.genetree.label).to_numpy()))
            for x in grill:
                left_split = self.partition.with_columns(a=(pl.col(column) < x)).with_columns(b=(pl.col("a") & pl.col("b"))).filter(pl.col('b'))
                right_split = self.partition.with_columns(a=(pl.col(column) >= x)).with_columns(b=(pl.col("a") & pl.col("b"))).filter(pl.col('b'))
                n_left = left_split.select(pl.len()).collect().item()
                n_right = total - n_left

                if n_left < self.tree.genetree.min_child_per_leaf or n_right < self.tree.genetree.min_child_per_leaf:  # low data to split
                    l_entropy = 0.5
                    r_entropy = 0.5
                else:
                    r_entropy = 1
                    l_entropy = 1

                    for clas in classes:
                        r_entropy += entropy(right_split.filter(pl.col(self.tree.genetree.label).eq(clas)).select(pl.len()).collect().item() / n_right)
                        l_entropy += entropy(left_split.filter(pl.col(self.tree.genetree.label).eq(clas)).select(pl.len()).collect().item() / n_left)
                    r_entropy = n_right * total_inverse * r_entropy
                    l_entropy = n_left * total_inverse * l_entropy

                grill_entropy.append(l_entropy + r_entropy)

            # best pivot
            pivot = grill[grill_entropy.index(min(grill_entropy))]
            # builds the next mask
            partition_left = self.partition.with_columns(a=(pl.col(column) < pivot)).with_columns(b=(pl.col("a") & pl.col("b")))
            partition_right = self.partition.with_columns(a=(pl.col(column) < pivot)).with_columns(b=(pl.col("a") & pl.col("b")))
            left_count = partition_left.filter(pl.col("b")).select(pl.len()).collect().item()
            if left_count < self.tree.genetree.min_child_per_leaf or total-left_count < self.tree.genetree.min_child_per_leaf:  # low data to split
                return None, None, None

            return partition_left, partition_right, pivot  # return mask and pivot

    # Select the tag the leaf will have
    def set_leaf_tag(self):
        value_counts = self.partition.filter(pl.col("b")).group_by(self.tree.genetree.label).count().collect()
        self.tags_count = [value_counts.filter(pl.col(self.tree.genetree.label).eq(label)).get_column("count").item()
                           if label in value_counts.get_column(self.tree.genetree.label).to_numpy() else 0
                           for label in self.tree.genetree.label_binarizer.classes_]
        return 0

    # Return the expected class
    def evaluate(self, tree, criteria, probability=False):
        total = sum(self.tags_count)   # If total is 0, then the leaf has no train data
        if probability:
            if total != 0:
                probabilities = [[v / total for v in self.tags_count]] * len(criteria)
            else:    # give frequency of tag in train data
                probabilities = [[v / len(criteria) for v in self.tree.genetree.tags_count.get_column("count").to_numpy()]] * len(criteria)
            return probabilities
        else:
            if total != 0:
                return [self.tree.genetree.label_binarizer.classes_[pd.Series(self.tags_count).idxmax()]] * len(criteria)
            else:    # give most frequent tag in train data
                return [self.tree.genetree.mayoritary_class] * len(criteria)

    # Plot the try, on terminal by now
    def plot(self):
        print(self.tags_count)
        return None

    # The selection arrived to a leaf, so return the parent of that leaf
    def select_random_branch(self, min_depth):
        return None, False, 0

    def get_num_nodes(self):
        return 0

    def mutate(self):
        return

    def repartition(self, partition):
        self.partition = partition
        self.set_leaf_tag()
        return 0
