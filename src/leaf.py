from src.node import Node
from random import sample
import numpy as np
import math
import sys


def entropy(v):           # v es la proporcion de la clase (frec/total)
    if v == 0 or v == 1:
        return 0
    return v * math.log(v, 2)


class Leaf:
    root = None 			# Tree's root
    tag = None 				# Decision to take
    partition = None		# Boolean vector that mask the initial data belonging to the leaf

    def __init__(self, root, partition):
        self.root = root
        self.partition = partition

    # Split the data into two new leaves
    def warm(self, levels):
        ret_node = self
        if sum(self.partition) < 3 * 2:  # Min data on leaf to split multipled by num of branches (2) # TODO: use a parameter instead of hardcoded number
            return ret_node

        criteria = None
        shuffle_columns = sample(self.root.features, self.root.n_features)  # random column to split

        for column in shuffle_columns:
            (criteria, pivot) = self.select_pivot(column)
            if criteria is not None:  # If good split
                right = Leaf(self.root, criteria & self.partition)
                left = Leaf(self.root, ~criteria & self.partition)

                if levels > 1:
                    right = right.warm(levels-1)
                    left = left.warm(levels-1)
                ret_node = Node(self.root, column, pivot, right, left)
                break

        if criteria is None:
            if levels == self.root.deepness:  # First node
                print('Exit with status 1 \n  Error while initialization tree - the first branch cannot be generated because the data is not splitable')
                sys.exit(1)

        return ret_node

    # Look for the pivot with best split, depending of the generated entropy
    def select_pivot(self, column):
        split_column = self.root.data[column]
        if split_column.dtype == np.float64 or split_column.dtype == np.int64:
            max_val = split_column[self.partition].min()
            min_val = split_column[self.partition].max()
            if split_column.dtype == np.int64:
                grill = [math.ceil((max_val - min_val)*(x/10)+min_val) for x in range(1, 10)]  # create pivot grill for int
                grill = list(dict.fromkeys(grill))
            else:
                grill = [(max_val - min_val)*(x/10)+min_val for x in range(1, 10)]  # create pivot grill for float

            grill_entropy = []

            total = sum(self.partition)
            total_inverse = 1 / total
            classes = list(dict.fromkeys(self.root.label.unique()))
            for x in grill:
                n_left = sum(split_column[self.partition] < x)
                n_right = sum(split_column[self.partition] >= x)

                if n_left < 3 or n_right < 3:  # low data to split # TODO: use a parameter instead of hardcoded number
                    l_entropy = 0.5
                    r_entropy = 0.5
                else:
                    r_entropy = 1  # TODO: Review this hardcoded number
                    l_entropy = 1

                    for clas in classes:
                        r_entropy += entropy(sum((split_column[self.partition] >= x) & self.root.label[self.partition] == clas) / n_right)
                        l_entropy += entropy(sum((split_column[self.partition] < x) & self.root.label[self.partition] == clas) / n_left)
                    r_entropy = n_right * total_inverse * r_entropy
                    l_entropy = n_left * total_inverse * l_entropy

                grill_entropy.append(l_entropy + r_entropy)

            min_index = grill_entropy.index(min(grill_entropy))
            pivot = grill[min_index]  # best pivot
            criteria = split_column < pivot  # builds the next mask
            left_count = sum(criteria & self.partition)
            if left_count < 3 or sum(self.partition)-left_count < 3:  # low data to split
                return None, None

            return criteria, pivot  # return mask and pivot

    # Select the tag the leaf will have
    def set_leaf_tag(self):
        self.tag = self.root.label.mask(self.partition).value_counts().idxmax()

    # Return the expected class
    def evaluate(self, row):
        return self.tag

    # Plot the try, on terminal by now
    def plot(self):
        print(str(self.tag) + ' with ' + str(sum(self.partition)) + ' observations')
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
