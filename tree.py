from random import randrange
from numpy.random import normal
import pandas as pd
import numpy as np
import math
import sys


def entropy(v):           # v es la proporcion de la clase (frec/total)
	if v == 0 or v == 1:
		return 0
	return v * math.log(v, 2)


class Genetreec:
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


class Node:
	root = None       # Tree's root
	column = None     # Column name to split
	pivot = None      # Pivot to split data

	right: Leaf = None   # Node or Leaf positive
	left: Leaf = None    # Node or Leaf negative

	def __init__(self, column, pivot, right, left):
		self.column = column
		self.pivot = pivot
		self.right = right
		self.left = left

	def set_leaf_tag(self):
		self.right.set_leaf_tag()
		self.left.set_leaf_tag()

	def evaluate(self, row):
		if row[[self.column]] < self.pivot:
			return self.left.evaluate(row)
		return self.right.evaluate(row)

	def plot(self):
		print('---- Column ' + self.column + ' < ' + str(self.pivot) + ' ----')
		self.left.plot()
		print('\n')
		print('---- Column ' + self.column + ' >= ' + str(self.pivot) + ' ----')
		self.right.plot()

	def select_random_branch(self):
		r = randrange(3)
		father = None
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
			columns = self.root.data.columns
			columns = columns[columns != self.column]
			self.column = columns[randrange(len(columns))]
			self.pivot = self.root.data[self.column].mean()

		self.left.mutate()
		self.right.mutate()
		return

	def repartition(self, partition):
		split_column = self.root.data[self.column]
		criteria = split_column < self.pivot
		self.right.repartition(criteria & self.partition)
		self.left.repartition(criteria & self.partition)
		return


class Leaf:
	root = None 			# Tree's root
	tag = None 				# Decision to take
	partition = None		# Boolean vector that mask the initial data belonging to the leaf

	def __init__(self, root, partition):
		self.root = root
		self.partition = partition

	# Split the data into two new leaves
	def warm(self, levels) -> Node:
		column = self.root.data.columns[randrange(len(self.root.data.columns))]  # random column to split
		(criteria, pivot) = self.select_pivot(column)
		if isinstance(criteria, int):  # not a good split #TODO: Efficiency not good, random column select should be avoid
			if self.root.deepness == levels:    # if first node, take an other column to split
				ret_node = self.warm(levels)
			else:
				ret_node = self
		else:		# build two leaves and return the new node that keeps this leaves
			right = Leaf(self.root, criteria & self.partition)
			left = Leaf(self.root, ~criteria & self.partition)

			if levels > 1:
				right = right.warm(levels-1)
				left = left.warm(levels-1)
			ret_node = Node(column, pivot, right, left)
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
			for x in grill:
				n_left = sum(split_column[self.partition] < x)
				n_right = sum(split_column[self.partition] >= x)

				if n_left < 3 or n_right < 3:  # low data to split #TODO: use a parameter instead of hardcoded number
					l_entropy = 0.5
					r_entropy = 0.5
				else:
					classes = list(dict.fromkeys(self.root.label.iloc[:, 0].unique()))
					r_entropy = 1
					l_entropy = 1

					for clas in classes:
						r_entropy += entropy(sum((split_column[self.partition] >= x) & self.root.label.iloc[:, 0][self.partition] == clas) / n_right)
						l_entropy += entropy(sum((split_column[self.partition] < x) & self.root.label.iloc[:, 0][self.partition] == clas) / n_left)
					r_entropy = n_right * total_inverse * r_entropy
					l_entropy = n_left * total_inverse * l_entropy

				grill_entropy.append(l_entropy + r_entropy)

			min_index = grill_entropy.index(min(grill_entropy))
			pivot = grill[min_index]  # best pivot
			criteria = split_column < pivot  # builds the next mask
			left_count = sum(criteria & self.partition)
			if left_count < 3 or sum(self.partition)-left_count < 3:  # low data to split
				return 0, 0

			return criteria, pivot  # return mask and pivot

	# Select the tag the leaf will have
	def set_leaf_tag(self):
		self.tag = self.root.label.iloc[:, 0].mask(self.partition).value_counts().idxmax()

	# Return the expected class
	def evaluate(self, date):
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

	# TODO
	def mutate(self):
		return

	def repartition(self, partition):
		self.partition = partition
		self.set_leaf_tag()
		return
