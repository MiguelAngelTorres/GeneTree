from src.genetree import Genetree
import pandas as pd
import time
import polars as pl
import numpy as np

def small_test():
    data = pd.read_csv("data/Iris.csv")

    start = time.time()
    genetree = Genetree(data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']], data[['Species']],
                        score_function="auc", num_trees=20, deepness=4, num_rounds=10)
    genetree.train()
    end = time.time()
    print("\nMean time elapsed on warm: " + str(end-start))


def medium_test():
    data = pd.read_csv("data/ammount_earning.csv")

    start = time.time()
    genetree = Genetree(data[['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']], data[['ammount']],
                        score_function="accuracy", num_trees=10, deepness=4, num_rounds=10)
    genetree.train()
    end = time.time()
    print("\nMean time elapsed on warm: " + str(end-start))


def min_reproducible_error():
    def fun_partition(partition, column, pivot):
        if len(column) == 0:
            print(partition.filter(pl.col('b')).group_by("target_label").agg(pl.col('b')).collect())
            print(partition.filter(pl.col('b')).group_by("target_label").agg(pl.col('b').len()))
            print(partition.filter(pl.col('b')).group_by("target_label").agg(pl.col('b').len()).collect())
        else:
            fun_partition(partition.with_columns(a=(pl.col(column[0]) < pivot[0])).with_columns(b=(pl.col("a") & pl.col("b"))),
                               column[1:], pivot[1:])

    data = pd.DataFrame(data={'var1': [1, 2, 3], 'var2': [1, 2, 3], 'tar': ['<50k', '<50k', '<50k']})
    label_np = data[['tar']].squeeze().to_numpy()
    data = pl.from_pandas(data).with_columns(target_label=pl.Series(label_np)).lazy()
    fun_partition(data.with_columns(b=True), ['var1', 'var2'], [5, 3])

if __name__ == '__main__':
    # This work
    #small_test()

    # This work
    #min_reproducible_error()

    # This doesnt work
    medium_test()


