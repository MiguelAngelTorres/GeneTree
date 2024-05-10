from src.genetree import Genetree
import pandas as pd
import time


def small_test():
    data = pd.read_csv("data/Iris.csv")

    start = time.time()
    genetree = Genetree(data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm', 'Species']], 'Species',
                        score_function="auc", num_trees=20, deepness=4, num_rounds=10)
    genetree.train()
    end = time.time()
    print("\nMean time elapsed on warm: " + str(end-start))


def medium_test():
    data = pd.read_csv("data/ammount_earning.csv")

    start = time.time()
    genetree = Genetree(data[['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week', 'ammount']], 'ammount',
                        score_function="accuracy", num_trees=20, deepness=5, num_rounds=10)
    genetree.train()
    end = time.time()
    print("\nMean time elapsed on warm: " + str(end-start))


if __name__ == '__main__':
    small_test()
    #medium_test()



