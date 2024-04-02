from src.genetree import Genetree
import pandas as pd
import time

def small_test():
    data = pd.read_csv("data/Iris.csv")

    start = time.time()
    genetree = Genetree(data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']], data[['Species']],
                        score_function="auc", num_trees=10, deepness=4, num_rounds=10)
    end = time.time()
    print("\nMean time elapsed on warm: " + str(end-start))


def medium_test():
    data = pd.read_csv("data/ammount_earning.csv")

    start = time.time()
    genetree = Genetree(data[['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']], data[['ammount']],
                        score_function="accuracy", num_trees=10, deepness=4, num_rounds=10)

    end = time.time()
    print("\nMean time elapsed on warm: " + str(end-start))


if __name__ == '__main__':
    small_test()
    #medium_test()