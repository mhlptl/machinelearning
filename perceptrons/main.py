import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Perceptron import Perceptron


def main():
    df = pd.read_csv('data/dataset-iris_modified.csv')
    df['species'] = np.where(df.iloc[:, -1] == 'Iris-setosa', 0, 1)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
    perceptron = Perceptron(data=data, epochs=10)
    weights, bias = perceptron.predict()
    perceptron.evaluate()

    # print(weights)
    # print(bias)


if __name__ == '__main__':
    main()
