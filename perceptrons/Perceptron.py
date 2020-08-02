from typing import Tuple
import numpy as np

# Perceptron Implementation
class Perceptron:
    def __init__(self, data: list, learning_rate: float = 0.01, epochs: int = 100):
        """
            initializes the Perceptron model with the values the user inputted
        """
        self.X_train = np.array(data['X_train'])
        self.X_test = np.array(data['X_test'])
        nTrain, numFeatures = np.shape(self.X_train)
        nTest, _ = np.shape(self.X_test)
        self.y_train = np.reshape(np.array(data['y_train']), (nTrain, 1))
        self.y_test = np.reshape(np.array(data['y_test']), (nTest, 1))
        self.weights = np.zeros((1, numFeatures))
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bias = 0

    def predict(self) -> Tuple[list, float]:
        """
            using the training set, the model trains itself until it does 
            not misclassify any of the training set, or until it reaches the 
            number of epochs which were stated by the user
        """
        X = self.X_train
        y = self.y_train
        weights = self.weights
        learning_rate = self.learning_rate
        epochs = self.epochs
        bias = self.bias

        for _ in range(epochs):
            misclassified = 0
            for features, labels in zip(X, y):
                prediction = np.dot(features, np.transpose(weights)) + bias
                prediction = 0 if prediction < 0 else 1
                error = labels - prediction
                # if a misclassifiction occured, change the weights and the bias
                if error:
                    misclassified += 1
                    weights = weights + learning_rate * (labels - prediction) * features
                    bias = bias + learning_rate * (labels - prediction)
            # if no misclassification occured after an iteration, return the weights and bias to the user
            if misclassified == 0:
                self.weights = weights
                self.bias = bias
                return weights, bias
        self.weights = weights
        self.bias = bias
        return weights, bias

    def evaluate(self) -> None:
        """
            this evaluates the model on the test set to check if the model is accurate
            and prints the accuracy
        """
        X = self.X_test
        y = self.y_test
        weights = self.weights
        bias = self.bias
        n, _ = np.shape(X)
        classified = 0
        for features, labels in zip(X, y):
            prediction = np.dot(features, np.transpose(weights)) + bias
            prediction = 0 if prediction < 0 else 1
            error = labels - prediction
            if not error:
                classified += 1
        accuracy = classified / n * 100
        print('Accuracy: {accuracy: .2f}'.format(accuracy=accuracy))
