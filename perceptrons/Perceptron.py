import numpy as np


class Perceptron:
    def __init__(self, data, learning_rate=0.01, epochs=100):
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

    def predict(self):
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
                if error:
                    misclassified += 1
                    weights = weights + learning_rate * (labels - prediction) * features
                    bias = bias + learning_rate * (labels - prediction)
            if misclassified == 0:
                self.weights = weights
                self.bias = bias
                return weights, bias
        self.weights = weights
        self.bias = bias
        return weights, bias

    def evaluate(self):
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
