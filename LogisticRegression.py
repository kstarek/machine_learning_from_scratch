import numpy as np

data = np.loadtxt('train_dataset.tsv', delimiter="\t")

X = data[:, :-1]
y = data[:, -1]
bias = np.zeros((len(X), 1))
X = np.append(bias, X, axis=1)


class LogisticRegression(object):

    def __init__(self):
        self = self

    def logistic_func(self, z):
        denominator = (1 + np.e ** (-z))
        return 1 / denominator

    #def fit(self, X, y, max_iters=5000, learning_rate=0.01, stop_tol=0.0005): #default suggested hyperparameters, gave me 83.5% accuracy on validation
    def fit(self, X, y, max_iters=40000, learning_rate=0.002, stop_tol=0.00001):  #these hyperparameters gave me the best accuracy, 85.32% on validation set
        weights = np.zeros(X.shape[1])
        for i in range(max_iters):
            # y_hat

            y_pred = self.logistic_func(np.dot(X, weights))

            # update as shown in class and store old weights for stopping condition
            old_weights = weights.copy()
            weights = weights - (learning_rate * np.dot(X.transpose(), y_pred - y) / len(X))

            euclid_norm = np.linalg.norm(weights - old_weights, ord=2)
            if euclid_norm < stop_tol:
                break

        self.weights = weights

    def predict(self, X):
        # w.T*X
        dot_product = np.dot(X, self.weights)
        logistic_dot_prod = self.logistic_func(dot_product)
        # as shown in class we use the logistic function of w.T*X to specify the decision boundary
        prediction_row_vector = [1 if logistic > 0.5 else 0 for logistic in logistic_dot_prod]
        return prediction_row_vector


model = LogisticRegression()
model.fit(X, y)
np.savetxt("weights.tsv", model.weights, delimiter="\t")

