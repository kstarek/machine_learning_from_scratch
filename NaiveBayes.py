import numpy as np

data = np.loadtxt('train_dataset.tsv', delimiter="\t")
X = data[:, :-1]
y = data[:, -1]

class BernoulliNaiveBayes(object):
    def __init__(self):
        self = self

    x_y0 = X[y == 0]
    x_y1 = X[y == 1]

    def fit(self, X, y):
        num_feats = X.shape[0] # the number of training datum
        grouped_by_class = [[feat for feat, cl in zip(X, y) if cl == cla] for cla in np.unique(y)]

        self.log_prior_y = [np.log(len(i) / num_feats) for i in grouped_by_class] # group by class to calc f
        sum_feat_prob = np.array([np.array(i).sum(axis=0) for i in grouped_by_class])

        num_of_feats_in_class = np.array([len(self.x_y0), len(self.x_y1)])
        divisor = np.array([num_of_feats_in_class]).transpose()

        self.feat_prob = sum_feat_prob / divisor
        return self

    y1_prior = sum(y[y == 1]) / len(y)
    class_priors = np.array([y1_prior, 1 - y1_prior]).transpose()
    np.savetxt('class_priors.tsv', X=class_priors, delimiter="\t")

    def predict(self, X):
        log_feat_prob_1 = np.log(self.feat_prob)
        log_feat_prob_0 = np.log(1 - self.feat_prob)
        log_likelihood_feature_vector = [
            (log_feat_prob_1 * x + log_feat_prob_0 * np.abs(x - 1)).sum(axis=1) + self.log_prior_y for x in X]
        predictions = np.argmax(log_likelihood_feature_vector, axis=1)
        return predictions

    def save_params(self):
        paramDict = {}
        paramDict['class_log_prior'] = self.log_prior_y
        paramDict['feature_probability'] = self.feat_prob
        np.save('params.npy', paramDict)


model = BernoulliNaiveBayes()
model.fit(X, y)
neg_feat_likelihood = model.feat_prob[0]
pos_feat_likelihood = model.feat_prob[1]
np.savetxt('negative_feature_likelihoods.tsv', X=neg_feat_likelihood, delimiter="\t")
np.savetxt('positive_feature_likelihoods.tsv', X=pos_feat_likelihood, delimiter="\t")
model.save_params()
