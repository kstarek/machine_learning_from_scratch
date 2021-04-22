import numpy as np

X = np.genfromtxt('Data.tsv', delimiter="\t")
# clusters initialized from kmeans output
cents = np.genfromtxt('kmeans_cents.tsv', delimiter="\t")
k = 3
numDatum = X.shape[0]
numFeatures = X.shape[1]


class GMM(object):
    def __init__(self):
        self.assignments = np.zeros(shape=numDatum)
        self.k = k
        # convergence when delta between old and new log likelihoods = 0, Shubham mentioned we should check for convergence in announcements
        self.stoppingTol = 0
        self.log_likelihood_old = 0
        self.cur_log_likelihood = 0
        self.error = float("inf")
        self.pi_vector = [1 / self.k for x in range((self.k))]
        self.sigma = [np.zeros((len(X[0]), len(X[0]))) for i in range(self.k)]

    def fit(self, X):

        self.r_score = np.zeros((numDatum, self.k))
        # clusters initialized from kmeans output
        self.mus = cents
        # 3 4x4 cov matrices
        self.sigma = np.full((self.k, 4, 4), np.cov(X, rowvar=0))
        # set error to 0 for convergence
        while self.error != self.stoppingTol:
            # max expectation, take current log likelihood for convergence testing
            self.eStep(X)
            # max maximization ;)
            self.mStep(X)
            # absolute value for dif between likelihoods
            self.error = abs(self.cur_log_likelihood - self.log_likelihood_old)
            self.log_likelihood_old = self.cur_log_likelihood
        # parse assignments from r scores
        for i in range(numDatum):
            max = np.max(self.r_score[i])
            index_max = np.where(self.r_score[i] == max)
            self.assignments[i] = index_max[0]

        np.savetxt('gmm_output.tsv', X=self.assignments, delimiter="\t")
    def eStep(self, X):
        # loop through k times, set r score of ith iteration to multivariate gaussian * weights
        for i in range(self.k):
            pi = self.pi_vector[i]
            # current implementation of multivariate gaussian returns a nxn matrix, we want the diagonals along that matrix
            fakenewslikelihood = [self.mv_gauss(X, self.mus[i], self.sigma[i])[j] for j in range(numDatum)]
            # take matrix row by row and convert to n arrays
            fakenewslikelihood2 = [np.squeeze(np.asarray(x)) for x in fakenewslikelihood]
            # take diagonal element from said matrix
            fakenewslikelihood3 = [fakenewslikelihood2[h][h] for h in range(len(fakenewslikelihood2))]
            # array of diagonal elements/likelihoods
            self.fakenewslikelihood4 = np.array(fakenewslikelihood3)
            # multiply assignments/likelihoods by weights and assign to r score matrix
            self.r_score[:, i] = pi * self.fakenewslikelihood4
        # log likelihood = sum of log(sum of weights * mv gaussian across all k as in (15.13))
        log = np.sum(np.log(np.sum(self.r_score, axis=1)))
        r_score_sum = self.r_score.sum(axis=1)
        r_score_sum = np.atleast_2d(r_score_sum).transpose()
        # divide by sum of r scores across all k as in (15.12)
        self.r_score = self.r_score / r_score_sum
        self.cur_log_likelihood = log

    def mStep(self, X):
        # sum of r score assigned for each cluster
        sum_r_score = self.r_score.sum(axis=0)
        # update weights using summed r scores
        self.pi_vector = sum_r_score / numDatum

        r_score_X_product = np.dot(self.r_score.transpose(), X)
        # update means, shape correctly
        self.mus = np.atleast_2d(r_score_X_product / sum_r_score[:, np.newaxis])

        # update sigma
        for i in range(self.k):
            # delta between means and X[i] by feature
            delta_x_mu = (X - self.mus[i].transpose()).transpose()
            # dot product of deltas and r scores for x_i (numerator of (15.17))
            pi_r_score_prod = self.r_score[:, i] * delta_x_mu
            pi_r_score_prod = np.dot(pi_r_score_prod, delta_x_mu.transpose())
            # divide by summed r scores as in (15.17)
            self.sigma[i] = pi_r_score_prod / sum_r_score[i]

    def mv_gauss(self, X, mus, sigma):
        # transpose mus to row vector
        mus = np.atleast_2d(mus).transpose()
        # get determinant
        det = np.linalg.det(sigma)
        # constant term
        const_term = 1 / (((2 * np.pi) ** (len(mus) / 2)) * (det ** (1 / 2)))
        # exponent term, split up for clarity
        delta_x_mu = np.matrix(X - mus.transpose())
        inv = np.linalg.inv(sigma)
        exponent_term = (-1 / 2) * (delta_x_mu.dot(inv))
        exponent_term = exponent_term.dot(delta_x_mu.transpose())
        exponent_term = np.exp(exponent_term)
        return const_term * exponent_term


model = GMM()
model.fit(X)
