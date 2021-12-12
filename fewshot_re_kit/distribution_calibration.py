# Contains functions for implementing a modified version of distribution calibration https://openreview.net/pdf?id=JWOiYxMG92s
# See the related notebook for further information
# Also exposes the important variables for sampling: all_mean, all_covariance, base_mean, base_covariance
# Rewriting this with torch on cuda would improve the performance
import numpy as np

rng = np.random.default_rng()
def distribution_cal(sample, base_mean, base_cov, k=2, a=0.2):
    """
    A function that estimates the mean and covariance matrix for a sample (assumed to come from a novel class), given the covariance matrices of the base classes
    k: the number of base classes to use for estimating the novel class's statistics
    a: a constant added to the covariance matrix 
    """
    # Calculate the most similar classes to the sample using class means
    distances = np.sum(np.power(sample - base_mean, 2), axis=1)
    top_k = np.argsort(distances)[:k]
    
    # Use the top_k classes to estimate statistics
    novel_mean = (np.sum(base_mean[top_k, :], axis=0) + sample)/(k+1)
    novel_cov = np.mean(base_cov[top_k, :], axis=0) + a
    
    return novel_mean.reshape(1, -1), novel_cov

def sample_vector(mean, cov, n_samples):
    """
    A function to sample from a distribution given covariance matrix and mean
    Uses the cholesky square root, see here for an explanation: https://stats.stackexchange.com/questions/120179/generating-data-with-a-given-sample-covariance-matrix
    """
    #samples = np.random.multivariate_normal(mean.reshape(-1), cov, size=n_samples) using this decreases performance (training time) for some reason
    chol = np.linalg.cholesky(cov)
    normal_sample = rng.normal(size=(n_samples, mean.shape[1]))
    samples = np.dot(chol, normal_sample.T).T + mean
    return samples

class SurrogateGenerator():
    """A generator that uses distribution calibration to generate new surrogate novel classes"""
    def __init__(self, n_classes=1, k_examples=5, k=2, a=0.2):
        "Load the encodings and calculate the base class statistics needed for generation"
        # Store hyperparameters
        self.n_classes = n_classes
        self.k_examples = k_examples
        self.k = k # the number of base classes to use in estimating statistics
        self.a = a
        
        # Entire distribution statistics
        encodings = np.load("results/train_encodings_tinybert.npy")
        batch, n, k, h = encodings.shape
        encodings = encodings.reshape(batch, n, h)
        all_points = encodings.reshape(-1, h)
        self.all_mean = np.mean(all_points, axis=0).reshape(1, h) # transform to a row vector
        self.all_covariance = np.cov(all_points.T)

        # Statistics for each of the base classes seprately
        self.base_mean = np.mean(encodings, axis=0)

        # Problem: once again we only have 550 observations of 1536 dimensions, so the covariance matrix doesn't have full rank
        # I guess we can add gaussian noise to create more samples? Probably shouldn't have much of an affect on the approximation issue
        extra_samples = []
        for i in range(3):
            noise = rng.normal(loc=0, scale=0.01, size=encodings.shape)
            extra_samples.append(encodings + noise)
        base_upsampled = np.concatenate([encodings] + extra_samples, axis=0)

        base_cov = []
        for i in range(n):
            class_samples = base_upsampled[:, i, :].T
            cov = np.cov(class_samples)
            base_cov.append(cov)
        self.base_covariance = np.stack(base_cov, axis=0)
        
    def __next__(self):
        "Returns an array containing K examples of N classes generated using distribution calibration"
        # First sample the class means from the distribution of all samples
        class_means = sample_vector(self.all_mean, self.all_covariance, self.n_classes)

        # Then sample more examples of each surrogate class, using distribution calibration to get statistics
        surrogate_classes = []
        for class_i in range(class_means.shape[0]):
            mean, cov = distribution_cal(class_means[class_i, :], self.base_mean, self.base_covariance, k=self.k, a=self.a)
            samples = sample_vector(mean, cov, self.k_examples)
            surrogate_classes.append(samples)
        surrogate_classes = np.stack(surrogate_classes, axis=0) # Of shape N, K, hidden_size
        return surrogate_classes
