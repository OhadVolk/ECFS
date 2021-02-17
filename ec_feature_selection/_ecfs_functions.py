import numpy as np
from sklearn.metrics import mutual_info_score
from typing import Union


def get_fisher_score(X: np.ndarray, y: np.ndarray, negative_class: Union[int, float, str],
                     positive_class: Union[int, float, str], epsilon: Union[int, float] = 1e-5) -> np.ndarray:
    """
    Computes the Fisher score for the features.

    The Fisher score is the the ratio of interclass separation and intraclass variance,
    where features are evaluated independently.

    For every feature i the following is computed:

    fi = |µ(i,1) − µ(i,2)| ^ 2 / σ(i,1) ^ 2 + σ(i,2) ^ 2

    where µ(i,C) and σ(i,C) are the mean and standard deviation respectively, assumed by the
    i-th feature when considering the samples of the C-th class. 

    The higher fi - the more discriminative the i-th feature is.

    Parameters
    ----------
    X : array-like, shape (n_samples, m_features)
        Feature set to compute the Fisher score on

    y : array-like, shape (n_samples)
        Training labels.
        
    positive_class : int, float, or str
        Label of the positive class.

    negative_class : int, float, or str
        Label of the negative class.        
        
    epsilon : float >=0 (default 1e-5)
        A small number. Used for avoiding division by zero.        

    Returns
    -------
    fisher_score: array-like (m_features,)
        Reduced feature matrix.        
    """

    y_labels = set(np.unique(y))

    input_labels = set([negative_class, positive_class])

    if (y_labels != input_labels):
        raise ValueError(
            'Positive and negative class labels should match y labels. {} != {}'.format(y_labels, input_labels))

    if positive_class == negative_class:
        raise ValueError('Positive class label and negative class label can not be the same.')

    positive_class = X[y == positive_class]
    negative_class = X[y == negative_class]

    positive_mu = positive_class.mean(axis=0)
    negative_mu = negative_class.mean(axis=0)

    positive_var = positive_class.var(axis=0)
    positive_var[positive_var == 0] = epsilon

    negative_var = negative_class.var(axis=0)
    negative_var[negative_var == 0] = epsilon

    fisher_score = (positive_mu - negative_mu) ** 2 / (positive_var + negative_var)

    return fisher_score


def get_mutual_information(X: np.ndarray, y: np.ndarray, epsilon: float = 1e-5):
    """
    The Mutual Information is a measure of the similarity between two labels of
    the same data.

    For every feature i the following Mutual Information score is computed:
    mi_score =Σy∈YΣz∈x(i) p(z, y)log * log(p(z, y) / p(z) * p(y))

    Following that, and s computed in the original implemenation:

    mi = (2 * mi_score) / (H(X) + H(Y))

    Where H(X) and H(Y) are the entropy of X and Y.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
        Feature to compute Mutual Information

    y : array-like, shape (n_samples,)
        Training labels.
        
    epsilon : float >=0 (default 1e-5)
        A small number. Used for avoiding division by zero.            

    Returns
    -------
    mutual_information: float 
        Mutual Information of X and y.                

    """

    n = X.shape[0]

    # Binning like FSLib
    if (n / 10 > 20):
        nbins = 20
    else:
        nbins = max(n // 10, 10)

    pX = np.histogram(X, nbins)[0] / n
    pY = np.histogram(y, 2)[0] / n

    score = mutual_information_score(X, y, nbins)

    HX = entropy(pX, epsilon)
    HY = entropy(pY, epsilon)

    mutual_information = (2 * score) / (HX + HY)

    return mutual_information


def mutual_information_score(X: np.ndarray, y: np.ndarray, nbins: int) -> float:
    """
    Computes the Mutual Information score with the contingency table pf X and y.

    Parameters
    ----------
    X : array-like, shape (n_samples,)
        Feature to compute Mutual Information

    y : array-like, shape (n_samples,)
        Training labels.  

    nbins : int >= 0 
        Number of bins to use when creating the contingency table.

    Returns
    -------
    score: float 
        Mutual Information score as computed with the contingency table of X and y.                                    
    """

    contingency = np.histogram2d(x=X, y=y, bins=nbins)[0]
    score = mutual_info_score(None, None, contingency=contingency)

    return score


def entropy(p: np.ndarray, epsilon: float = 1e-5) -> float:
    """
    Computes the entropy of a distribution for given probability values.

    entropy = -Σp * log(p)

    Parameters
    ----------
    p : array-like, 
        Probability values.
        
    epsilon : float >=0 (default 1e-5)
        A small number. Used for avoiding division by zero.        
        
    Returns
    -------
    entropy: float 
        The entropy.
    """
    p[p == 0] = epsilon
    entropy = -np.sum(p * np.log(p))

    return entropy


def normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize a matrix in the the range 0 to 1.

    Parameters
    ----------
    matrix : array-like, 
        The matrix to noramlize

    Returns
    -------
    normalized_matrix: array-like, 
        Normalized matrix in the range 0 to 1.
    """

    matrix = matrix - np.min(matrix)
    normalized_matrix = matrix / np.max(matrix)

    return normalized_matrix


def build_kernel(fisher_score: np.ndarray, mutual_information: np.ndarray) -> np.ndarray:
    """
    Building the kernel.

   Parameters
    ----------
    fisher_score : array-like, shape(n_features,)
        The Fisher scores for the features.

    mutual_information : array-like, shape(n_features,)
        The Mutual Information scores for the features.

    Returns
    -------
    kernel: array-like, shape(n_features, n_features)
        The kernel.
    """

    kernel = np.abs(fisher_score + mutual_information) / 2
    kernel = np.dot(kernel.reshape(-1, 1), kernel.reshape(1, -1))
    kernel = normalize(kernel)

    return kernel


def build_sigma(X: np.ndarray) -> np.ndarray:
    """
    Building the matrix Sigma (Σ).

    Feature-evaluation metric based on standard deviation – capturing the 
    amount of variation or dispersion of features from average.

   Parameters
    ----------
    X : array-like, shape(n_samples, m_features)
        Feature set to build the sigma matrix from.

    Returns
    -------
    sigma: array-like, shape(m_features, m_features)
        The sigma matrix.        
    """
    std = X.std(axis=0)
    n = len(std)

    sigma = np.repeat(std, n).reshape(n, n).T

    for i, row in enumerate(sigma):
        row[row < std[i]] = std[i]

    sigma = normalize(sigma)

    return sigma
