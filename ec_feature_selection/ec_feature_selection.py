import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy import sparse
from sklearn.metrics import mutual_info_score

class ECFS():
    """
    Feature Ranking and Selection via Eigenvector Centrality is a graph-based 
    method for feature selection that ranks feature by identifying the most important ones.
    
    References
    --------
    Based on the algorithm as introduced in:
     
    Roffo, G. and Melzi, S., 2017, July. Ranking to learn: Feature ranking and selection via eigenvector centrality.
    In New Frontiers in Mining Complex Patterns: 5th International Workshop,
    NFMCP 2016, Held in Conjunction with ECML-PKDD 2016, Riva del Garda, Italy, September 19, 2016,
    Revised Selected Papers (Vol. 10312, p. 19). Springer.
    
    % @InProceedings{RoffoECML16, 
    % author={G. Roffo and S. Melzi}, 
    % booktitle={Proceedings of New Frontiers in Mining Complex Patterns (NFMCP 2016)}, 
    % title={Features Selection via Eigenvector Centrality}, 
    % year={2016}, 
    % keywords={Feature selection;ranking;high dimensionality;data mining}, 
    % month={Oct}}
    
    This Python implementation is inspired by the 'Feature Ranking and Selection via Eigenvector Centrality'
    MATLAB implementation that can be found in the Feature Selection Library (FSLib) by Giorgio Roffo.
    Many more Feature Selection methods are also available in the Feature Selection Library:
    https://www.mathworks.com/matlabcentral/fileexchange/56937-feature-selection-library
 
     Parameters
    ----------
    n_features : int > 0 and lower than the m original features, or None (default=None)
        Number of features to select.
        If n_features is set to None all features are kept and a ranked dataset is returned.
        
    Attributes
    ----------
    n_features : int
        Number of features to select.
        
    epsilon : int or float >=0 (default 1e-5)
        A small number. Used for avoiding division by zero.
        
    alpha : int or float ∈ [0, 1]
        Loading coefficent.
        The adjacency matrix A is given by: A = (alpha * kernel) + (1 − alpha) * sigma

    positive_class : int, float, or str (default 1)
        Label of the positive class.

    negative : int, float, or str (default -1)
        Label of the negative class.
        
     fisher_score: numpy array, shape (m_features,)
        The fisher scores for each feature.
    
    mutual_information: float 
        Mutual Information of X and y. 
        
    A: numpy array (m_features, m_features)
        The adjacency matrix.
        
    eigenvalues: numpy array (n_features,)
        The eigenvalues of the adjacency matrix.

    eigenvectors: numpy array (n_features, n_features)
        The eigenvectors of the adjacency matrix.
        
    ranking: numpy array (n_features,)
        Ranking of features (0 is the most important feature, 1 is 2nd most imporant etc... ).    
    """
    
    def __init__(self, n_features=None, epsilon=1e-5):

        
        self.n_features = n_features
        self.epsilon = epsilon
        
    def fit(self, X, y, alpha, positive_class, negative_class): 
        """
        Computes the feature ranking from the training data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, m_features)
            Training data to compute the feature importance scores from.

        y : array-like, shape (n_samples)
            Training labels.

        alpha : int or float ∈ [0, 1]
            Loading coefficent.
            The adjacency matrix A is given by: A = (alpha * kernel) + (1 − alpha) * sigma

        positive_class : int, float, or str (default 1)
            Label of the positive class.

        negative : int, float, or str (default -1)
            Label of the negative class.

         Returns
        -------
        self : object
            Returns the instance itself.
        """
        
        X = self.check_array(X)
        y = self.check_array(y)
        assert X.shape[0] == y.shape[0], 'X and y should have the same number of samples. {} != {}'.format(X.shape[0], y.shape[0])
        
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.alpha = alpha
        
        self.fisher_score = self.get_fisher_score(X, y)
        self.mutual_information = np.apply_along_axis(self.get_mutual_information, 0, X, y)
        
        self.kernel = self.build_kernel(self.fisher_score, self.mutual_information)
        self.sigma = self.build_sigma(X)
        
        self.A =  self.alpha * self.kernel + (1-self.alpha) * self.sigma
        
        self.eigenvalues, self.eigenvectors = eigh(self.A)
        
        self.ranking = np.abs(self.eigenvectors[:, self.eigenvalues.argmax()]).argsort()[::-1]
        
        
    def transform(self, X):
        """           
        Reduces the feature set down to the top n_features.
        
        Parameters
        ----------
        X: array-like (n_samples, m_features)
            Data to perform feature ranking and selection on.
            
        Returns
        -------
        X_ranked: array-like (n_samples, n_top_features)
            Reduced feature matrix.
        """
        if self.n_features:
            if self.n_features > X.shape[1]:
                raise ValueError('Number of features to select is higher than the original number of features. {} > {}'.format(self.n_features, X.shape[1]))
        
        X_ranked = X[:, self.ranking]
        
        if self.n_features is not None:
            X_ranked = X_ranked[:, :self.n_features]
        return X_ranked
    
    def fit_transform(self, X, y, alpha, positive_class=1, negative_class=-1):
        """
        Computes the feature ranking from the training data, then reduces
        the feature set down to the top n_features.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, m_features)
            Training data to compute the feature importance scores from
            and to perform feature ranking and selection on.
            
        y : array-like, shape (n_samples)
            Training labels.

        alpha : int or float ∈ [0, 1]
            Loading coefficent.
            The adjacency matrix A is given by: A = (alpha * kernel) + (1 − alpha) * sigma

        positive_class : int, float, or str (default 1)
            Label of the positive class.

        positive_class : int, float, or str (default -1)
            Label of the negative class.
            
        Returns
        -------
        X_ranked: array-like (n_samples, n_top_features)
            Reduced feature matrix.
        """
        
        
        self.fit(X, y, alpha, positive_class, negative_class)
        return self.transform(X)
    
    def get_fisher_score(self, X, y):
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
            
        Returns
        -------
        fisher_score: array-like (m_features,)
            Reduced feature matrix.        
        """
        
        y_labels = set(np.unique(y))
        
        input_labels = set([self.negative_class, self.positive_class])
        
        if (y_labels != input_labels):
            raise ValueError('Positive and negative class labels should match y labels. {} != {}'.format(y_labels, input_labels))
        
        if self.positive_class == self.negative_class:
            raise ValueError('Positive class label and negative class label can not be the same.')
            
        positive_class = X[y==self.positive_class]
        negative_class = X[y==self.negative_class]

        positive_mu = positive_class.mean(axis=0)
        negative_mu = negative_class.mean(axis=0)

        positive_var = positive_class.var(axis=0)
        positive_var[positive_var == 0] = self.epsilon
        
        negative_var = negative_class.var(axis=0)
        negative_var[negative_var == 0] = self.epsilon

        fisher_score = (positive_mu - negative_mu) ** 2 / (positive_var + negative_var) 

        return fisher_score
    
    def get_mutual_information(self, X, y):
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
            
        Returns
        -------
        mutual_information: float 
            Mutual Information of X and y.                

        """

        n = X.shape[0]
        
        # Binning like FSLib
        if(n/10 > 20):
            nbins = 20
        else:
            nbins = max(n//10, 10)

        pX = np.histogram(X, nbins)[0] / n
        pY = np.histogram(y, 2)[0] / n
        
        score = self.mutual_information_score(X, y, nbins)
        
        HX = self.entropy(pX)
        HY = self.entropy(pY)

        mutual_information = (2 * score) / (HX + HY)

        return mutual_information    
    
    def mutual_information_score(self, X, y, nbins):
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
          
    def entropy(self, p):
        """
        Computes the entropy of a distribution for given probability values.
        
        entropy = -Σp * log(p)
        
        Parameters
        ----------
        p : array-like, 
            Probability values.
            
        Returns
        -------
        entropy: float 
            The entropy.
        """

        p[p==0] = self.epsilon
        entropy = -np.sum(p * np.log(p))
        return entropy
    
    def normalize(self, matrix):
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
                                  
    def build_kernel(self, fisher_score, mutual_information):
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
        kernel = self.normalize(kernel)

        return kernel

    def build_sigma(self, X):
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
        
        return sigma

    def check_array(self, arr):
        """
        Utility function for checking and validating input arrays.
        """

        # Check array type    
        if isinstance(arr, (pd.DataFrame, pd.Series)):
            arr = arr.to_numpy()
            
        elif sparse.issparse(arr):
            arr = arr.toarray()

        elif isinstance(arr, np.ndarray):
            pass
        else:
            raise TypeError('Expected one of [numpy.array, pandas.DataFrame, pandas.Series, scipy.sparse], but got {}'.format(type(arr)))
            
        # Check array dimension
        if arr.ndim > 2:
            raise ValueError('Expected 1D or 2D array,but got {}D  instead.'.format(arr.ndim))
        
        # Check if the data is numerical
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError('Expected numeric data, but got {}'.format(arr.dtype))
        
        # Check if the data contains NaN of inf
        if not np.isfinite(arr).any():
            raise TypeError('Data can not contain np.nan or np.inf')

        return arr
    
