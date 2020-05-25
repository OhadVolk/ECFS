import numpy as np
from scipy.linalg import eigh
from ec_feature_selection.utils import check_array
from ec_feature_selection._ecfs_functions import get_fisher_score, get_mutual_information, build_kernel, build_sigma

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
    
    epsilon : int or float >=0 (default 1e-5)
        A small number. Used for avoiding division by zero.     
        
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

        y : array-like, shape (n_samples,)
            Training labels.

        alpha : int or float ∈ [0, 1]
            Loading coefficent.
            The adjacency matrix A is given by: A = (alpha * kernel) + (1 − alpha) * sigma

        positive_class : int, float, or str 
            Label of the positive class.

        negative : int, float, or str 
            Label of the negative class.

         Returns
        -------
        self : object
            Returns the instance itself.
        """
        
        X = check_array(X)
        y = check_array(y)
        assert X.shape[0] == y.shape[0], 'X and y should have the same number of samples. {} != {}'.format(X.shape[0], y.shape[0])
        
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.alpha = alpha
        
        self.fisher_score = get_fisher_score(X=X, y=y, negative_class=self.negative_class,
                                             positive_class=self.positive_class, epsilon=self.epsilon)
        
        self.mutual_information = np.apply_along_axis(get_mutual_information, 0, X, y, self.epsilon)
        
        self.kernel = build_kernel(self.fisher_score, self.mutual_information)
        self.sigma = build_sigma(X)
        
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
