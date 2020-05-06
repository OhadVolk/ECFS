import numpy as np
import pandas as pd
from scipy import sparse

def check_array(arr):
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