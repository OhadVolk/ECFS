# Feature Ranking and Selection via Eigenvector Centrality

ECFS is a Python implementation of the [Feature Ranking and Selection via Eigenvector Centrality](https://arxiv.org/abs/1704.05409) algorithm by Giorgio Roffo and Simone Melz.
Please check [the code](https://github.com/OhadVolk/ECFS/blob/master/ec_feature_selection/__init__.py) for full references.


## Installation

Windows users can run the following:

```bash
git clone https://github.com/OhadVolk/ECFS.git
cd ECFS
python setup.py install
```
Linux users can run the following:

```bash
git clone https://github.com/OhadVolk/ECFS.git
cd ECFS
sudo python setup.py install
```


## Usage
Check the [Example.ipynb](https://github.com/OhadVolk/ECFS/blob/master/Example.ipynb) notebook for more details.

```python
from ec_feature_selection import ECFS

# Create an instance, select top 10 features
ecfs = ECFS(n_features=10)
# Fit and Transform the training data
X_train_reduced = ecfs.fit_transform(X=X_train, y=y_train, alpha=0.5, positive_class=1, negative_class=0)
# Transform the test data
X_test_reduced = ecfs.transform(X_test)
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
