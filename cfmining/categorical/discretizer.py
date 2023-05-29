import pandas as pd
from scipy.stats import beta

from typing import Union, List

def _categorical_discretizer(
    X: pd.DataFrame,
    categorical_columns: List[str],
    target_column: str='y',
    *args,
    **kwargs) -> pd.DataFrame:
    
    """
    :param X: either a Series or ndarray continaing only categorical information.
    
    
    :raises ValueError: not expected datatype.
    """
    assert isinstance(X, pd.DataFrame)
    assert isinstance(target_column, str)
    assert target_column in X.columns

    y = X[target_column]
    X[categorical_columns] = X[categorical_columns].apply(lambda x: _apply_discretizer(x, y))
  
    return X

def _apply_discretizer(X: pd.Series, target: pd.Series) -> pd.Series:

    _X = pd.concat([X, target], axis=1)
    _cat = X.name
    _y = target.name

    _categories_mean = _X.groupby(_cat)[_y].mean()
    _categories_count = _X.groupby(_cat)[_y].count()
    
    alpha0, beta0, _, _ = beta.fit(_categories_mean)
    
    _categories_corrected_mean = (
        (alpha0 + _categories_mean * _categories_count) /
        (alpha0 + beta0 + _categories_count)
    )

    _categorical_index = _categories_corrected_mean.sort_values().rank().astype(int)

    X = _X.replace(_categorical_index)[_cat]
    
    return X