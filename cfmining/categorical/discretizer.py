import pandas as pd
from scipy.stats import beta

from typing import Union, List, Optional


class CategoricalDiscretizerMixIn:
    """ MixIn class for Categorical Discretizer implementations.
    """
    _category_index_dict = {}
    
    def _categorical_discretizer(
        self,
        X: pd.DataFrame,
        categorical_columns: List[str],
        target: pd.Series,
        inplace: bool=False,
        *args,
        **kwargs) -> Optional[pd.DataFrame]:
        
        """
        :param X: a Dataframe.
        :param categorical_columns: list of categorical columns' names.
        :param target: a Series that drives the discretization.
        :param inplace: whether is to return a transformed copy of the dataframe or to change it inplace.
        
        :raises ValueError: not expected datatype.
        """

        assert isinstance(X, pd.DataFrame)
        assert isinstance(target, pd.Series)
        assert target.shape[0] == X.shape[0]

        if not inplace:
          X = X.copy()         
        X[categorical_columns] = X[categorical_columns].apply(lambda x: self._apply_discretizer(x, target))
    
        return X if not inplace else None

    def _apply_discretizer(self, X: pd.Series, target: pd.Series) -> pd.Series:

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
        self._category_index_dict[_cat] = _categorical_index.to_dict()
        
        X = _X.replace(_categorical_index)[_cat]
        
        return X
    
    def category_to_index(self, X: pd.DataFrame) -> pd.DataFrame:
        _cat = X.name 
        
        assert _cat in self._category_index_dict
        return X.replace(self._category_index_dict[_cat])
    
    def index_to_category(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Use the class attribute `_category_index_dict` to revert the 
        index to category. 
        """
        _cat = X.name 
        
        assert _cat in self._category_index_dict
        return X.replace(
            to_replace=self._category_index_dict[_cat].values(),
            value=self._category_index_dict[_cat].keys()
        )