import pandas as pd
import numpy as np
from scipy.stats import beta

from typing import Union, List, Optional

__all__ = ['CategoricalDiscretizer']

class CategoricalDiscretizer(object):
    """Discretize categorical features into ordinal indexes.
    
    :param X: a Dataframe.
    :param categorical_columns: list of categorical columns' names.
    """
    _category_index_dict = {}
    
    def __init__(self,
                 df: Union[pd.DataFrame, np.ndarray],
                 categorical_columns: List[str],
                 names: Optional[List[str]]=None
    ) -> None:
        
        assert isinstance(df, (pd.DataFrame, np.ndarray))
        assert len(categorical_columns) > 0
        
        self.categorical_columns = categorical_columns
        self._is_array = isinstance(df, np.ndarray)
        
        if self._is_array:
            assert isinstance(names, list)
            assert len(names) > 0
            self.X = pd.DataFrame(data=df, columns=names)
        
        else:
            self.X = df
            
    def transform(self, *args, **kwargs):
        return self._categorical_discretizer(*args, **kwargs)
    
    def _categorical_discretizer(
        self,
        target: pd.Series,
        inplace: bool=False
        ) -> Optional[pd.DataFrame]:
        
        """
        :param target: a Series that drives the discretization.
        :param inplace: whether is to return a transformed copy of the dataframe or to change it inplace.
        
        :raises ValueError: not expected datatype.
        """

        assert isinstance(target, pd.Series)
        assert target.shape[0] == self.X.shape[0]

        if not inplace:
          self.X = self.X.copy()         
        self.X[self.categorical_columns] = self.X[self.categorical_columns].apply(lambda x: self._apply_discretizer(x, target))

        self.X = self.X.values if self._is_array else self.X
      
        return self.X if not inplace else None

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