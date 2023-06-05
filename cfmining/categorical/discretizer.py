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
    _index_to_category_dict = {}
    
    def __init__(self,
                 categorical_columns: Union[str, List[str]],
                 names: Optional[List[str]]=None
    ) -> None:
        
        assert len(categorical_columns) > 0
        
        self.categorical_columns = categorical_columns
        self.names = names
            
    def transform(self, *args, **kwargs):
        
        return self.category_to_index(*args, **kwargs)
    
    def inverse_transform(self, *args, **kwargs):
        
        return self.index_to_category(*args, **kwargs)
    
    def fit(self, 
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            **kwargs) -> None:
        
        assert isinstance(X, (pd.DataFrame, np.ndarray)) 
        self._is_array = isinstance(X, np.ndarray)
        
        if self._is_array:
            assert isinstance(self.names, list)
            assert len(self.names) > 0
            self.X = pd.DataFrame(data=X, columns=self.names)
        
        else:
            self.X = X
            
        self._categorical_discretizer(y, **kwargs)
    
    def fit_transform(self,
                      X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.Series, np.ndarray],
                      **kwargs) -> Union[pd.DataFrame, np.ndarray]:
            
        self.fit(X, y, **kwargs)
        
        return self.X
    
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
        self._index_to_category_dict[_cat] = {k:v for v, k in self._category_index_dict[_cat].items()}
        
        X = _X.replace(_categorical_index)[_cat]
        
        return X
    
    def category_to_index(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        
        return X.replace(self._category_index_dict, **kwargs)
    
    def index_to_category(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """ Use the class attribute `_category_index_dict` to revert the 
        index to category. 
        """
        
        return X.replace(self._index_to_category_dict, **kwargs)