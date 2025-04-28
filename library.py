from __future__ import annotations  #must be first line in your library!
import pandas as pd
import numpy as np
import types
from typing import Dict, Any, Optional, Union, List, Set, Hashable, Literal, Tuple, Self, Iterable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import sklearn
import warnings
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices

class CustomOHETransformer(BaseEstimator, TransformerMixin):
    """
    Applies one-hot encoding to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface. It takes the name
    of a categorical column as input and transforms it into multiple binary columns,
    one for each unique category in the original column. This is useful for preparing
    categorical data for machine learning algorithms that require numerical input.

    Parameters
    ----------
    target_column : str
        The name of the column to be one-hot encoded.

    Attributes
    ----------
    target_column : str
        The name of the column to be transformed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'color': ['red', 'green', 'blue', 'red']})
    >>> ohe_transformer = CustomOHETransformer(target_column='color')
    >>> transformed_df = ohe_transformer.fit_transform(df)
    >>> transformed_df
       color_blue  color_green  color_red
    0           0            0           1
    1           0            1           0
    2           1            0           0
    3           0            0           1
    """
    def __init__(self, target_column: str) -> None:
        """
        Initialize the CustomOHETransformer with the target column name.

        Parameters
        ----------
        target_column : str
            The name of the column to be one-hot encoded.

        Raises
        ------
        AssertionError
            If target_column is not a string.
        """
        assert isinstance(target_column, str), f'{self.__class__.__name__} constructor expected string but got {type(target_column)} instead.'
        self.target_column: str = target_column
        return

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation as one-hot encoding doesn't require fitting.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomOHETransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self  #always the return value of fit

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot encoding to the target column in the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to be transformed.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with the target column one-hot encoded.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame or if target_column is not in X.
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        #warnings.filterwarnings('ignore', message='.*downcasting.*')  #squash warning in replace method below
        assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'

        X_: pd.DataFrame = X.copy()
        X_ = pd.get_dummies(X,
                               prefix=f'{self.target_column}',    #your choice
                               prefix_sep='_',     #your choice
                               columns=[self.target_column],
                               dummy_na=False,    #will try to impute later so leave NaNs in place
                               drop_first=False,   #really should be True but could screw us up later
                               dtype=int
                               )
        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience. In this case, fit does
        nothing, so this method simply calls transform().

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with the target column one-hot encoded.
        """
        result: pd.DataFrame = self.transform(X)
        return result

class CustomDropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that either drops or keeps specified columns in a DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It allows for selectively keeping or dropping columns
    from a DataFrame based on a provided list.

    Parameters
    ----------
    column_list : List[str]
        List of column names to either drop or keep, depending on the action parameter.
    action : str, default='drop'
        The action to perform on the specified columns. Must be one of:
        - 'drop': Remove the specified columns from the DataFrame
        - 'keep': Keep only the specified columns in the DataFrame

    Attributes
    ----------
    column_list : List[str]
        The list of column names to operate on.
    action : str
        The action to perform ('drop' or 'keep').

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>>
    >>> # Drop columns example
    >>> dropper = CustomDropColumnsTransformer(column_list=['A', 'B'], action='drop')
    >>> dropped_df = dropper.fit_transform(df)
    >>> dropped_df.columns.tolist()
    ['C']
    >>>
    >>> # Keep columns example
    >>> keeper = CustomDropColumnsTransformer(column_list=['A', 'C'], action='keep')
    >>> kept_df = keeper.fit_transform(df)
    >>> kept_df.columns.tolist()
    ['A', 'C']
    """

    def __init__(self, column_list: List[str], action: Literal['drop', 'keep'] = 'drop') -> None:
        """
        Initialize the CustomDropColumnsTransformer.

        Parameters
        ----------
        column_list : List[str]
            List of column names to either drop or keep.
        action : str, default='drop'
            The action to perform on the specified columns.
            Must be either 'drop' or 'keep'.

        Raises
        ------
        AssertionError
            If action is not 'drop' or 'keep', or if column_list is not a list.
        """
        assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
        assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
        self.column_list: List[str] = column_list
        self.action: Literal['drop', 'keep'] = action

    #your code below
    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation as dropping or keeping columns
        doesn't require fitting.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomDropColumnsTransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self  #always the return value of fit

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drop or keep specified columns in the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with columns either dropped or kept
            based on the specified action and column list. If columns to be dropped
            are not present, it ignores them and prints a warning.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame or if any specified column name to be kept
            is not in the DataFrame.
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        #warnings.filterwarnings('ignore', message='.*downcasting.*')  #squash warning in replace method below
        missing_keys = set(self.column_list) - set(X)

        X_: pd.DataFrame = X.copy()
        if self.action == 'drop':
            if missing_keys:
                print(f'Warning: DropColumnsTransformer does not contain these columns to drop: {missing_keys}.')
            X_ = X_.drop(columns=self.column_list, errors='ignore')
        elif self.action == 'keep':
            assert not missing_keys, f"{self.__class__.__name__} unknown columns to keep: {missing_keys}"
            X_ = X_[self.column_list]
        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience. In this case, fit does
        nothing, so this method simply calls transform().

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to transform.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with columns either dropped or kept
            based on the specified action and column list. If columns to be dropped
            are not present, it ignores them and prints a warning.
        """
        result: pd.DataFrame = self.transform(X)
        return result

class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that maps values in a specified column according to a provided dictionary.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It applies value substitution to a specified column using
    a mapping dictionary, which can be useful for encoding categorical variables or
    transforming numeric values.

    Parameters
    ----------
    mapping_column : str or int
        The name (str) or position (int) of the column to which the mapping will be applied.
    mapping_dict : dict
        A dictionary defining the mapping from existing values to new values.
        Keys should be values present in the mapping_column, and values should
        be their desired replacements.

    Attributes
    ----------
    mapping_dict : dict
        The dictionary used for mapping values.
    mapping_column : str or int
        The column (by name or position) that will be transformed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'category': ['A', 'B', 'C', 'A']})
    >>> mapper = CustomMappingTransformer('category', {'A': 1, 'B': 2, 'C': 3})
    >>> transformed_df = mapper.fit_transform(df)
    >>> transformed_df
       category
    0        1
    1        2
    2        3
    3        1
    """

    def __init__(self, mapping_column: Union[str, int], mapping_dict: Dict[Hashable, Any]) -> None:
        """
        Initialize the CustomMappingTransformer.

        Parameters
        ----------
        mapping_column : str or int
            The name (str) or position (int) of the column to apply the mapping to.
        mapping_dict : Dict[Hashable, Any]
            A dictionary defining the mapping from existing values to new values.

        Raises
        ------
        AssertionError
            If mapping_dict is not a dictionary.
        """
        assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
        self.mapping_dict: Dict[Hashable, Any] = mapping_dict
        self.mapping_column: Union[str, int] = mapping_column  #column to focus on

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : CustomMappingTransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self  #always the return value of fit

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the mapping to the specified column in the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame or if mapping_column is not in X.

        Notes
        -----
        This method provides warnings if:
        1. Keys in mapping_dict are not found in the column values
        2. Values in the column don't have corresponding keys in mapping_dict
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
        #warnings.filterwarnings('ignore', message='.*downcasting.*')  #squash warning in replace method below

        #now check to see if all keys are contained in column
        column_set: Set[Any] = set(X[self.mapping_column].unique())
        keys_not_found: Set[Any] = set(self.mapping_dict.keys()) - column_set
        if keys_not_found:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

        #now check to see if some keys are absent
        keys_absent: Set[Any] = column_set - set(self.mapping_dict.keys())
        if keys_absent:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

        X_: pd.DataFrame = X.copy()
        X_[self.mapping_column] = X_[self.mapping_column].replace(self.mapping_dict)
        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.
        """
        #self.fit(X,y)  #commented out to avoid warning message in fit
        result: pd.DataFrame = self.transform(X)
        return result
    
class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies Tukey's fences (inner or outer) to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in a scikit-learn pipeline.
    It clips values in the target column based on Tukey's inner or outer fences.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply Tukey's fences on.
    fence : Literal['inner', 'outer'], default='outer'
        Determines whether to use the inner fence (1.5 * IQR) or the outer fence (3.0 * IQR).

    Attributes
    ----------
    inner_low : Optional[float]
        The lower bound for clipping using the inner fence (Q1 - 1.5 * IQR).
    outer_low : Optional[float]
        The lower bound for clipping using the outer fence (Q1 - 3.0 * IQR).
    inner_high : Optional[float]
        The upper bound for clipping using the inner fence (Q3 + 1.5 * IQR).
    outer_high : Optional[float]
        The upper bound for clipping using the outer fence (Q3 + 3.0 * IQR).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'values': [10, 15, 14, 20, 100, 5, 7]})
    >>> tukey_transformer = CustomTukeyTransformer(target_column='values', fence='inner')
    >>> transformed_df = tukey_transformer.fit_transform(df)
    >>> transformed_df
    """
    def __init__(self, target_column: str, fence : Literal['inner', 'outer'], default='outer') -> None:
      self.target_column: str = target_column 
      self.fence = fence
      self.inner_low: Optional[float] = None
      self.outer_low: Optional[float] = None
      self.inner_high: Optional[float] = None
      self.outer_high: Optional[float] = None
      return 
    
    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
      assert self.target_column in X.columns, f'unknown column {self.target_column}'
      q1 = X[self.target_column].quantile(0.25)
      q3 = X[self.target_column].quantile(0.75)
      iqr = q3-q1
      self.inner_low = q1-1.5*iqr
      self.outer_low = q1-3.0*iqr
      self.inner_high = q3+1.5*iqr
      self.outer_high = q3+3.0*iqr
      return X
      
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
      assert self.inner_low is not None and self.inner_high is not None, "TukeyTransformer.fit has not been called."
      
      X_ = X.copy()
      if self.fence == 'inner':  
        X_[self.target_column] = X_[self.target_column].clip(lower=self.inner_low, upper=self.inner_high)
      elif self.fence == 'outer':
        X_[self.target_column] = X_[self.target_column].clip(lower=self.outer_low, upper=self.outer_high)
      #X_[self.target_column] = X_[self.target_column].clip(lower=self.low_wall, upper=self.high_wall)
      X_.reset_index(drop=True)
      return X_


      

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
      result: pd.DataFrame = self.fit(X)
      result: pd.DataFrame = self.transform(X)
      return result


class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies 3-sigma clipping to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It clips values in the target column to be within three standard
    deviations from the mean.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply 3-sigma clipping on.

    Attributes
    ----------
    high_wall : Optional[float]
        The upper bound for clipping, computed as mean + 3 * standard deviation.
    low_wall : Optional[float]
        The lower bound for clipping, computed as mean - 3 * standard deviation.
    """
    def __init__(self, target_column: str) -> None:
      self.target_column: str = target_column 
      self.high_wall: Optional[float] = None
      self.low_wall: Optional[float] = None
      return 
    
    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
      assert self.target_column in X.columns, f'unknown column {self.target_column}'
      #q1 = X[self.target_column].quantile(0.25)
      #q3 = X[self.target_column].quantile(0.75)
      #iqr = q3-q1
      #self.low_wall = q1-3*iqr
      #self.high_wall = q3+3*iqr
      mean = X[self.target_column].mean()
      std = X[self.target_column].std()

  # Compute the low and high boundaries
      self.low_wall = mean - 3 * std
      self.high_wall = mean + 3 * std
      return X
      
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
      assert self.low_wall is not None and self.high_wall is not None, "Sigma3Transformer.transform called before fit."
      X_ = X.copy()
      X_[self.target_column] = X_[self.target_column].clip(lower=self.low_wall, upper=self.high_wall)
      X_.reset_index(drop=True)
      return X_


      return 

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
      result: pd.DataFrame = self.fit(X)
      result: pd.DataFrame = self.transform(X)
      return result
    
class CustomRobustTransformer(BaseEstimator, TransformerMixin):
  """Applies robust scaling to a specified column in a pandas DataFrame.
    This transformer calculates the interquartile range (IQR) and median
    during the `fit` method and then uses these values to scale the
    target column in the `transform` method.

    Parameters
    ----------
    column : str
        The name of the column to be scaled.

    Attributes
    ----------
    target_column : str
        The name of the column to be scaled.
    iqr : float
        The interquartile range of the target column.
    med : float
        The median of the target column.
        value_new = (value – median) / iqr #iqr = q3-q1
  """
  def __init__(self, column):
      self.target_column = column
      self.iqr = None 
      self.med = None
      return 

  def fit(self, X: pd.DataFrame) -> Self:
      assert self.target_column in X.columns, f'unknown column {self.target_column}'
      new_col = X[self.target_column].tolist()
      new_col = [x for x in new_col if not math.isnan(x)]
      self.med = statistics.median(new_col)
      q1 = np.percentile(new_col, 25)
      q3 = np.percentile(new_col, 75)
      self.iqr = q3 - q1

  # Compute the low and high boundaries
      return X

  def transform(self, X: pd.DataFrame) -> pd.DataFrame:
      assert self.med is not None and self.iqr is not None, f'Transform called before Fit'
      X_ = X.copy()
      if self.iqr != 0:
        X_[self.target_column] = (X_[self.target_column] - self.med) / self.iqr
      X_.reset_index(drop=True)
      return X_


  def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
      result: pd.DataFrame = self.fit(X)
      result: pd.DataFrame = self.transform(X)
      return result
  

    
customer_transformer = Pipeline(steps=[
    #fill in the steps on your own
    ('drop', CustomDropColumnsTransformer(['ID'], 'drop')),
    ('experience', CustomMappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high': 2})),
    ('gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('os', CustomOHETransformer('OS')),
    ('isp', CustomOHETransformer('ISP')),
    ('time spent', CustomTukeyTransformer('Time Spent', 'inner')),
    ], verbose=True)

titanic_transformer = Pipeline(steps=[
    ('gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('joined', CustomOHETransformer('Joined')),
    ('fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    #add your new ohe step below


    ], verbose=True)


