import pandas as pd
import numpy as np
class DataCleaner:
    """
        A configurable data cleaning pipeline for Pandas DataFrames.

        This class provides common data cleaning steps to prepare data for analysis or machine learning algorithms. 
        It standardizes column names, handles missing values, detects outliers & optimizes column dtypes.

        Parameters
        ----------
        verbose : bool, default= True
            If True, logs each cleaning step. If False, runs silently.
        num_strategy : {'mean', 'median'} or scalar, default= 'median'
            Strategy for imputing missing values in numeric columns/
            - 'mean'  : fill with column mean
            - 'median': fill with column median
            - scalar  : fill with a constant value
        high_card_thresh : int, default= 100
            Threshold for detecting high cardinality categorical columns.
        low_card_ratio : float, default= 0.05
            Ratio of unique values to total rows below which object columns are converted to pandas category.
        missing_values : list, optional
            List of string tokens to treat as missing values (e.g., ['na', 'null']).
            Defaults to common placeholders: ['n/a', 'na', '--', '-', 'none', 'null', '', 'nan'].

        Attributes
        ----------
        report : dict
            Stores summary information from the last cleaning run, such as:
                - 'duplicates_removed'
                - 'missing_before' & 'missing_after'
                - 'constant_cols'
                - 'high_card_cols'
                - 'outliers'
                - 'converted_to_category'

        Methods
        -------
        clean(df: pd.DataFrame) -> pd.DataFrame
            Apply the cleaning pipeline to the given DataFrame & return a new cleaned copy.
    """
    def __init__(self, verbose: bool = True, num_strategy: str = 'median', cat_strategy: str = 'mode', 
                 high_card_thresh: int = 100, low_card_ratio: float = 0.05, missing_values: list = None):
        self.verbose = verbose
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.high_card_thresh = high_card_thresh
        self.low_card_ratio = low_card_ratio
        self.missing_values = missing_values or ['n/a', 'na', '--', '-', 'none', 'null', '', 'nan']
        self.report = {}

    def _log(self, msg) -> None:
        '''
        Print a log message if verbose is enabled.

        Parameters
        ----------
        msg : str
            The message to display in the log output.

        Returns
        -------
        None
        '''
        if self.verbose:
            print(f'[INFO] {msg}')


    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Apply the full data cleaning pipeline to a DataFrame.

        This method standardizes column names, remove duplicates, normalizes text columns, handles missing values, 
        detects constant & high cardinality columns, identifies numeric outliers & converts suitable object column to the 'category' dtype.
        
        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to be cleaned. A copy is created so the original DataFrame remains unchanged.

        Returns
        -------
        pd.DataFrame
            A cleaned DataFrame with standardized columns, imputed values, & optimized data types.

        Notes
        -----
        - Duplicates are dropped in-place.
        - Missing values are imputed according to the strategies specified in the constructor('num_strategy', 'cat_strategy')
        - The class attribute 'report' is updated with data about the cleaning process (e.g., duplicates removed, columns converted).
        '''
        df = df.copy()
    
        # Cleaning column names
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r'[^a-z0-9]+', '_', regex= True)
            .str.replace(r'^_|_$', '', regex= True)
        )
        self._log('Standardized column names.')
        
        # Removing duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            df.drop_duplicates(inplace= True)
            self._log(f'Removed {dup_count} duplicate rows.')
            self.report['duplicates_removed'] = dup_count
        
        # Standardizing string columns by trimming any spaces if present at the beginning/end of the string & 
        # converting all the strings to lowercase.
        for col in df.select_dtypes(include= 'object'):
            df[col] = df[col].astype(str).str.strip().str.lower()
        self._log('Standardized string columns (lowercased + trimmed).')

        # Replacing missing values
        df.replace(self.missing_values, np.nan, inplace= True)

        # Imputing missing values
        null_cols = df.isnull().sum()
        null_cols = null_cols[null_cols > 0]
        if not null_cols.empty:
            self._log(f'Missing values found in columns: \n{null_cols}')
            self.report['missing_before'] = null_cols.to_dict()
                    
            for col in null_cols.index:
                if np.issubdtype(df[col].dtype, np.number):
                    if self.num_strategy == 'mean':
                            df[col].fillna(df[col].mean(), inplace= True)
                    elif self.num_strategy == 'median':
                        df[col].fillna(df[col].median(), inplace= True)
                    else:
                        df[col].fillna(self.num_strategy, inplace= True)
                else:
                    if self.cat_strategy == 'mode':
                        df[col].fillna(df[col].mode()[0], inplace= True)
                    else:
                        df[col].fillna(self.cat_strategy, inplace= True)
            self._log(f'Missing values imputed using num_strategy = "{self.num_strategy}" & cat_strategy = "{self.cat_strategy}".')
            self.report['missing_after'] = df.isnull().sum()[df.isnull().sum() > 0].to_dict()

        # Detecting constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            self._log(f'Constant columns found. Consider removing: {constant_cols}')
            self.report['constant_cols'] = constant_cols
            
        # Detecting high cardinality columns
        high_card_cols = [col for col in df.select_dtypes(include= 'object') if df[col].nunique() > self.high_card_thresh]
        if high_card_cols:
            self._log(f'High-cardinality columns (consider encoding strategies): {high_card_cols}')
            self.report['high_card_cols'] = high_card_cols
            
        # Detecting potential numeric outliers
        numeric_cols = df.select_dtypes(include= np.number).columns
        outlier_report = {}
        for col in numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                outlier_report[col] = outliers
        if outlier_report:
            self._log(f'Potential numeric outliers detected:\n{outlier_report}')
            self.report['outliers'] = outlier_report
            
        # Converting low cardinality object columns to category
        converted_cols = []
        for col in df.select_dtypes(include= 'object'):
            if df[col].nunique() < len(df) * self.low_card_ratio:
                df[col] = df[col].astype('category')
                converted_cols.append(col)
        if converted_cols:
            self._log(f'Converted object columns to category: {converted_cols}')
            self.report['converted_to_category'] = converted_cols
    
        self._log('Data cleaning completed.')
        return df