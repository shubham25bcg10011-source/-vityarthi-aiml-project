from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataProcessor:
    def __init__(self, filepath: str, target_column: Optional[str] = None):
        self.filepath = filepath
        self.target_column = target_column
        self.df: Optional[pd.DataFrame] = None
        self.feature_columns: Optional[List[str]] = None
        self.preprocessor: Optional[ColumnTransformer] = None

    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.filepath)
        return self.df

    def _detect_target(self) -> str:
        if self.target_column:
            return self.target_column
        if self.df is None:
            raise ValueError("Dataframe not loaded yet.")
        return self.df.columns[-1]

    def prepare_preprocessor(self) -> ColumnTransformer:
        if self.df is None:
            raise ValueError("Dataframe not loaded yet.")

        # detect numeric and categorical columns
        num_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        cat_cols = list(self.df.select_dtypes(include=["object", "category"]).columns)

        # remove target if present
        target = self._detect_target()
        if target in num_cols:
            num_cols.remove(target)
        if target in cat_cols:
            cat_cols.remove(target)

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])

        transformers = []
        if num_cols:
            transformers.append(("num", numeric_transformer, num_cols))
        if cat_cols:
            transformers.append(("cat", categorical_transformer, cat_cols))

        self.preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        self.feature_columns = num_cols + cat_cols
        return self.preprocessor

    def fit_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.df is None:
            raise ValueError("Dataframe not loaded yet.")

        target = self._detect_target()
        X = self.df.drop(columns=[target])
        y = self.df[target].copy()

        # encode labels if needed outside this class (or use LabelEncoder here)
        if self.preprocessor is None:
            self.prepare_preprocessor()

        X_transformed = self.preprocessor.fit_transform(X)
        return X_transformed, y.values
