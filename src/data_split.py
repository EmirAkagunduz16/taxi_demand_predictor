from datetime import datetime
from typing import Tuple

import pandas as pd


def train_test_split(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_column_name: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    """
    # Convert pickup_hour to timestamp with timezone information
    df['pickup_hour'] = pd.to_datetime(df['pickup_hour'], format='%Y-%m-%d %H:%M:%S', utc=True)

    # Convert cutoff_date to timestamp with timezone information
    cutoff_date = pd.to_datetime(cutoff_date, utc=True)

    # Split data into training and testing sets
    train_data = df[df.pickup_hour < cutoff_date].reset_index(drop=True)
    test_data = df[df.pickup_hour >= cutoff_date].reset_index(drop=True)

    # Split features and target
    X_train = train_data.drop(columns=[target_column_name])
    y_train = train_data[target_column_name]
    X_test = test_data.drop(columns=[target_column_name])
    y_test = test_data[target_column_name]

    return X_train, y_train, X_test, y_test