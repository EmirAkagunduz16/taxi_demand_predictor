from datetime import datetime, timedelta

import hopsworks
from hsfs.feature_store import FeatureStore
import pandas as pd
import numpy as np

import src.config as config


def get_hopsworks_project() -> hopsworks.project.Project:

    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )



def get_feature_store() -> FeatureStore:
    project = get_hopsworks_project()
    return project.get_feature_store()



def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """"""
    # past_rides_columns = [c for c in features.columns if c.startswith('rides_')]
    predictions = model.predict(features)

    results = pd.DataFrame()
    results['pickup_location_id'] = features['pickup_location_id'].values
    results['predicted_demand'] = predictions.round(0)
    
    return results



def load_batch_of_features_from_store(
    current_date: pd.Timestamp,    
) -> pd.DataFrame:
    """Fetches the batch of features used by the ML system at `current_date`

    Args:
        current_date (datetime): datetime of the prediction for which we want
        to get the batch of features

    Returns:
        pd.DataFrame: 4 columns:
            - `pickup_hour`
            - `rides`
            - `pickup_location_id`
            - `pickpu_ts`
    """
    features_store = get_feature_store()

    n_features = config.N_FEATURES

    feature_view = features_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )


    # fetch data from the feature store
    fetch_data_from = current_date - timedelta(days=28)
    fetch_data_to = current_date - timedelta(hours=1)

    
    # add plus minus margin to make sure we do not drop any observation
    ts_data = feature_view.get_batch_data(
        start_time=fetch_data_from - timedelta(days=1),
        end_time=fetch_data_to + timedelta(days=1)
    )

    ts_data['pickup_hour'] = pd.to_datetime(ts_data['pickup_hour'], format="%m/%d/%Y").dt.date

    # filter data to the time period we are interested in
    pickup_ts_from = int(fetch_data_from.timestamp() * 1000)
    pickup_ts_to = int(fetch_data_to.timestamp() * 1000)


    pickup_date_from = datetime.fromtimestamp(pickup_ts_from / 1000).date()
    pickup_date_to = datetime.fromtimestamp(pickup_ts_to / 1000).date()

    ts_data = ts_data[ts_data.pickup_hour.between(pickup_date_from, pickup_date_to)]
    # ts_data = ts_data[(ts_data['pickup_hour'] >= fetch_data_from) & (ts_data['pickup_hour'] <= fetch_data_to)]


    # validate we are not missing data in the feature store
    location_ids = ts_data['pickup_location_id'].unique()
    # assert len(ts_data) == config.N_FEATURES * len(location_ids), \
    #     "Time-series data is not complete. Make sure your feature pipeline is up and runnning."


    # sort data by location and time
    ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)


    x = []

    for location_id in location_ids:
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, :]
        if not ts_data_i.empty:
            ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
            x.append(ts_data_i['rides'].values)
        else:

            x.append(np.zeros(n_features, dtype=np.float32))


    max_length = max(len(arr) for arr in x)

    for i in range(len(x)):
        x[i] = np.pad(x[i], (0, max_length - len(x[i])), mode='constant')

    # numpy arrays to Pandas dataframes
    features = pd.DataFrame(
        x,
        columns=[f'rides_previous_{i+1}_hour' for i in range(max_length)]
    )
    features['pickup_hour'] = current_date
    features['pickup_location_id'] = location_ids
    features.sort_values(by=['pickup_location_id'], inplace=True)

    return features
    


def load_model_from_registry():
    
    import joblib
    from pathlib import Path

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model = model_registry.get_model(
        name=config.MODEL_NAME,
        version=1,
    )  
    
    model_dir = model.download()
    model = joblib.load(Path(model_dir)  / 'model.pkl')
       
    return model