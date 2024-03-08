# Native libraries
from warnings import filterwarnings
from tempfile import mkdtemp
from shutil import rmtree
from os.path import isdir
from os import getcwd
from os import mkdir

# Local source code
from datamining.data_visualization import plot_correlation_matrix
from datamining.data_visualization import plot_feature_importance
from datamining.data_visualization import analyze_investment
from datamining.data_visualization import plot_price_boxplot
from datamining.data_visualization import plot_wordclouds
from datamining.data_visualization import plot_eda_graphs
from datamining.data_visualization import plot_price_kde
from datamining.data_visualization import dataset_stats
from datamining.data_visualization import plot_metrics
from datamining.data_visualization import plot_pca
from datamining.preprocessing import preprocess_instance
from datamining.preprocessing import fill_missing_values
from datamining.preprocessing import generate_encodings
from datamining.preprocessing import discretize_values
from datamining.preprocessing import discard_features
from datamining.preprocessing import remove_outliers
from datamining.preprocessing import optimize_memory
from datamining.preprocessing import equal_frequency_binning
from datamining.ml_methods import initial_regressions
from datamining.ml_methods import tuned_regressions
from datamining.ml_methods import feature_selection
from datamining.ml_methods import predict_instance
from datamining.ml_methods import paired_ttest
from datamining.ml_methods import fine_tuning
from utils.RfecvPipeline import RfecvPipeline

# External libraries
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pandas.api.types import is_integer_dtype
from pandas.api.types import is_float_dtype
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from joblib import Memory
import pandas as pd
import numpy as np
import joblib


def load_dataset(file_name: str, extension: str) -> pd.DataFrame:

    try:
        if extension == 'csv':
            df = pd.read_csv(f'{getcwd()}/../datasets/{file_name}.{extension}')
        elif extension == 'parquet':
            df = pd.read_parquet(f'{getcwd()}/../datasets/{file_name}.{extension}')
        else:
            print('Invalid file extension.')
            exit()
    except (IsADirectoryError, NotADirectoryError, FileExistsError, FileNotFoundError):
        print("Dataset not found or doesn't exists!")
        exit()

    df = df.rename(columns={
        'id': 'ID',
        'nome': 'Name',
        'host_id': 'Host ID',
        'host_name': 'Host Name',
        'bairro_group': 'Borough',
        'bairro': 'District',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'room_type': 'Room Type',
        'price': 'Price',
        'minimo_noites': 'Minimum Nights',
        'numero_de_reviews': 'Reviews',
        'ultima_review': 'Last Review',
        'reviews_por_mes': 'Monthly Reviews',
        'calculado_host_listings_count': 'Number of Listings',
        'disponibilidade_365': "Days Available"
    })
    return df


def eda_step(df: pd.DataFrame) -> None:

    dataset_stats(df=df.copy())
    plot_eda_graphs(df=df.copy())

    plot_price_boxplot(df=df.copy())
    plot_price_kde(df=df.copy())
    plot_wordclouds(df=df.copy())

    plot_pca(df=df.copy())
    analyze_investment(df=df.copy())

    plot_correlation_matrix(df=df.copy().select_dtypes(exclude=['object']), graph_width=8)
    plot_correlation_matrix(df=df.copy()[['Minimum Nights', 'Days Available', 'Price']], graph_width=8, file_name='reduced')


def preprocess_step(df: pd.DataFrame) -> pd.DataFrame:

    # Optimizing memory changing and downcasting types
    obj_cols = list(df.select_dtypes(include=['object']).columns)
    obj_cols.remove('Name')
    df = optimize_memory(df=df, file_name='reduced', object_cols=obj_cols)

    # Generates categorical features encodings in a key-value format (e.g., {'Manhattan': 0, 'Brooklyn': 1, ...})
    generate_encodings(df=df)

    # Discarding irrelevant features, reducing features cardinality and filling missing values
    df = discard_features(df=df.copy())
    '''
    Careful with this step, columns that have many unique values may take long to fit the model and be memory intensive
    Host ID column has 11452 unique labels ==> ~12 GB RAM
    '''
    df = fill_missing_values(df=df.copy())
    df = optimize_memory(df=df, file_name='filled')

    # Removing outliers by price (IQR method) and applying equal frequency binning on columns that have many unique values
    df = remove_outliers(df=df.copy(), col='Price')
    df = equal_frequency_binning(df=df.copy())

    # Binned features still in category type needs to be encoded
    features = df.columns.values
    for col in features:
        if not is_float_dtype(df[col]) and not is_integer_dtype(df[col]):
            df = discretize_values(df=df, column=col)

    # Dataset ready for working with ML algorithms
    final = optimize_memory(df=df, file_name='preprocessed')

    print(final.info())
    print(final.nunique())
    plot_correlation_matrix(df=df.copy().select_dtypes(exclude=['object']), graph_width=8, file_name='complete')

    return final


def tune_models(attributes: pd.DataFrame, classes: pd.DataFrame) -> None:

    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)

    scaler = StandardScaler()

    # LightGBM fine tuning
    lgbm_pipeline = Pipeline(steps=[
        ('scaler', scaler),
        ('regressor', LGBMRegressor(n_jobs=-1, verbose=-1))
    ], memory=memory)

    lgbm_param_grid = [{
        'regressor__n_estimators': [n for n in range(50, 351, 100)],
        'regressor__max_depth': [d for d in range(3, 19, 3)],
        'regressor__learning_rate': [r for r in np.arange(0.01, 0.52, 0.1)]
    }]

    fine_tuning(attributes=attributes, classes=classes,
                regressor=lgbm_pipeline, param_grid=lgbm_param_grid, alg_name='lgbm')

    # XGBoost fine tuning
    xgb_pipeline = Pipeline(steps=[
        ('scaler', scaler),
        ('regressor', XGBRegressor(n_jobs=-1))
    ], memory=memory)

    xgb_param_grid = [{
        'regressor__n_estimators': [n for n in range(50, 351, 100)],
        'regressor__max_depth': [d for d in range(3, 19, 3)],
        'regressor__learning_rate': [r for r in np.arange(0.01, 0.52, 0.1)]
    }]

    fine_tuning(attributes=attributes, classes=classes,
                regressor=xgb_pipeline, param_grid=xgb_param_grid, alg_name='xgb')

    # HistGradientBoosting fine tuning
    hist_pipeline = Pipeline(steps=[
        ('scaler', scaler),
        ('regressor', HistGradientBoostingRegressor())
    ], memory=memory)

    hist_param_grid = [{
        'regressor__max_depth': [d for d in range(3, 19, 3)],
        'regressor__learning_rate': [r for r in np.arange(0.01, 0.52, 0.05)]
    }]

    fine_tuning(attributes=attributes, classes=classes,
                regressor=hist_pipeline, param_grid=hist_param_grid, alg_name='histgb')

    rmtree(cachedir)


def ml_step(attributes: pd.DataFrame, classes: pd.DataFrame) -> None:

    # First scores with vanilla models
    results = initial_regressions(attributes, classes)
    plot_metrics(df=results)

    # Fine tuning and new scores
    tune_models(attributes=attributes, classes=classes)
    tuned_regressions(attributes=attributes, classes=classes)

    # Checking features importances
    scaler = StandardScaler()
    algorithm = RfecvPipeline(steps=[
        ('scaler', scaler),
        ('regressor', LGBMRegressor(learning_rate=0.11, max_depth=9,
                                    n_estimators=150, n_jobs=-1, verbose=-1))
    ])

    algorithm.fit(attributes, classes)
    plot_feature_importance(attributes=attributes.columns,
                            f_importances=algorithm['regressor'].feature_importances_,
                            alg_name='LightGBM')

    # Recursive feature selection with 5-fold CV
    feature_selected = feature_selection(attributes=attributes, classes=classes, cv_estimator=algorithm)
    final_features = feature_selected.drop(['Price'], axis='columns')
    final_classes = feature_selected['Price']

    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)

    tuned_lgbm_pipeline = Pipeline(steps=[
        ('scaler', scaler),
        ('regressor', LGBMRegressor(learning_rate=0.11, max_depth=9,
                                    n_estimators=150, n_jobs=-1, verbose=-1))
    ], memory=memory)

    tuned_xgb_pipeline = Pipeline(steps=[
        ('scaler', scaler),
        ('regressor', XGBRegressor(learning_rate=0.11, max_depth=6,
                                   n_estimators=50, n_jobs=-1))
    ], memory=memory)

    tuned_hist_pipeline = Pipeline(steps=[
        ('scaler', scaler),
        ('regressor', HistGradientBoostingRegressor(learning_rate=0.11, max_depth=6))
    ], memory=memory)

    # Statistical tests
    paired_ttest(tuned_lgbm_pipeline, tuned_xgb_pipeline, final_features, final_classes)
    paired_ttest(tuned_lgbm_pipeline, tuned_hist_pipeline, final_features, final_classes)
    paired_ttest(tuned_xgb_pipeline, tuned_hist_pipeline, final_features, final_classes)

    # Training and saving models
    tuned_lgbm_pipeline.fit(final_features, final_classes)
    joblib.dump(tuned_lgbm_pipeline, f'{getcwd()}/../models/lgbm_model.pkl')

    tuned_xgb_pipeline.fit(final_features, final_classes)
    joblib.dump(tuned_xgb_pipeline, f'{getcwd()}/../models/xgb_model.pkl')

    tuned_hist_pipeline.fit(final_features, final_classes)
    joblib.dump(tuned_hist_pipeline, f'{getcwd()}/../models/histgb_model.pkl')

    rmtree(cachedir)


# Disclaimer: Preprocessing and ML steps are COMPUTATIONALLY EXPENSIVE. Run at your own risk.
def main() -> None:

    # EDA and preprocessing with vanilla dataset
    dataset = load_dataset(file_name='pricing', extension='csv')
    eda_step(df=dataset.copy())
    # polished_dataset = preprocess_step(df=dataset.copy())

    # Applying Machine learning methods with preprocessed dataset
    # polished_dataset = load_dataset(file_name='preprocessed', extension='parquet')
    # attributes = polished_dataset.drop(['Price'], axis='columns')
    # classes = polished_dataset['Price']
    # ml_step(attributes=attributes, classes=classes)

    feature_selected = load_dataset(file_name='feature_selected', extension='parquet')
    final_features = feature_selected.drop(['Price'], axis='columns')
    final_classes = feature_selected['Price']

    scaler = StandardScaler()
    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)

    tuned_lgbm_pipeline = Pipeline(steps=[
        ('scaler', scaler),
        ('regressor', LGBMRegressor(learning_rate=0.11, max_depth=9,
                                    n_estimators=150, n_jobs=-1, verbose=-1))
    ], memory=memory)

    final_scores = cross_validate(
        estimator=tuned_lgbm_pipeline,
        X=final_features,
        y=final_classes,
        cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=42),
        scoring='r2',
        n_jobs=-1
    )

    r2_score = final_scores['test_score'].mean()
    print(f'R² Score: {r2_score}')
    rmtree(cachedir)

    # First row of the original dataset
    apt_data = {
        'id': 2595,
        'nome': 'Skylit Midtown Castle',
        'host_id': 2845,
        'host_name': 'Jennifer',
        'bairro_group': 'Manhattan',
        'bairro': 'Midtown',
        'latitude': 40.75362,
        'longitude': -73.98377,
        'room_type': 'Entire home/apt',
        'price': 225,
        'minimo_noites': 1,
        'numero_de_reviews': 45,
        'ultima_review': '2019-05-21',
        'reviews_por_mes': 0.38,
        'calculado_host_listings_count': 2,
        'disponibilidade_365': 355
    }

    # Predicting an instance from the dataset
    instance = preprocess_instance(instance=pd.DataFrame([apt_data]))
    predict_instance(instance=instance, algorithm='lgbm')


if __name__ == '__main__':

    filterwarnings('ignore')

    if not isdir(f'{getcwd()}/../plots'):
        mkdir(f'{getcwd()}/../plots')
    if not isdir(f'{getcwd()}/../models'):
        mkdir(f'{getcwd()}/../models')

    main()
