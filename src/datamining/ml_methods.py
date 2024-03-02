from tempfile import mkdtemp
from shutil import rmtree
from os import getcwd
import json

from datamining.data_visualization import plot_selection_graph
from datamining.data_visualization import plot_learning_curve

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from mlxtend.evaluate import paired_ttest_5x2cv
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from joblib import Memory
import pandas as pd
import numpy as np
import joblib


def initial_regressions(attributes: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:

    # Caching transformed data to avoid repeated computation
    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)

    scaler = StandardScaler()

    lr_pipeline = Pipeline(steps=[
        ('scaler', scaler),
        ('regressor', LinearRegression())
    ], memory=memory)

    knn_pipeline = Pipeline(steps=[
        ('scaler', scaler),
        ('regressor', KNeighborsRegressor(n_jobs=-1))
    ], memory=memory)

    xgb_pipeline = Pipeline(steps=[
        ('scaler', scaler),
        ('regressor', XGBRegressor(n_jobs=-1))
    ], memory=memory)

    lgbm_pipeline = Pipeline(steps=[
        ('scaler', scaler),
        ('regressor', LGBMRegressor(n_jobs=-1, verbose=-1))
    ], memory=memory)

    hist_pipeline = Pipeline(steps=[
        ('scaler', scaler),
        ('regressor', HistGradientBoostingRegressor())
    ], memory=memory)

    pipelines = {
        'Linear Regression': lr_pipeline,
        'KNN': knn_pipeline,
        'XGBoost': xgb_pipeline,
        'LightGBM': lgbm_pipeline,
        'HistGB': hist_pipeline
    }

    mean_results, std_results = pd.DataFrame(), pd.DataFrame()
    mean_results['Metric'] = std_results['Metric'] = ['R²', 'MAE', 'RMSE']

    # 5-fold cross-validation
    for key in pipelines:
        scores = cross_validate(
            estimator=pipelines[key],
            X=attributes,
            y=classes,
            cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=42),
            scoring=('r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'),
            n_jobs=-1
        )
        mean_results[key] = [scores['test_r2'].mean(),
                             abs(scores['test_neg_mean_absolute_error'].mean()),
                             abs(scores['test_neg_root_mean_squared_error'].mean())]
        std_results[key] = [scores['test_r2'].std(),
                            abs(scores['test_neg_mean_absolute_error'].std()),
                            abs(scores['test_neg_root_mean_squared_error'].std())]

    print('Average R²')
    print(mean_results)
    print('\nStandard Deviation R²')
    print(std_results)

    # Clear the cache directory for transformers
    rmtree(cachedir)
    return mean_results


# COMPUTATIONALLY EXPENSIVE
def fine_tuning(attributes: pd.DataFrame, classes: pd.DataFrame,
                regressor: any, param_grid: list, alg_name: str) -> None:

    grid_search = GridSearchCV(estimator=regressor,
                               param_grid=param_grid,
                               scoring='r2',
                               n_jobs=-1,
                               cv=5,
                               verbose=0)
    grid_search.fit(attributes, classes)

    best_params = dict()
    best_params[alg_name] = grid_search.best_params_

    with open(f'{getcwd()}/../models/{alg_name}_best_params.json', 'w') as file:
        dump_file = json.dumps(best_params, indent=4)
        file.write(dump_file)

    print(f'{alg_name} best hyperparameters: {grid_search.best_params_}\n')
    return grid_search.best_estimator_


def tuned_regressions(attributes: pd.DataFrame, classes: pd.DataFrame) -> None:

    cachedir = mkdtemp()
    memory = Memory(location=cachedir, verbose=0)
    scaler = StandardScaler()

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

    tuned_pipelines = {
        'XGBoost': tuned_xgb_pipeline,
        'LightGBM': tuned_lgbm_pipeline,
        'HistGB': tuned_hist_pipeline
    }

    tuned_mean = pd.DataFrame()
    tuned_mean['Metric'] = ['R²', 'MAE', 'RMSE']

    # 5-fold cross-validation
    for key in tuned_pipelines:
        scores = cross_validate(
            estimator=tuned_pipelines[key],
            X=attributes,
            y=classes,
            cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=42),
            scoring=('r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'),
            n_jobs=-1
        )
        tuned_mean[key] = [scores['test_r2'].mean(),
                           abs(scores['test_neg_mean_absolute_error'].mean()),
                           abs(scores['test_neg_root_mean_squared_error'].mean())]

    print(tuned_mean)

    reg_estimators = {
        'LightGBM': tuned_lgbm_pipeline,
        'XGBoost': tuned_xgb_pipeline,
        'HistGradientBoosting': tuned_hist_pipeline
    }
    plot_learning_curve(attributes=attributes, classes=classes,
                        estimators=reg_estimators, n_trainings=20)


def feature_selection(attributes: pd.DataFrame, classes: pd.DataFrame, cv_estimator: any) -> pd.DataFrame:

    # Recursive feature selection with repeated CV
    cv_selector = RFECV(estimator=cv_estimator,
                        cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=42),
                        step=1, scoring='r2',
                        verbose=False, n_jobs=-1)
    cv_selector = cv_selector.fit(attributes, classes)

    rfecv_mask = cv_selector.get_support()
    match_features = np.array(attributes.columns)
    selected_features = match_features[rfecv_mask]

    # Visualization
    plot_selection_graph(rfecv=cv_selector)
    print(f'\nOriginal number of features: {len(attributes.columns.values)}')
    print(f'Features: {list(attributes.columns.values)}\n')

    print(f'Optimal number of features: {cv_selector.n_features_}')
    print(f'Best features: {selected_features}')

    selection = dict()
    selection['Optimal Number'] = str(cv_selector.n_features_)
    selection['Best Features'] = list(selected_features)
    with open(f'{getcwd()}/../models/best_features.json', 'w') as file:
        dump_file = json.dumps(selection, indent=4)
        file.write(dump_file)

    # Exports the feature selected dataset
    save_df = attributes.copy()
    save_df.drop(selected_features, axis='columns')
    save_df['Price'] = classes
    save_df.to_parquet(f'{getcwd()}/../datasets/feature_selected.parquet', index=False)

    return save_df


def paired_ttest(model1, model2, attributes, classes):
    '''
    Significance level
    5% risk of concluding that a difference exists when there is no actual difference
    '''
    alpha = 0.05
    _, p = paired_ttest_5x2cv(model1, model2, attributes, classes)

    print(f'alpha:       {alpha}')
    print(f'p value:     {p}')

    if p > alpha:
        print('Models are statistically equal (Fail to reject null hypothesis)')
    else:
        print('Models are statistically different (Reject null hypothesis)')


def predict_instance(instance: pd.DataFrame, algorithm: str):

    try:
        model = joblib.load(f'{getcwd()}/../models/{algorithm}_model.pkl')
    except (IsADirectoryError, NotADirectoryError, FileExistsError, FileNotFoundError):
        print("Model not found or doesn't exists!")
        exit()

    prediction = model.predict(instance)
    price = round(prediction[0], 2)
    print(f'\nPrice: US$ {str(price)}')
