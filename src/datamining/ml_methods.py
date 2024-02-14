import numpy as np
from os import getcwd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def holdout_split(attributes, classes):

    # Training and test sets obtained by stratified and random splitting (80% and 20% - holdout)
    training_attributes, test_attributes, training_classes, test_classes = train_test_split(attributes,
                                                                                            classes,
                                                                                            test_size=0.2,
                                                                                            shuffle=True,
                                                                                            random_state=42)
    return training_attributes, test_attributes, training_classes, test_classes


def regressions(attributes, classes):

    # imputer = SimpleImputer(strategy='median', missing_values=np.nan)
    scaler = StandardScaler()

    lr_pipeline = Pipeline(steps=[
        # ('imputer', imputer),
        ('scaler', scaler),
        ('regressor', LinearRegression())
    ])

    knn_pipeline = Pipeline(steps=[
        # ('imputer', imputer),
        ('scaler', scaler),
        ('regressor', KNeighborsRegressor())
    ])

    xgb_pipeline = Pipeline(steps=[
        # ('imputer', imputer),
        ('scaler', scaler),
        ('regressor', XGBRegressor())
    ])

    lgbm_pipeline = Pipeline(steps=[
        # ('imputer', imputer),
        ('scaler', scaler),
        ('regressor', LGBMRegressor())
    ])

    pipelines = {'Linear Regression': lr_pipeline, 'KNN': knn_pipeline, 'XGBoost': xgb_pipeline, 'LightGBM': lgbm_pipeline}

    # 10-fold cross-validation
    for key in pipelines:
        scores = cross_val_score(
            estimator=pipelines[key],
            X=attributes,
            y=classes,
            cv=10,
            scoring='r2'
        )
        print(key)
        print(f'R² Mean: {scores.mean()}')
        print(f'R² Std: {scores.std()}\n')


def fine_tuning(attributes, classes, algorithm, param_grid):

    grid_search = GridSearchCV(estimator=algorithm,
                               param_grid=param_grid,
                               scoring="r2",
                               n_jobs=-1,
                               cv=5,
                               verbose=0,
                               return_train_score=True)
    grid_search.fit(attributes, classes)

    print(grid_search.best_params_)
    with open(f'{getcwd()}/../models/{algorithm}_best_params.txt', 'w') as file:
        for key in grid_search.best_params_:
            file.write(f'{key}: {grid_search.best_params_[key]}\n')
