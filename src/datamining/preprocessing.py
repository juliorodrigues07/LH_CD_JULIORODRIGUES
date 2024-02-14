from os import getcwd
import pandas as pd
from datamining.ml_methods import holdout_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV


def discretize_values(df: pd.DataFrame, column: str) -> pd.DataFrame:

    encoding = LabelEncoder()

    encoding.fit(df[column])
    df[column] = encoding.transform(df[column])

    return df


def normalization(df: pd.DataFrame) -> pd.DataFrame:

    attr_scaler = StandardScaler()
    return attr_scaler.fit_transform(df)


def basic_feature_selection(df: pd.DataFrame) -> pd.DataFrame:

    features = df.columns.values
    for feat in features:
        proportion = round(len(df[feat].unique()) / len(df[feat]), 2)

        # Discard a column if it has more than 90% of unique values
        if proportion * 100 > 90:
            df = df.drop([feat], axis='columns')

    return df


def fit_model(df, algorithm, training, y_training, rows_to_predict, raw_rows, column) -> pd.DataFrame:

    normalized = normalization(df=training)

    algorithm.fit(normalized, y_training[column])
    predictions = algorithm.predict(normalization(df=rows_to_predict))
    raw_rows[column] = predictions

    # Reappends the recent predicted rows to the original dataset without changing order
    return pd.concat([df, raw_rows], sort=False).sort_index()


def regression_fill(df: pd.DataFrame, column: str) -> pd.DataFrame:

    # For real values uses Linear Regression to predict missing values
    fill_data = df[df[column].isnull()]
    rows_to_predict = fill_data.drop([column], axis='columns')
    raw_rows = rows_to_predict.copy()

    # Discards rows where the selected column has missing value
    df = df.dropna(axis='index', subset=column)
    features = df.drop([column], axis='columns')
    y_training = discretize_values(df, column)

    # Encodes categorical data
    training = features.copy()
    for col in features.columns.values:
        training = discretize_values(df=training, column=col)
        rows_to_predict = discretize_values(df=rows_to_predict, column=col)

    lr_model = LinearRegression()
    concat_df = fit_model(df=df, algorithm=lr_model, training=training,
                          y_training=y_training, rows_to_predict=rows_to_predict,
                          raw_rows=raw_rows, column=column)

    return concat_df


def classification_fill(df: pd.DataFrame, column: str) -> pd.DataFrame:

    #For integer or categorical values uses Decistion Tree classifier to predict missing values
    fill_data = df[df[column].isnull()]
    rows_to_predict = fill_data.drop([column], axis='columns')
    raw_rows = rows_to_predict.copy()

    df = df.dropna(axis='index', subset=column)
    features = df.drop([column], axis='columns')
    y_training = discretize_values(df, column)

    training = features.copy()
    for col in features.columns.values:
        training = discretize_values(df=training, column=col)
        rows_to_predict = discretize_values(df=rows_to_predict, column=col)

    dt_model = DecisionTreeClassifier()
    concat_df = fit_model(df=df, algorithm=dt_model, training=training,
                          y_training=y_training, rows_to_predict=rows_to_predict,
                          raw_rows=raw_rows, column=column)

    return concat_df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    '''
    Fills missing values by predicting based on existing data
    It only applies this method if a column has at maximum 30% of its values missing
    '''
    features = df.columns.values
    for feat in features:
        if df[feat].isnull().sum() > 0 and df[feat].isnull().sum() / len(df[feat]) < 0.3:

            if df[feat].dtype == float:
                df = regression_fill(df=df, column=feat)
            else:
                df = classification_fill(df=df, column=feat)

    return df


def remove_outliers(df, col, th1=0.25, th3=0.75):

    q1 = df[col].quantile(th1)
    q3 = df[col].quantile(th3)

    iqr = q3 - q1
    upper_limit = q3 + 1.5 * iqr
    lower_limit = q1 - 1.5 * iqr

    df = df[(df[col] > lower_limit) & (df[col] < upper_limit)]
    return df


def equal_frequency_binning(df: pd.DataFrame) -> pd.DataFrame:

    # Gets the bins to apply preprocessing on the example later
    bins = dict()
    n_bins = pd.qcut(df['Host Name'], q=15, retbins=True)
    bins['Host Name'] = n_bins[1]
    n_bins = pd.qcut(df['Host ID'], q=80, retbins=True)
    bins['Host ID'] = n_bins[1]
    n_bins = pd.qcut(df['Latitude'], q=25, retbins=True)
    bins['Latitude'] = n_bins[1]
    n_bins = pd.qcut(df['Longitude'], q=25, retbins=True)
    bins['Longitude'] = n_bins[1]

    # Remove outlier rows based on price (first and third quantiles)
    df = remove_outliers(df, 'Price')

    # Equal frequency binning on scattered features
    categorical_size = [x for x in range(15)]
    df['Host Name'] = pd.qcut(df['Host Name'], q=15, labels=categorical_size)

    categorical_size = [x for x in range(80)]
    df['Host ID'] = pd.qcut(df['Host ID'], q=80, labels=categorical_size)

    categorical_size = [x for x in range(25)]
    df['Latitude'] = pd.qcut(df['Latitude'], q=25, labels=categorical_size)

    categorical_size = [x for x in range(25)]
    df['Longitude'] = pd.qcut(df['Longitude'], q=25, labels=categorical_size)

    # Discretize categorical features
    for col in df.columns.values:
        if df[col].dtypes != int and df[col].dtypes != float:
            df = discretize_values(df=df, column=col)

    # df.to_csv(f'{getcwd()}/../datasets/preprocessed.csv', index=False)
    return df


def feature_selection(attributes, classes, cv_estimator) -> None:

    # Recursive feature selection with CV
    x_train, _, y_train, _ = holdout_split(attributes=attributes, classes=classes)
    cv_estimator.fit(x_train, y_train)

    cv_selector = RFECV(cv_estimator, cv=5, step=1, scoring='r2', verbose=False, n_jobs=-1)
    cv_selector = cv_selector.fit(x_train, y_train)
    rfecv_mask = cv_selector.get_support()

    rfecv_features = list()
    for check, feature in zip(rfecv_mask, x_train.columns):
        if check:
            rfecv_features.append(feature)

    print(f'Original number of features: {len(attributes.columns.values)}')
    print(f'Features: {list(attributes.columns.values)}\n')

    print(f'Optimal number of features: {cv_selector.n_features_}')
    print(f'Best features: {rfecv_features}')

    with open(f'{getcwd()}/../models/best_features.txt', 'w') as file:
        file.write(f'Optimal number of features: {cv_selector.n_features_}\n')
        file.write(f'Best features: {rfecv_features}')

    save_df = attributes.copy()
    save_df.drop(rfecv_features, axis='columns')
    save_df['Price'] = classes
    save_df.to_csv(f'{getcwd()}/../datasets/feature_selected.csv', index=False)
