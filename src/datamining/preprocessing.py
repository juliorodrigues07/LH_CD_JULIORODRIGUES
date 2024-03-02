from os import getcwd
import json

from utils.CustomEncoder import CustomEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_integer_dtype
from pandas.api.types import is_float_dtype
import pandas as pd
import numpy as np


def optimize_memory(df: pd.DataFrame, file_name: str = 'optimized',
                    object_cols: list = [], in_place: bool = False) -> pd.DataFrame | None:

    if not in_place:
        df = df.copy()

    memory = df.memory_usage(deep=True)
    print(f'Before optimization: {round(memory.sum() / (1024 ** 2), 2)} MB')

    # Downcast int columns (e.g., int64 -> int32)
    int_cols = df.select_dtypes(include=['int']).columns
    for col in int_cols:
        df[col] = df[[col]].apply(pd.to_numeric, downcast='integer')

    # Downcast float columns (e.g., float64 -> float32)
    float_cols = df.select_dtypes(include=['float']).columns
    for col in float_cols:
        df[col] = df[[col]].apply(pd.to_numeric, downcast='float')

    # Change object type columns to category type to save space
    for col in object_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    memory = df.memory_usage(deep=True)
    print(f'After optimization:  {round(memory.sum() / (1024 ** 2), 2)} MB\n')
    df.to_parquet(f'{getcwd()}/../datasets/{file_name}.parquet')

    if in_place:
        return None
    else:
        return df


def discard_features(df: pd.DataFrame) -> pd.DataFrame:

    features = df.columns.values

    for col in features:
        proportion = round(len(df[col].unique()) / len(df[col]), 2)

        # Discard a column if it has more than 90% of unique values
        if proportion * 100 > 90:
            df = df.drop([col], axis='columns')

    return df


def discretize_values(df: pd.DataFrame, column: str) -> pd.DataFrame:

    encoding = LabelEncoder()

    encoding.fit(df[column])
    df[column] = encoding.transform(df[column])

    return df


def normalization(df: pd.DataFrame) -> pd.DataFrame:

    attr_scaler = StandardScaler()
    return attr_scaler.fit_transform(df)


def predict_missing(df: pd.DataFrame, algorithm: any, attributes: pd.DataFrame,
                    classes: pd.DataFrame, rows_to_predict: pd.DataFrame,
                    raw_rows: pd.DataFrame, column: str) -> pd.DataFrame:

    normalized = normalization(df=attributes)

    algorithm.fit(normalized, classes[column])
    predictions = algorithm.predict(normalization(df=rows_to_predict))
    raw_rows[column] = predictions

    # Uniformly encodes the missing value column if it's categorical before returning
    if not is_float_dtype(df[column]) and not is_integer_dtype(df[column]):
        df = discretize_values(df=df.copy(), column=column)

    # Reappends the recent predicted rows to the original dataset without changing order
    return pd.concat([df, raw_rows], sort=False).sort_index()


def split_missing(df: pd.DataFrame, column: str, ml_type: str) -> pd.DataFrame:

    whole_dataset = df.copy()

    # Gets the rows where the selected column has missing values (predict)
    fill_data = df[df[column].isnull()]
    rows_to_predict = fill_data.drop([column], axis='columns')
    raw_rows = rows_to_predict.copy()

    # Discards rows where the selected column has missing values (training)
    df = df.dropna(axis='index', subset=column)
    attributes = df.drop([column], axis='columns')
    classes = discretize_values(df=df.copy(), column=column)

    '''
    Encodes categorical data (only columns with missing values need it,
    since the others have been previously encoded)
    '''
    for col in attributes.columns.values:
        if whole_dataset[col].isnull().sum() > 0:
            attributes = discretize_values(df=attributes, column=col)
            rows_to_predict = discretize_values(df=rows_to_predict, column=col)

    # Linear Regression for real values and Decision Tree for integers
    model = None
    if ml_type == 'classifier':
        model = DecisionTreeClassifier()
    elif ml_type == 'regressor':
        model = LinearRegression()
    else:
        print('Model incorrectly specified! [classifier, regressor]')
        return whole_dataset

    concat_df = predict_missing(df=df, algorithm=model, attributes=attributes,
                                classes=classes, rows_to_predict=rows_to_predict,
                                raw_rows=raw_rows, column=column)

    return concat_df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:

    # Last Review column cardinality will be reduced by filtering the date only by year and month
    df['Last Review'] = pd.to_datetime(df['Last Review'])
    df['Last Review'] = df['Last Review'].apply(lambda row: f'{str(row.year)}-{str(row.month)}')

    # Null values have to be "properly" inserted again, and datetime type converted to category
    df['Last Review'] = df['Last Review'].astype('category')
    df = df.replace({'Last Review': {'nan-nan': np.nan}})

    # Rows with 0 reviews have missing values in Monthly Reviews and Last Review, which is expected
    missing = len(df.query('`Last Review`.isna() & `Monthly Reviews`.isna() & Reviews == 0'))
    print(f'{missing} rows with no reviews, no last review date and no monthly reviews rate.')

    # Therefore, we can reliably fill these missing values with N/A to last review date and 0 monthly reviews rate
    df['Last Review'] = df['Last Review'].cat.add_categories('N/A')
    df['Last Review'] = df['Last Review'].fillna('N/A')
    df['Monthly Reviews'] = df['Monthly Reviews'].fillna(0)

    # Discretizes columns which have categorical values and no missing ones
    features = df.columns.values
    for col in features:
        if df[col].isnull().sum() == 0 \
                and not is_float_dtype(df[col]) \
                and not is_integer_dtype(df[col]):
            df = discretize_values(df=df, column=col)

    '''
    Fills missing values by predicting based on existing data
    It only applies this method if a column has at maximum 30% of its values missing
    '''
    features = df.columns.values
    for col in features:
        if df[col].isnull().sum() > 0 and df[col].isnull().sum() / len(df[col]) < 0.3:

            if is_float_dtype(df[col]):
                df = split_missing(df=df, column=col, ml_type='regressor')
            else:
                df = split_missing(df=df, column=col, ml_type='classifier')

    print(df.isnull().sum())
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
    bins['Host Name'] = list(n_bins[1])
    n_bins = pd.qcut(df['Host ID'], q=80, retbins=True)
    bins['Host ID'] = list(n_bins[1])
    n_bins = pd.qcut(df['Latitude'], q=25, retbins=True)
    bins['Latitude'] = list(n_bins[1])
    n_bins = pd.qcut(df['Longitude'], q=25, retbins=True)
    bins['Longitude'] = list(n_bins[1])

    with open(f'{getcwd()}/../models/bins.json', 'w') as file:
        dump_file = json.dumps(bins, indent=4)
        file.write(dump_file)

    # Equal frequency binning on scattered features
    categorical_size = [x for x in range(15)]
    df['Host Name'] = pd.qcut(df['Host Name'], q=15, labels=categorical_size)

    categorical_size = [x for x in range(80)]
    df['Host ID'] = pd.qcut(df['Host ID'], q=80, labels=categorical_size)

    categorical_size = [x for x in range(25)]
    df['Latitude'] = pd.qcut(df['Latitude'], q=25, labels=categorical_size)

    categorical_size = [x for x in range(25)]
    df['Longitude'] = pd.qcut(df['Longitude'], q=25, labels=categorical_size)

    return df


def generate_encodings(df: pd.DataFrame) -> None:

    # Unused features and target
    try:
        df = df.drop(['ID', 'Name', 'Price'], axis='columns')
    except KeyError:
        pass

    # Reducing Last Review cardinality
    df['Last Review'] = pd.to_datetime(df['Last Review'])
    df['Last Review'] = df['Last Review'].apply(lambda row: f'{str(row.year)}-{str(row.month)}')

    # Reapplying null values and category type (necessary for adding categories)
    df['Last Review'] = df['Last Review'].astype('category')
    df = df.replace({'Last Review': {'nan-nan': np.nan}})

    # Filling missing values
    df['Last Review'] = df['Last Review'].cat.add_categories('N/A')
    df['Last Review'] = df['Last Review'].fillna('N/A')
    df['Monthly Reviews'] = df['Monthly Reviews'].fillna(0)

    # Discarding rows with missing values (at this point, only Host Name column has)
    df = df.dropna(axis='index', subset='Host Name')

    matches = dict()
    encoder = LabelEncoder()
    for col in df.columns.values:

        # Discretizes only categorical features
        if not is_float_dtype(df[col]) and not is_integer_dtype(df[col]):
            encoder.fit(df[col])

            # Stores each column mapping (label: value)
            col_matches = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            matches[col] = col_matches

    with open(f'{getcwd()}/../models/matches.json', 'w') as json_file:
        # ASCII to deal with distinct data (e.g., japanese characters, emojis, ...)
        matches_file = json.dumps(matches, indent=4, cls=CustomEncoder, ensure_ascii=False)
        json_file.write(matches_file)


def preprocess_instance(instance: pd.DataFrame) -> pd.DataFrame:

    instance = instance.rename(columns={
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

    try:
        instance = instance.drop(['ID', 'Name', 'Price'], axis='columns')
    except KeyError:
        pass

    if not is_integer_dtype(instance['Last Review']):
        instance['Last Review'] = pd.to_datetime(instance['Last Review'])
        instance['Last Review'] = instance['Last Review'].apply(lambda row: f'{str(row.year)}-{str(row.month)}')

    with open(f'{getcwd()}/../models/matches.json', 'r') as encode_file:
        encodes = json.load(encode_file)

    for col in instance.columns.values:
        if not is_float_dtype(instance[col]) and not is_integer_dtype(instance[col]):

            # Gets the associated values to the column keys (-1 if the key isn't known)
            instance[col] = instance[col].apply(lambda x: encodes[col].get(x, -1))

    # Gets the binning threshold file generated in preprocessing
    with open(f'{getcwd()}/../models/bins.json', 'r') as file:
        bins = json.load(file)

    # Binning mapping
    for feature in bins:
        binned_feature = np.digitize(instance[feature], bins[feature])
        instance[feature] = binned_feature - 1

    return instance
