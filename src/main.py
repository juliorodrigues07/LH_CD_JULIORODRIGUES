from os.path import isdir
from os import getcwd
from os import chdir
from os import mkdir
import pandas as pd

from datamining.data_visualization import plot_correlation_matrix
from datamining.data_visualization import plot_feature_importance
from datamining.preprocessing import basic_feature_selection
from datamining.preprocessing import fill_missing_values
from datamining.preprocessing import discretize_values
from datamining.preprocessing import feature_selection
from datamining.preprocessing import remove_outliers
from datamining.preprocessing import normalization
from datamining.ml_methods import holdout_split
from datamining.ml_methods import fine_tuning
from datamining.ml_methods import regressions


def load_dataset(filename: str) -> pd.DataFrame:
    df = pd.read_csv(f'../datasets/{filename}.csv')
    df = df.rename(columns={
        'id': 'ID',
        'nome': 'Name',
        'host_id': 'Host ID',
        'host_name': 'Host Name',
        'bairro_group': 'Neighborhood',
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


def grind_dataset(df: pd.DataFrame) -> pd.DataFrame:

    # dataset['Last Review'] = pd.to_datetime(dataset['Last Review'])
    # dataset['Last Review'] = dataset['Last Review'].apply(lambda row: f'{str(row.year)}-{str(row.month)}')

    df = remove_outliers(df, 'Price')

    categorical_size = [x for x in range(15)]
    df['Host Name'] = pd.qcut(df['Host Name'], q=15, labels=categorical_size)

    categorical_size = [x for x in range(80)]
    df['Host ID'] = pd.qcut(df['Host ID'], q=80, labels=categorical_size)

    categorical_size = [x for x in range(25)]
    df['Latitude'] = pd.qcut(df['Latitude'], q=25, labels=categorical_size)

    categorical_size = [x for x in range(25)]
    df['Longitude'] = pd.qcut(df['Longitude'], q=25, labels=categorical_size)

    df = basic_feature_selection(df=df)
    # df = fill_missing_values(df)

    for col in df.columns.values:
        if df[col].dtypes != int and df[col].dtypes != float:
            df = discretize_values(df=df, column=col)

    # df.to_csv(f'{getcwd()}/../datasets/final.csv', index=False)
    return df


def main():

    dataset = load_dataset('teste_indicium_precificacao')
    # print(dataset.describe().transpose())
    # print(dataset.nunique())
    # print(dataset.isnull().sum())

    # dataset = grind_dataset(dataset)
    attributes = dataset.drop(['Price'], axis='columns')
    classes = dataset['Price']

    for column in attributes.columns:
        attributes = discretize_values(df=attributes, column=column)

    # attributes = normalization(df=attributes)
    # regressions(attributes, classes)
    regressions(attributes, classes)
    # X, x, Y, y = holdout_split(attributes=attributes, classes=classes)
    # importances = xgb_classification(training_attributes=X, test_attributes=x, training_classes=Y, test_classes=y)

    # plot_feature_importance(dataset.drop(['Price'], axis='columns').columns, dataset.drop(['Price'], axis='columns').columns, importances)
    # plot_correlation_matrix(df=dataset, graph_width=8)


if __name__ == '__main__':

    if not isdir(f'{getcwd()}/../plots'):
        mkdir(f'{getcwd()}/../plots')
    if not isdir(f'{getcwd()}/../models'):
        mkdir(f'{getcwd()}/../models')

    main()
