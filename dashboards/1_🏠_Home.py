from warnings import filterwarnings
from datetime import datetime
from typing import Dict
from os import getcwd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_integer_dtype
from pandas.api.types import is_float_dtype
from cerberus import Validator
import streamlit as st
import pandas as pd
import joblib


st.set_page_config(layout="wide", page_title="Rent Pricing", page_icon=":heavy_dollar_sign:")
filterwarnings('ignore', category=FutureWarning)


@st.cache_data
def load_model(model_name: str) -> any:

    try:
        model = joblib.load(f'{getcwd()}/../models/{model_name}_model.pkl')
    except (IsADirectoryError, NotADirectoryError, FileExistsError, FileNotFoundError):
        print("Model not found or doesn't exists!")
        exit()

    return model


@st.cache_data
def load_dataset(filename: str) -> pd.DataFrame:

    try:
        df = pd.read_csv(f'{getcwd()}/../datasets/{filename}.csv')
    except (IsADirectoryError, NotADirectoryError, FileExistsError, FileNotFoundError):
        print("Dataset not found or doesn't exists!")
        exit()

    df = df.rename(columns={
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


if 'df' not in st.session_state:
    st.session_state['df'] = load_dataset('pricing')


def validate_input(input_data: Dict) -> (bool, Dict):

    schema = {
        'id': {'type': 'integer', 'required': False, 'empty': True},
        'prop_name': {'type': 'string', 'required': False, 'empty': True},
        'host_id': {'type': 'integer', 'required': True, 'empty': False},
        'host_name': {'type': 'string', 'required': True, 'empty': False},
        'borough': {'type': 'string', 'allowed': list(st.session_state['df']['Neighborhood'].unique()),
                    'required': True, 'empty': False},
        'district': {'type': 'string', 'allowed': list(st.session_state['df']['District'].unique()),
                     'required': True, 'empty': False},
        'latitude': {'type': 'float', 'min': -90.0, 'max': 90.0, 'required': True, 'empty': False},
        'longitude': {'type': 'float', 'min': -180.0, 'max': 180.0, 'required': True, 'empty': False},
        'room_type': {'type': 'string', 'allowed': list(st.session_state['df']['Room Type'].unique()),
                      'required': True, 'empty': False},
        'min_nights': {'type': 'integer', 'min': 1, 'required': True, 'empty': False},
        'reviews': {'type': 'integer', 'min': 0, 'required': False, 'empty': False},
        'last_review': {'type': 'date', 'nullable': True, 'required': False, 'empty': True},
        'monthly_reviews': {'type': 'float', 'min': 0.0, 'required': True, 'empty': False},
        'host_listings': {'type': 'integer', 'min': 1, 'required': True, 'empty': False},
        'availability': {'type': 'integer', 'min': 0, 'max': 365, 'required': True, 'empty': False},
        'model_name': {'type': 'string', 'allowed': ['LightGBM', 'XGBoost', 'HistGradientBoosting'],
                       'required': True, 'empty': False}
    }
    input_validator = Validator(schema)

    if input_validator.validate(input_data) is False:
        print(input_validator.errors)

    return input_validator.validate(input_data), input_validator.errors


def discretize_values(df: pd.DataFrame, column: str) -> pd.DataFrame:

    filterwarnings('ignore')

    # Encodes entire columns of categorical data
    encoding = LabelEncoder()

    encoding.fit(df[column])
    df[column] = encoding.transform(df[column])

    return df


def predict_instance(input_data: Dict, algorithm: str) -> float:

    input_data.pop('model_name')
    instance = pd.DataFrame([input_data])
    instance = instance.rename(columns={
        'id': 'ID',
        'prop_name': 'Name',
        'host_id': 'Host ID',
        'host_name': 'Host Name',
        'borough': 'Neighborhood',
        'district': 'District',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'room_type': 'Room Type',
        'min_nights': 'Minimum Nights',
        'reviews': 'Reviews',
        'last_review': 'Last Review',
        'monthly_reviews': 'Monthly Reviews',
        'host_listings': 'Number of Listings',
        'availability': 'Days Available'
    })

    # Preprocessing and discretization
    instance = instance.drop(['ID', 'Name'], axis='columns')
    instance['Last Review'] = pd.to_datetime(instance['Last Review'])
    instance['Last Review'] = instance['Last Review'].apply(lambda row: f'{str(row.year)}-{str(row.month)}')

    for col in instance.columns.values:
        if not is_float_dtype(instance[col]) and not is_integer_dtype(instance[col]):
            instance = discretize_values(instance, col)

    # Gets the binning threshold file generated in preprocessing
    with open(f'{getcwd()}/../models/bins.json', 'r') as file:
        bins = json.load(file)

    # Binning mapping
    for feature in bins:
        binned_feature = np.digitize(instance[feature], bins[feature])
        instance[feature] = binned_feature

    # Prediction
    model = joblib.load(f'{getcwd()}/../models/{algorithm}_model.pkl')
    prediction = model.predict(instance)
    price = round(prediction[0], 2)

    return price


if __name__ == '__main__':

    try:
        st.sidebar.image('../assets/stock.png', width=280)
    except (IsADirectoryError, NotADirectoryError, FileExistsError, FileNotFoundError):
        print("Image not found or doesn't exists!")
        exit()

    # Property form
    form1 = st.form(key='options')
    form1.title('NY Price Regressor')
    form1.header('Property Specifications')
    col1, col2 = form1.columns(2)

    # Inputs, buttons, select boxes, slider and warnings
    prop_id = col1.number_input(label="Property ID", value=1, placeholder='Insert the number here...')
    host_id = col1.number_input(label="Host ID", value=1, placeholder='Insert the number here...')

    latitude = col1.number_input(label='Latitude', value=0.000000, placeholder='Insert the number here...')
    latitude_warning = col1.container()

    longitude = col1.number_input(label='Longitude', value=0.000000, placeholder='Insert the number here...')
    longitude_warning = col1.container()

    min_nights = col1.number_input(label='Minimum Nights', value=1, placeholder='Insert the number here...')
    minnights_warning = col1.container()

    reviews = col1.number_input(label='Number of Reviews', value=0, placeholder='Insert the number here...')
    reviews_warning = col1.container()

    monthly_reviews = col1.number_input(label='Monthly Reviews Rate', value=0.0, placeholder='Insert the number here...')
    monthlyreviews_warning = col1.container()

    prop_name = col2.text_input(label='Property Name', placeholder='Insert the name here...').strip()

    host_name = col2.text_input(label='Host Name', placeholder='Insert the name here...').strip()
    hostname_warning = col2.container()

    borough = col2.selectbox(label='Select the Borough', options=st.session_state['df']['Neighborhood'].unique())
    district = col2.selectbox(label='Select the District', options=st.session_state['df']['District'].unique())
    room_type = col2.selectbox(label='Select the Room Type', options=st.session_state['df']['Room Type'].unique())
    last_review = col2.date_input(label='Last Review Date', value=None, max_value=datetime.now(), format="DD/MM/YYYY")

    host_listings = col2.number_input(label='Listings per Host', value=1, placeholder='Insert the number here...')
    hostlistings_warning = col2.container()

    availability = form1.slider(label='Days Available per Year', min_value=0, max_value=365, value=70)

    algorithm = form1.selectbox(label='Select the ML Algorithm', options=['LightGBM', 'XGBoost', 'HistGradientBoosting'])
    submit_button = form1.form_submit_button('Predict')
    prediction = st.container()

    input_data = {
        'id': prop_id,
        'prop_name': prop_name,
        'host_id': host_id,
        'host_name': host_name,
        'borough': borough,
        'district': district,
        'latitude': latitude,
        'longitude': longitude,
        'room_type': room_type,
        'min_nights': min_nights,
        'reviews': reviews,
        'last_review': last_review,
        'monthly_reviews': monthly_reviews,
        'host_listings': host_listings,
        'availability': availability,
        'model_name': algorithm
    }

    model_name = str()
    match algorithm:
        case 'LightGBM':
            model_name = 'lgbm'
        case 'XGBoost':
            model_name = 'xgb'
        case 'HistGradientBoosting':
            model_name = 'histgb'
        case _:
            print("ML model not available or doesn't exists")
            exit()

    check, errors = validate_input(input_data=input_data)
    if submit_button is True and check is True:
        print(input_data)
        price = predict_instance(input_data=input_data, algorithm=model_name)
        prediction.success(f'Price: US$ {price}')

    elif submit_button is True and check is False:
        for key in input_data.keys():
            if key in errors.keys():
                match key:
                    case 'latitude':
                        latitude_warning.error('Latitude values must be between -90 and 90!')
                    case 'longitude':
                        longitude_warning.error('Longitude values must be between -180 and 180!')
                    case 'min_nights':
                        minnights_warning.error('Minimum nights must be greater than 0!')
                    case 'reviews':
                        reviews_warning.error('Number of reviews must not be negative!')
                    case 'monthly_reviews':
                        monthlyreviews_warning.error('Monthly reviews rate must not be negative!')
                    case 'host_name':
                        hostname_warning.error('Host name cannot be empty!')
                    case 'host_listings':
                        hostlistings_warning.error('Number of listings per host must be greater than 0!')
                    case _:
                        print("Data doesn't exist!")
