import plotly.express as px
import streamlit as st
import pandas as pd


st.set_page_config(layout="wide")


@st.cache_data
def load_dataset(filename: str) -> pd.DataFrame:
    df = pd.read_csv(f'../datasets/{filename}.csv')
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
        'minimo_noites': 'Minimun Nights',
        'numero_de_reviews': 'Reviews',
        'ultima_review': 'Last Review',
        'reviews_por_mes': 'Monthly Reviews',
        'calculado_host_listings_count': 'Number of Listings',
        'disponibilidade_365': "Days Available"
    })
    return df


st.title('Rent Pricing at American Districts')
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

df = load_dataset('pricing')
df = df.drop(['id'], axis='columns')

st.sidebar.image('./indicium.png', width=200)
st.sidebar.title('Interactive Graphs')
st.sidebar.header('Parameters')

districts = st.sidebar.multiselect('Select the district', df["Neighborhood"].unique())
neighbor = df[df["Neighborhood"].isin(districts)]

graph1 = neighbor.groupby(('Neighborhood'))[['Price']].mean().reset_index()
fig = px.bar(graph1, x='Neighborhood', y='Price', color='Neighborhood', title="Average price per district")
col1.plotly_chart(fig, use_container_width=True)

graph2 = neighbor.groupby(('Neighborhood'))[['Days Available']].mean().reset_index()
fig = px.bar(graph2, x='Neighborhood', y='Days Available', color='Neighborhood', title='Average availability per district')
col2.plotly_chart(fig, use_container_width=True)

room_type = st.sidebar.multiselect('Select the room type', df["Room Type"].unique())
rooms = df[df["Room Type"].isin(room_type)]

graph3 = rooms.groupby(('Room Type'))[['Price']].mean().reset_index()
fig = px.bar(graph3, x='Room Type', y='Price', color='Room Type', title="Average price per room type")
col3.plotly_chart(fig, use_container_width=True)

graph4 = rooms.groupby(('Room Type'))[['Days Available']].mean().reset_index()
fig = px.bar(graph4, x='Room Type', y='Days Available', color='Room Type', title="Average availability per room type")
col4.plotly_chart(fig, use_container_width=True)
