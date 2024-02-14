import plotly.express as px
import streamlit as st
import pandas as pd


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


st.title('Reviews at American Districts Properties')
col1, col2 = st.columns(2)

df = load_dataset('teste_indicium_precificacao')
df = df.drop(['id'], axis='columns')

df['Last Review'] = pd.to_datetime(df['Last Review'])
df = df.sort_values('Last Review')

graph1 = df.groupby(('Neighborhood'))[['Monthly Reviews']].mean().reset_index()
fig = px.bar(graph1, x='Neighborhood', y='Monthly Reviews', color='Neighborhood', title="Average montlhy reviews per district")
col1.plotly_chart(fig, use_container_width=True)

graph2 = df.groupby(['Neighborhood', 'Room Type'])[['Minimun Nights']].mean().reset_index()
fig = px.bar(graph2, x='Neighborhood', y='Minimun Nights', color='Room Type', title="Average minimum nights required")
col2.plotly_chart(fig, use_container_width=True)
