import plotly.express as px
import streamlit as st


st.set_page_config(layout="wide", page_title="Interactive Dashboard", page_icon=":chart_with_upwards_trend:")

st.title('Rent Pricing at NY Boroughs')
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

st.sidebar.title('Interactive Graphs')
st.sidebar.header('Parameters')

interactive_dataset = st.session_state['df'].copy()

boroughs = st.sidebar.multiselect('Select the Borough', interactive_dataset["Borough"].unique())
neighbor = interactive_dataset[interactive_dataset["Borough"].isin(boroughs)]

graph1 = neighbor.groupby('Borough')[['Price']].mean().reset_index()
fig = px.bar(graph1, x='Borough', y='Price', color='Borough', title="Average Price per Borough")
col1.plotly_chart(fig, use_container_width=True)

graph2 = neighbor.groupby('Borough')[['Days Available']].mean().reset_index()
fig = px.bar(graph2, x='Borough', y='Days Available', color='Borough', title='Average Availability per Borough')
col2.plotly_chart(fig, use_container_width=True)

room_type = st.sidebar.multiselect('Select the Room Type', interactive_dataset["Room Type"].unique())
rooms = interactive_dataset[interactive_dataset["Room Type"].isin(room_type)]

graph3 = rooms.groupby('Room Type')[['Price']].mean().reset_index()
fig = px.bar(graph3, x='Room Type', y='Price', color='Room Type', title="Average Price per Room Type")
col3.plotly_chart(fig, use_container_width=True)

graph4 = rooms.groupby('Room Type')[['Days Available']].mean().reset_index()
fig = px.bar(graph4, x='Room Type', y='Days Available', color='Room Type', title="Average Availability per Room Type")
col4.plotly_chart(fig, use_container_width=True)
