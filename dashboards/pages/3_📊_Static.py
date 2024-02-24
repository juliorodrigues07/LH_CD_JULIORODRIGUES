import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(layout="wide", page_title="Static Dashboard", page_icon=":bar_chart:")

st.title('Static Graphs About NY Properties')
col1, col2 = st.columns(2)

static_dataset = st.session_state['df'].copy()
static_dataset['Last Review'] = pd.to_datetime(static_dataset['Last Review'])
static_dataset = static_dataset.sort_values('Last Review')

col1.header('Reviews and Availability')

graph1 = static_dataset.groupby(['Borough'])[['Monthly Reviews']].mean().reset_index()
fig1 = px.bar(graph1, x='Borough', y='Monthly Reviews', color='Borough', title="Average Montlhy Reviews per Borough")
col1.plotly_chart(fig1, use_container_width=True)

graph2 = static_dataset.groupby(['Borough', 'Room Type'])[['Minimum Nights']].mean().reset_index()
fig2 = px.bar(graph2, x='Borough', y='Minimum Nights', color='Room Type', title="Average Minimum Nights Required")
col1.plotly_chart(fig2, use_container_width=True)

col2.header('Price x Features')

graph3 = static_dataset.groupby(['Minimum Nights'])[['Price']].mean().reset_index()
graph3['Price'] = np.log2(graph3['Price'])
fig3 = px.bar(graph3, x='Minimum Nights', y='Price', title="Average Price by Minimum Nights Required",
              labels={'Price': 'Price (log 2)'})
col2.plotly_chart(fig3, use_container_width=True)

graph4 = static_dataset.groupby(['Days Available'])[['Price']].mean().reset_index()
fig4 = px.bar(graph4, x='Days Available', y='Price', title="Average Price by Availability")
col2.plotly_chart(fig4, use_container_width=True)

st.header('Receipt')

graph5 = static_dataset.groupby(['Room Type'])[['Price']].sum().reset_index()
fig5 = px.pie(graph5, values='Price', names='Room Type', title='Total Receipt by Room Type')
st.plotly_chart(fig5, use_container_width=True)
