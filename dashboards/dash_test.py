import plotly.express as px
from dash import Output
from dash import Input
from dash import Dash
from dash import html
from dash import dcc
import pandas as pd


app = Dash(__name__)
df = pd.read_csv('../datasets/teste_indicium_precificacao.csv')
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
df = df.drop(['id'], axis='columns')

static1 = df.groupby(('Neighborhood'))[['Monthly Reviews']].mean().reset_index()
static2 = df.groupby(['Neighborhood', 'Room Type'])[['Minimun Nights']].mean().reset_index()

app.layout = html.Div([

    html.H1("Rent Pricing at American Districts", style={'text-align': 'center'}),

    html.H3("Interactive Graphs by districts", style={'text-align': 'center'}),

    html.Div([
        'Select districts',
        dcc.Dropdown(id="slct_district",
                     options=[
                         {'label': 'Manhattan', 'value': 'Manhattan'},
                         {'label': 'Brooklyn', 'value': 'Brooklyn'},
                         {'label': 'Bronx', 'value': 'Bronx'},
                         {'label': 'Staten Island', 'value': 'Staten Island'},
                         {'label': 'Queens', 'value': 'Queens'}
                     ],
                     multi=True,
                     value=['Queens'],
                     style={'color': '#000000'}
                     ),
    ], style={'width': "20%", 'margin-bottom': '20px'}),

    html.Div(children=[
        dcc.Graph(id='map1'),
        dcc.Graph(id='map2')
        ],
        style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center'}
    ),

    html.H3("Interactive Graphs by room type", style={'text-align': 'center', 'margin-top': '50px'}),

    html.Div([
        'Select room type',
        dcc.Dropdown(id="slct_roomtype",
                     options=[
                         {'label': 'Entire home/apt', 'value': 'Entire home/apt'},
                         {'label': 'Private room', 'value': 'Private room'},
                         {'label': 'Shared room', 'value': 'Shared room'}
                     ],
                     multi=True,
                     value=['Private room'],
                     style={'color': '#000000'}
                     ),
    ], style={'width': "20%", 'margin-bottom': '20px'}),

    html.Div(children=[
        dcc.Graph(id='map3'),
        dcc.Graph(id='map4')
        ],
        style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center'}
    ),

    html.H3("Static Graphs", style={'text-align': 'center', 'margin-top': '50px'}),

    html.Div(children=[
        dcc.Graph(id='map5', figure=px.bar(static1, x='Neighborhood', y='Monthly Reviews', color='Neighborhood', title="Average montlhy reviews per district")),
        dcc.Graph(id='map6', figure=px.bar(static2, x='Neighborhood', y='Minimun Nights', color='Room Type', title="Average minimum nights required"))
        ],
        style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center'}
    )

], style={'font-family': 'DM Sans, sans-serif', 'padding': '20px', 'color': '#ffffff', 'background-color': '#232323'})


@app.callback(
    [Output(component_id='map1', component_property='figure'),
     Output(component_id='map2', component_property='figure')],
    [Input(component_id='slct_district', component_property='value')]
)
def update_by_district(district_slctd):

    neighbor = df.copy()
    neighbor = neighbor[neighbor["Neighborhood"].isin(district_slctd)]

    graph1 = neighbor.groupby(('Neighborhood'))[['Price']].mean().reset_index()
    fig1 = px.bar(graph1, x='Neighborhood', y='Price', color='Neighborhood', title="Average price per district")

    graph2 = neighbor.groupby(('Neighborhood'))[['Days Available']].mean().reset_index()
    fig2 = px.bar(graph2, x='Neighborhood', y='Days Available', color='Neighborhood',
                  title='Average availability per district')

    return fig1, fig2


@app.callback(
    [Output(component_id='map3', component_property='figure'),
     Output(component_id='map4', component_property='figure')],
    [Input(component_id='slct_roomtype', component_property='value')]
)
def update_by_roomtype(slct_roomtype):

    rooms = df.copy()
    rooms = rooms[rooms["Room Type"].isin(slct_roomtype)]

    graph1 = rooms.groupby(('Room Type'))[['Price']].mean().reset_index()
    fig1 = px.bar(graph1, x='Room Type', y='Price', color='Room Type', title="Average price per district")

    graph2 = rooms.groupby(('Room Type'))[['Days Available']].mean().reset_index()
    fig2 = px.bar(graph2, x='Room Type', y='Days Available', color='Room Type',
                  title='Average availability per district')

    return fig1, fig2


if __name__ == '__main__':
    app.run_server(debug=True)
